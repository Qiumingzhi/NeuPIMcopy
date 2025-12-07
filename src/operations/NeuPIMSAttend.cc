#include "NeuPIMSAttend.h"

/*
从StageProgram.cc里面调用的NeuPIMSAttend
NeuPIMSAttend 的 inputs 向量包含两部分数据：

Attention Scores (Logits):
来源：inputs = get_outputs(logit_softmax, mha_pim_inputs); (StageProgram.cc 第
123 行) 这是上一步 NeuPIMSLogitSoftmax 的输出，即 Query 和 Key 计算并经过
Softmax 后的注意力分数。


Value Cache (Values):
来源：inputs.insert(inputs.end(), values.begin(), values.end());
(StageProgram.cc 第 126 行) 这是直接从 KV Cache 中获取的 Value 张量。

总结： NeuPIMSAttend 接收的数据 = [Softmax后的分数, Value向量]。
这符合 Attention 机制的计算公式 $Attention(Q, K, V) =
Softmax(\frac{QK^T}{\sqrt{d_k}})V$， 其中 NeuPIMSAttend 负责最后一步乘上 $V$
的操作。

*/

NeuPIMSAttend::NeuPIMSAttend(std::string name) : Operation(name) {}
// L3 (构造函数)：“我是谁？”（我叫 NeuPIMSAttend，名字是 Layer0...）

/*当你创建一个 NeuPIMSAttend 对象时（比如 new
NeuPIMSAttend("Layer0.Attn")），这个名字 "Layer0.Attn" 就被传进去了，并保存在
_name变量里。*/

// 只要定义为 std::vector<Ptr<NPUTensor>>（或者 std::vector<Ptr<BTensor>> 等），
// 它本质上就是一个“存放张量指针的列表”。

std::vector<Ptr<BTensor>>
NeuPIMSAttend::get_outputs(std::vector<Ptr<BTensor>> inputs) {
  // 构建一个函数：L13
  // (get_outputs)：“我要干什么？”（给我这些输入，我给你算出那些输出。）

  set_as_parent_tensor(inputs);

  _inputs = inputs;

  _batch_size =
      inputs.size() / 2; // inputs.size() 它是 std::vector 类的一个内置方法。
  // 由C++定义, 返回的是传入的张量总数 切分为两部分 一部分是logits 另一部分是vs
  uint32_t i = 0;

  for (auto tensor : inputs) {
    if (i < _batch_size) {
      _logits.push_back(std::static_pointer_cast<NPUTensor>(tensor));
    } else {
      _vs.push_back(std::static_pointer_cast<PIMTensor>(tensor));
    }
    i++;
  } // 填满logits和vs

  _outputs.resize(_batch_size);

  _nh = _vs[0]->get_dims()[0]; //  vs= [h, seq_len, dk]
  _dk = _vs[0]->get_dims()[2];

  // assert(inputs.size() == 2);
  for (int i = 0; i < _batch_size; ++i) {
    auto L = _logits[i]; // [h, l, seq_len] // l must be seq_len or 1
    // l = seq_len：Prefill Phase 我们需要一次性计算输入中所有 Token 的
    // Attention
    //  此时 Query 的长度等于输入的长度
    // l = 1:  Decoding Phase
    // 我们只需要计算最新生成的那个 Token 对之前所有 Token 的 Attention。此时
    // Query 只有一个 Token。
    auto V = _vs[i]; // [h, seq_len, dk]

    // spdlog::info("(NeuPIMSAttend) L: {}, V: {}", L->get_dims(),
    // V->get_dims()); seq_len of L == seq_len of V
    assert(L->get_dims()[2] == V->get_dims()[1]);
    // nh of L == nh of V
    assert(L->get_dims()[0] == V->get_dims()[0]);

    uint32_t l = L->get_dims()[1];
    std::vector<uint32_t> attend_output_dim{_nh, l, _dk};

    _outputs[i] = std::make_shared<NPUTensor>(
        _name + "_output", attend_output_dim, NPUTensorBufType::ACT, false);
  }

  // todo tiling and instruction initialization.
  calculate_loops(); //计算需要的sram空间
  initialize_tiles();

  spdlog::info("output dim (batch size): {}", _batch_size);

  return _outputs;
}

void NeuPIMSAttend::initialize_tiles() {
  int num_npu_tiles =
      _req_idxs
          .size(); //在caculate_loops里面已经分过了 满足于spad缓冲大小的分块
  int prev_idx = 0;
  for (int i = 0; i < num_npu_tiles; i++) {
    int req_idx = _req_idxs[i];
    if (i == num_npu_tiles - 1)
      assert(req_idx == _batch_size - 1);

    _tiles.push_back(initialize_instructions(prev_idx, req_idx));
    // _tiles是所有算子（Operation）共有的属性，用于存放该算子拆解后的具体执行单元。
    prev_idx = req_idx;
  }
}

// 这段代码的主要作用是为 Attention 操作生成底层的指令序列
// 以刚刚的分块序列逐块的进行生成操作序列
Tile NeuPIMSAttend::initialize_instructions(int start, int end) {
  // start = prev_idx， end = req_idx
  auto tile = Tile{
      .status = Tile::Status::INITIALIZED, // 定义在common.h里面 状态机初始化
      .optype = get_name(),                //操作类型
      .operation_id = _id,                 // 操作ID
      .batch = 0,
      .K = 0,
      .accum = false,
  }; // 初始化一个 Tile 对象用于存放指令

  for (int i = start; i < end + 1; ++i) { // 遍历 Batch
    auto logit = _logits[i]; // Attention Score (Query * Key 的结果)
    auto value = _vs[i];     // Value 矩阵

    uint32_t seq_len = value->get_dims()[1]; // Value 矩阵的 seq_len
    uint32_t ch = value->get_channel();      // Value 矩阵的 channel
    uint32_t chunks =
        ceil((double)seq_len / _page_size); // Value 矩阵的 chunk 数
    // spdlog::info("seq_len: {}", seq_len);

    // 这段代码是用于prefill阶段的，但是貌似整个项目只服务于解码阶段
    if (logit->get_dims()[1] !=
        1) { // 如果 logit 的 seq_len 不为 1，即处于prefill阶段
      // spdlog::info("logit dim:{}", logit->get_dims());
      // spdlog::info("value dim:{}", value->get_dims());
      assert(logit->get_dims()[1] == seq_len);

      for (int h_idx = 0; h_idx < _nh; h_idx++) {
        std::vector<addr_type> dram_logit_addrs;
        std::vector<addr_type> dram_value_addrs;

        for (int dk_idx = 0; dk_idx < _dk; dk_idx++) {
          for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
            dram_value_addrs.push_back(value->get_addr(
                std::vector<uint32_t>{static_cast<unsigned int>(h_idx),
                                      static_cast<unsigned int>(seq_idx),
                                      static_cast<unsigned int>(dk_idx)}));

            for (int sseq_idx = 0; sseq_idx < seq_len; sseq_idx++) {
              dram_logit_addrs.push_back(
                  logit->get_addr({static_cast<unsigned int>(h_idx),
                                   static_cast<unsigned int>(seq_idx),
                                   static_cast<unsigned int>(sseq_idx)}));
            }
          }
        }
        auto sram_l_entry = allocate_sram_addr(seq_len * seq_len, false);
        auto sram_v_entry = allocate_sram_addr(seq_len * _dk, false);
        auto sram_a_entry = allocate_sram_addr(seq_len * _dk, true);
        // -- load --
        // MOVIN logit, value
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_l_entry.first,
            .size = sram_l_entry.second,
            .src_addrs = dram_logit_addrs,
            .operand_id = _INPUT_OPERAND // logit
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_v_entry.first,
            .size = sram_v_entry.second,
            .src_addrs = dram_value_addrs,
            .operand_id = _INPUT_OPERAND // logit
        });

        // -- compute --
        // GEMM (l*v -> a)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_a_entry.first,
            .size = sram_a_entry.second,
            .src_addrs =
                std::vector<addr_type>{sram_l_entry.first, sram_v_entry.first},

            .tile_m = _dk,
            .tile_k = seq_len,
            .tile_n = seq_len,
        });

        // MOVOUT
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_a_entry.first,
            .size = sram_a_entry.second,
            .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                             ->_inners[h_idx]
                             ->get_all_addrs(),
            .operand_id = _OUTPUT_OPERAND,
        });
      }

      continue;
    }

    // 这段代码是用于decode阶段的
    for (int hi = 0; hi < _nh; hi++) {
      std::map<uint32_t, std::vector<addr_type>> sram_readres_addrs;
      for (int ci = 0; ci < chunks; ci++) {
        uint64_t logit_row = 0; // FIXME: decode row index from dram address
        uint64_t p_header_addr =
            AddressConfig::encode_pim_header(ch, logit_row, true, 0, 0);

        addr_type sram_addr_gw = allocate_sram_addr(0, false).first;

        // GWRITE (channel, bank, row)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::PIM_GWRITE,
            .dest_addr = sram_addr_gw,
            .size = 0,
            .src_addrs =
                std::vector<addr_type>{p_header_addr}, // FIXME: gwrite addr
            .operand_id = _INPUT_OPERAND,
        });

        uint32_t num_comps =
            (ci == chunks - 1 && (seq_len % _page_size) > 0)
                ? ceil((double)(seq_len % _page_size) / _datas_per_comp_cmd)
                : _page_size / _datas_per_comp_cmd;
        uint32_t decoded_num_comps = 1 << LogBase2(num_comps);

        // spdlog::info("num_comps: {}, decoded_num_comps: {}", num_comps,
        // decoded_num_comps);
        if (num_comps > decoded_num_comps) {
          decoded_num_comps *= 2;
        }
        assert(num_comps <= decoded_num_comps);
        assert(num_comps > 0);

        for (int ti = 0; ti < _tiles_per_chunk; ti++) {
          auto sram_entry = allocate_sram_addr(_banks_per_channel, false);
          addr_type sram_addr = sram_entry.first;

          uint32_t DRAM_row = value->_rows[ti * chunks + ci];
          p_header_addr = AddressConfig::encode_pim_header(
              ch, DRAM_row, false, decoded_num_comps, 1);
          // P_HEADER (num_comps, num_readres)
          tile.instructions.push_back(Instruction{
              .opcode = Opcode::PIM_HEADER,
              .dest_addr = sram_addr,
              .size = 0,
              .src_addrs = std::vector<addr_type>{p_header_addr},
              .operand_id = _INPUT_OPERAND,
          });
          std::string cmds = "P_HEADER ";

          uint64_t dram_addr = AddressConfig::encode_pim_comps_readres(
              ch, DRAM_row, num_comps, true);

          if (_config.dram_type == DramType::NEWTON) {
            Instruction comp_inst = Instruction{
                .opcode = Opcode::PIM_COMP,
                .dest_addr = sram_addr,
                .size = 0,
                .src_addrs = std::vector<addr_type>{dram_addr},
                .operand_id = _INPUT_OPERAND,
            };

            for (int j = 0; j < num_comps; j++) {
              // COMP * num_comps (channnel, row)
              tile.instructions.push_back(comp_inst);
              cmds += "COMP ";
            }
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_READRES,
                .dest_addr = sram_addr,
                .size = sram_entry.second,
                .src_addrs = std::vector<addr_type>{dram_addr},
                .operand_id = _INPUT_OPERAND,
            });
            cmds += "READRES ";
          } else {
            tile.instructions.push_back(Instruction{
                .opcode = Opcode::PIM_COMPS_READRES,
                .dest_addr = sram_addr,
                .size = sram_entry.second,
                .src_addrs = std::vector<addr_type>{dram_addr},
                .operand_id = _INPUT_OPERAND,
            });
          }

          if (sram_readres_addrs.find(ti) ==
              sram_readres_addrs.end()) // not exists
            sram_readres_addrs[ti] = std::vector<addr_type>{sram_addr};
          else
            sram_readres_addrs[ti].push_back(sram_addr);
        }
      }
      if (chunks > 1) {
        for (int ti = 0; ti < _tiles_per_chunk; ++ti) {
          assert(sram_readres_addrs[ti].size() == chunks);

          uint32_t column_height = _tiles_per_chunk * _banks_per_channel;
          auto sram_acc_entry = allocate_sram_addr(column_height, true);

          tile.instructions.push_back(Instruction{
              .opcode = Opcode::ADD,
              .dest_addr = sram_acc_entry.first,
              .size = sram_acc_entry.second,
              .src_addrs = sram_readres_addrs[ti],
          });
          tile.instructions.push_back(Instruction{
              .opcode = Opcode::MOVOUT,
              .dest_addr = sram_acc_entry.first,
              .size = sram_acc_entry.second,
              .src_addrs = std::static_pointer_cast<NPUTensor>(_outputs[i])
                               ->_inners[hi]
                               ->get_all_addrs(),
              .operand_id = _OUTPUT_OPERAND,
          });
        }
      }
    }
  }
  // spdlog::info("tile size: {}", tile.instructions.size());
  return tile;
}

void NeuPIMSAttend::calculate_loops() {

  //用于计算所需的PIM和NPU注意力计算通信所需的SRAM大小，然后确定一个请求打包的序列

  // assert(sram_size_needed() < _config.spad_size KB / 2);

  uint32_t E = _config.model_n_embd / _config.n_tp;

  // memory spec
  _page_size = _config.dram_page_size / _config.precision;
  _banks_per_channel = _config.dram_banks_per_ch;

  _tiles_per_chunk = ceil((double)_dk / _banks_per_channel); // 不是很懂
  _datas_per_comp_cmd = _config.pim_comp_coverage; //一次可以比较16个

  // npu tiling
  int heads_per_dram_page =
      floor((double)_page_size / _dk); // 512/128=4 一个page 能放4个头
  int heads_space_in_page = heads_per_dram_page * _dk; // 4*128=512
  int chunks = ceil((double)E / heads_space_in_page);  // E = 1024
  // chunks=1024/512=2 需要2个page

  int sram_needs = 0;
  for (int i = 0; i < _batch_size; ++i) {
    auto L = _logits[i]; // [h, l, seq_len] // l must be seq_len or 1
    auto V = _vs[i];     // [h, seq_len, dk]

    uint32_t q_len = L->get_dims()[1];
    uint32_t seq_len = V->get_dims()[2];

    int need_sram_for_req = 0;

    if (q_len == 1) {
      // incremental phase 解码阶段
      need_sram_for_req = (seq_len + chunks * _dk) * _nh * _config.precision;
      // seq_len*nh  从PIM输入到NPU
      /*

      含义：存储 Attention Scores (Logits) 的空间。
      物理对应：
      在计算 $Q \times K^T$ 后，我们会得到一个形状为 [Batch=1, Heads, Seq_Len]
      的分数矩阵。 为了执行 Softmax
      操作（需要对整行的分数进行归一化），通常需要将这 seq_len 个分数都暂存在
      SRAM 中。 计算：每个 Head 有 seq_len 个分数，共有 _nh 个 Head。


      如果是1024序列 需要的存储是16KB
      */

      // chunks * _dk * _nh (Output Buffer) 从NPU输出到PIM
      /*
      物理对应：
      这是最终 $Attention \times V$ 的结果，形状为 [Batch=1, Heads, Head_Dim]。
      _dk * _nh 等于总的 Embedding 维度 E（在当前 TP 卡上）。
      */

      sram_needs += need_sram_for_req;
    } else {
      // initiation phase
      assert(false);
      // now support only incremental phases  NeuPIMs不支持prefill过程
      // SRAM不负责这里
    }

    if (sram_needs > _config.spad_size KB / _config.precision) {
      assert(i > 0);
      // 如果 i=0 时就超了，说明连第 1
      // 个请求都装不下，这是硬件配置错误或模型太大，程序直接报错。

      _req_idxs.push_back(i - 1);
      // 动作 1：记录切分点
      // 既然加上第 i 个请求会爆内存，那我们就把前一个请求（i-1）作为当前 Tile
      // 的终点。 也就是说，当前 Tile 包含从 [start, i-1] 的请求。 push_back 是
      // C++ 标准库容器 std::vector（动态数组）的一个成员函数。
      // 它用于在数组的末尾添加一个新元素。

      sram_needs = need_sram_for_req;
      // 动作 2：开启新的 Tile
      // 第 i 个请求因为没挤进上一个 Tile，所以它成为下一个 Tile 的第 1 个请求。
      // sram_needs 重置为第 i 个请求所需的大小。
    }
  } // 这段代码存在逻辑漏洞。它只处理了“累加和超过上限”的情况，没有处理“单个元素超过上限”的情况。
  _req_idxs.push_back(_batch_size - 1);
}

uint32_t NeuPIMSAttend::sram_size_needed() {
  /// space for gemvadd activation = dk * batch_size?
  uint32_t need_size =
      _batch_size * _config.model_n_head * _dk * _config.precision;

  return 0; // need_size;
}
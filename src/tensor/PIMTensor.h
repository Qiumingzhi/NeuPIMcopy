#pragma once

#include "BTensor.h"

// PIM Tensor Key/Value 类型枚举
// 指示该 Tensor 是 Attention 机制中的 Key 还是
// Value，这对内存布局有决定性影响。
enum class PIMTensorKVType { KEY, VALUE };

// PIM 专用张量类 (PIMTensor)
// 继承自 BTensor (Base Tensor)。专门用于 NeuPIMs 架构下的张量管理。
// 与 NPUTensor 不同，PIMTensor 不使用线性地址空间，而是基于 Channel 和 Row
// 的物理资源分配。
class PIMTensor : public BTensor {
public:
  PIMTensor() = default;
  PIMTensor(std::string name, uint32_t ch, std::vector<uint32_t> dims,
            PIMTensorKVType kv_type, bool produced);
  ~PIMTensor() = default;

  // 获取特定索引的地址（在 PIM 模式下通常返回 0
  // 或不适用，因为地址是隐式的或通过指令传递）
  virtual addr_type get_addr(std::vector<uint32_t> indexes) override;

  // 获取所有分配的地址列表
  virtual std::vector<addr_type> get_all_addrs() override;

  // 增加 Token
  // 当推理生成新的 Token 时调用。
  // 如果当前分配的行空间不足以容纳新的 seq_len，会自动从 KVCacheAlloc 申请新的
  // DRAM 行。
  virtual void add_token() override; // automatically allocates buffer each time
                                     // a token is added during iteration.

  // 获取已分配空间能容纳的 Sequence Length 上限
  uint32_t get_allocated_seq_len();

  // 获取当前占用的 DRAM 行数
  uint32_t get_num_rows();

  // 获取改 Tensor 所在的 DRAM Channel ID
  uint32_t get_channel();

  // 获取所有分配的 DRAM 行索引列表
  std::vector<uint64_t> get_rows();

  PIMTensorKVType _kv_type; // Key 或 Value 类型
  uint32_t _bank_per_ch; // 每个 Channel 的 Bank 数量（影响跨 Bank 并行度）
  uint32_t _E;           // Embedding 维度大小
  uint32_t _num_ele_per_row; // 每行 DRAM 能存储的元素个数

  // for here, row means DRAM row
  // how many rows to allocate at once when additional allocation is needed due
  // to increased seq_len. 每次分配的行数块大小。 当 seq_len
  // 增长导致空间不足时，一次性申请的行数。 对于 Key 和
  // Value，这个计算逻辑不同（取决于布局策略）。
  uint32_t _num_rows_per_alloc;

  uint32_t _ch;                // DRAM channel (绑定的 Channel ID)
  std::vector<uint64_t> _rows; // store the row index allocated from KVCache.
                               // (存储从 KVCacheAlloc 申请到的行索引)
  uint32_t _seq_len;           // 当前实际存储的 Sequence Length
};
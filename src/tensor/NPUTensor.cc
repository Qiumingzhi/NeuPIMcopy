#include "NPUTensor.h"

#include <memory>

#include "NPUTensor2D.h"

/**
 * NPUTensor with NPUTensor2D: Tensor object for weight, activation
 * Note that NPUTensorKV should be initialized with kv_type, not buf_type
 *  Weight:
 *      buf_type: NPUTensorBufType::WGT
 *      dimension: 2D (Fully connected)
 *
 *  Activation:
 *      buf_type: NPUTensorBufType::ACT
 *      dimension: 3D (including batch?)
 */

/*举几个dim例子：

标量 (Scalar):  dims = {} (空，或视情况而定)

向量 (1D Tensor): 如果有一个长度为 10 的向量。
dims就是 [10]。
dims.size() 是 1（表示它是 1 维的）。
dims[0] 是 10（表示第 0 维的长度是 10）。

矩阵 (2D Tensor - 比如一张 64x64 的图片):
dims就是 [64, 64]。
dims.size() 是 2。
dims[0] 是 64（高），dims[1] 是 64（宽）。

3D 张量 (比如 Batch 为 2，大小 64x64):
dims就是 [2, 64, 64]。
dims.size() 是 3。dims[0]=2, dims[1]=64, dims[2]=64。*/

NPUTensor::NPUTensor(std::string name, std::vector<uint32_t> dims,
                     NPUTensorBufType buf_type, bool produced) {
  ast(buf_type != NPUTensorBufType::KV);

  _id = generate_id();
  _name = name;
  _dims = dims;
  _produced = produced;
  _precision = Config::global_config.precision;

  uint32_t num_inners = 1;
  std::vector<uint32_t> inner_dims = dims;
  if (dims.size() == 3) {
    num_inners = dims[0]; // 第一维作为拆分数量（Batch 或 Head 数）
    inner_dims = slice(dims, 1, -1); // 剩余维度作为内层2D张量的维度
  }
  for (int i = 0; i < num_inners; ++i) {
    _inners.push_back(std::make_shared<NPUTensor2D>(inner_dims, buf_type));
  }

  /*处理 3D 张量：
    如果传入的是 3D 张量（例如 [Batch, M, K] 或 [NumHeads, M, K]），模拟器认为
    NPU（脉动阵列）一次只能处理 2D 矩阵乘法。
    因此，它将第一维（dims[0]）剥离出来，视为 num_inners 个独立的 2D 张量。
    例如：一个 [2, 64, 64] 的张量会被创建为 2 个 [64, 64] 的 NPUTensor2D对象，
        存放在 _inners 数组中。

    通过_inners[0] _inners[1] 调用这两个2D张量

    处理 2D 张量： 如果传入的是 2D 张量，则 num_inners 保持为 1。
    _inners 数组中只包含 1 个 NPUTensor2D 对象。

    */

  _is_transposed = false;
}

/**
 * NPUTensor with NPUTensorKV: Tensor object for Key and Value
 *  Key:
 *      kv_type: NPUTensorKVType::Key
 *      dimension: 3D (nh,dk,T)
 *
 *  Value:
 *      kv_type: NPUTensorKVType::Value
 *      dimension: 3D (nh,T,dk)
 */
NPUTensor::NPUTensor(std::string name, std::vector<uint32_t> dims,
                     NPUTensorKVType kv_type, bool produced) {
  _id = generate_id();
  _name = name;
  _dims = dims;
  _produced = produced;
  _precision = Config::global_config.precision;

  uint32_t num_inners = 1;
  std::vector<uint32_t> inner_dims = dims;

  // XXX: can dims size be 2, without num_heads?
  if (dims.size() == 2) {
    ast(0);
  }
  if (dims.size() == 3) {
    num_inners = dims[0]; // num_heads
    inner_dims = slice(dims, 1, -1);
  }
  for (int i = 0; i < num_inners; ++i) {
    _inners.push_back(std::make_shared<NPUTensorKV>(inner_dims, kv_type));
  }
}

NPUTensor::NPUTensor(std::string name, Ptr<NPUTensor2D> tensor, bool produced) {
  _id = generate_id();
  _name = name;
  _dims = tensor->_dims;
  _produced = produced;
  _precision = Config::global_config.precision;
  _inners = {tensor};
}

void NPUTensor::set_transposed() {
  assert(_inners[0]->_buf_type == NPUTensorBufType::ACT ||
         _inners[0]->_buf_type == NPUTensorBufType::WGT);
  _is_transposed = true;
}

void NPUTensor::unset_transposed() {
  assert(_inners[0]->_buf_type == NPUTensorBufType::ACT ||
         _inners[0]->_buf_type == NPUTensorBufType::WGT);
  _is_transposed = false;
}

std::vector<uint32_t> NPUTensor::get_dims() {
  if (_is_transposed) {
    std::vector<uint32_t> ret(_dims.size());
    std::reverse_copy(_dims.begin(), _dims.end(), ret.begin());
    return ret;
  }
  return _dims;
}

addr_type NPUTensor::get_addr(std::vector<uint32_t> indexes) {

  // 输入：接受一个 indexes 向量，表示要访问的元素的多维坐标。

  // 这个函数目前的实际功能仅仅是作为一个安全检查器：
  // 它验证索引是否越界，如果在界内则返回0，越界则返回错误码。
  // 真正复杂的层级化地址计算逻辑被暂时屏蔽了。

  // spdlog::info("(NPUTensor::get_addr) indexes:{}, inners.size:{}", indexes,
  //              _inners.size());
  // spdlog::info("_inners[0]->dims.size:{}", _inners[0]->_dims.size());
  // spdlog::info("_inners.size() + _inners[0]->_dims.size(): {}",
  //              _inners.size() + _inners[0]->_dims.size());
  // int idx_size = indexes.size();
  // ast(_inners.size() > 0);
  // ast(_dims.size() == idx_size);
  // ast(0 <= idx_size && idx_size <= 3);

  ast(_dims.size() == indexes.size());

  std::vector<uint32_t> dims(_dims.begin(), _dims.end());
  if (_is_transposed) {
    std::copy(_dims.rbegin(), _dims.rend(), dims.begin());
  }
  // 复制维度：创建一个局部变量 dims复制当前的维度信息。
  // 处理转置：如果 Tensor 被标记为转置（_is_transposed 为 true），
  // 代码会将维度信息反转。这意味着后续的边界检查是基于转置后的逻辑形状进行的。

  for (size_t i = 0; i < dims.size(); ++i) {
    if (indexes[i] >= dims[i]) {
      return GARBAGE_ADDR;
    }
  }

  return 0;

  if (indexes.size() <= 2) { // bias, wgt
    return _inners[0]->get_addr(indexes);
  }
  return _inners[indexes[0]]->get_addr(slice(indexes, 1, -1));
}

std::vector<addr_type> NPUTensor::get_all_addrs() {
  ast(_inners.size() > 0);
  std::vector<addr_type> res;
  for (int i = 0; i < _inners.size(); i++) {
    auto addrs = _inners[i]->get_all_addrs();
    for (auto addr : addrs) {
      res.push_back(addr);
    }
  }
  return res;
}

void NPUTensor::add_token() {
  for (auto inner : _inners) {
    std::static_pointer_cast<NPUTensorKV>(inner)->add_token();
  }
}

// get_row_addrs: row_idx -> [addr]
// Used when invoking a 2D tensor by 1D row units in LayerNorm,
// or when invoking a 3D tensor by 1D row units in Softmax.
// Should only be used when inner is NPUTensor2D.
std::vector<addr_type> NPUTensor::get_row_addrs(uint32_t row_idx) {
  // ast(_inners.size() == 1);
  // ast(_dims.size() == 2);
  if (_dims.size() == 2) {
    // ln
    return std::static_pointer_cast<NPUTensor2D>(_inners[0])
        ->get_row_addrs(row_idx);
  } else if (_dims.size() == 3) {
    // Softmax
    auto l = _dims[1];
    return std::static_pointer_cast<NPUTensor2D>(_inners[row_idx / l])
        ->get_row_addrs(row_idx % l);
  }
  ast(0);
}

std::vector<Ptr<NPUTensor>>
NPUTensor::split_by_row(std::vector<uint32_t> row_dims) {
  ast(_inners.size() == 1);
  ast(_dims.size() == 2);
  ast(_inners[0]->_buf_type == NPUTensorBufType::ACT);

  std::vector<Ptr<NPUTensor>> ret;
  Ptr<NPUTensor2D> inner = std::static_pointer_cast<NPUTensor2D>(_inners[0]);
  std::vector<Ptr<NPUTensor2D>> splited_inners = inner->split_by_row(row_dims);
  int i = 0;
  for (auto inner : splited_inners) {
    std::string new_name = _name + "_" + std::to_string(i);
    Ptr<NPUTensor> tensor = std::make_shared<NPUTensor>(new_name, inner, true);
    ret.push_back(tensor);
  }
  return ret;
}
#include "PIMTensor.h"

#include "../allocator/AddressAllocator.h"

PIMTensor::PIMTensor(std::string name, uint32_t ch, std::vector<uint32_t> dims,
                     PIMTensorKVType kv_type, bool produced) {
  _name = name;
  _ch = ch;
  _dims = dims; // [h, seq_len, d_k] or [h, d_k, seq_len]
  _precision = Config::global_config.precision;
  _produced = produced;
  _kv_type = kv_type;

  auto alloc = KVCacheAlloc::GetInstance();
  // 根据类型确定 Sequence Length，用于计算需要分配多少空间
  _seq_len = kv_type == PIMTensorKVType::KEY ? dims[2] : dims[1];
  _bank_per_ch = alloc->_bank_per_ch;
  _num_ele_per_row = alloc->_num_ele_per_row;
  _E = Config::global_config.model_n_embd;

  uint32_t num_alloc_iter =
      0; // calculate # of allocation iterations based on seq_len.
  if (kv_type == PIMTensorKVType::KEY) {
    // KEY: allocate (E / C) rows
    // Key 矩阵布局策略：为了支持并行比较，数据在 Bank 间条带化（Striping）。
    // 每个 Token 的 Embedding 向量被切分存储在不同 Bank 的同一行中。
    // _num_rows_per_alloc: 存储完整 Embedding 维度所需的一个“行组”的大小。
    _num_rows_per_alloc = ceil((double)_E / (double)_num_ele_per_row);
    // num_alloc_iter: 随着 seq_len 增长，需要分配多少次这样的“行组”。
    // Key 在 seq_len 方向上的增长是按 Bank 数量步进的。
    num_alloc_iter = ceil((double)_seq_len / (double)_bank_per_ch);
  } else {
    // VALUE: allocate (E / bank_per_ch) rows
    // Value 矩阵布局策略：为了支持并行累加，数据在列方向上连续。
    // Value 在 seq_len 方向上的增长是按每行元素个数 (_num_ele_per_row) 步进的。
    _num_rows_per_alloc = ceil((double)_E / (double)_bank_per_ch);
    num_alloc_iter = ceil((double)_seq_len / (double)_num_ele_per_row);
  }

  uint32_t num_required_alloc = num_alloc_iter * _num_rows_per_alloc;

  // 向 KVCacheAlloc 申请指定 Channel 的空闲行
  for (int i = 0; i < num_required_alloc; ++i)
    _rows.push_back(alloc->allocate(ch));
}

addr_type PIMTensor::get_addr(std::vector<uint32_t> indexes) { return 0; }

std::vector<addr_type> PIMTensor::get_all_addrs() {
  std::vector<addr_type> ret;
  return ret;
}

// 计算当前已分配的物理空间能够容纳的最大 Sequence Length
uint32_t PIMTensor::get_allocated_seq_len() {
  if (_kv_type == PIMTensorKVType::KEY)
    // 对于 Key，空间按 Bank 数量为块进行分配
    return ceil((double)_seq_len / (double)_bank_per_ch) * _bank_per_ch;
  else
    // 对于 Value，空间按每行元素数量为块进行分配
    return ceil((double)_seq_len / (double)_num_ele_per_row) * _num_ele_per_row;
}

// 增加 Token 时的动态扩容逻辑
void PIMTensor::add_token() {
  _seq_len++;
  if (_kv_type == PIMTensorKVType::KEY)
    _dims[2]++;
  else
    _dims[1]++;

  // 如果当前 seq_len 还在已分配容量范围内，无需操作
  if (_seq_len <= get_allocated_seq_len())
    return;

  // 否则，需要申请新的 DRAM 行来扩容
  for (int i = 0; i < _num_rows_per_alloc; ++i)
    _rows.push_back(KVCacheAlloc::GetInstance()->allocate(_ch));
}

uint32_t PIMTensor::get_num_rows() { return _rows.size(); }

uint32_t PIMTensor::get_channel() { return _ch; }

std::vector<uint64_t> PIMTensor::get_rows() { return _rows; }
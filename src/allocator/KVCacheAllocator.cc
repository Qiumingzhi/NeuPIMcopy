#include "AddressAllocator.h"

// KVCache分配器实现文件
// 负责在NPU模式和PIM模式下管理KV Cache的内存分配与释放

KVCacheAlloc::KVCacheAlloc()
    : _kv_cache_size(0), _kv_cache_limit(0), _kv_cache_entry_size(0),
      _base_addr(0), _base_row(0) {}

void KVCacheAlloc::init(addr_type base_addr) {
  _mode = Config::global_config.run_mode;
  if (_mode == RunMode::NPU_ONLY) { // NPU only mode
    // NPU模式：初始化NPU内存布局
    init_npu_layout(base_addr);
  } else if (_mode == RunMode::NPU_PIM) {
    // PIM模式：初始化PIM内存布局
    init_pim_layout(base_addr);
  } else {
    ast(0);
  }
}

/**
 * 初始化NPU内存布局。
 * 首先，考虑预确定的缓存大小来分配整个内存。
 * 缓存条目由32个键（Key）和值（Value）组成，大小均为 d_k。
 * 注意：缓存是以 d_k 粒度保存的，而不是 E。
 * 保存的缓存的内存布局为 (h, l, d_k)。
 * 这样做的原因是因为连续内存访问更快，
 * 所以特定头的相邻潜在向量（latent vector）应该能被更快地加载。
 */
void KVCacheAlloc::init_npu_layout(addr_type base_addr) {
  uint32_t max_active_reqs = Config::global_config.max_active_reqs;
  uint32_t max_seq_len = Config::global_config.max_seq_len;
  // h: 每个张量并行(TP)分片的头数
  uint32_t h = Config::global_config.model_n_head / Config::global_config.n_tp;
  // d_k: 每个头的维度大小
  uint32_t d_k =
      Config::global_config.model_n_embd / Config::global_config.model_n_head;
  uint32_t precision = Config::global_config.precision;

  _base_addr = base_addr;
  _kv_cache_entry_size =
      32; // allocate once per seq_len 32 (每次分配针对32个序列长度)
  // 计算总的KV Cache大小: max_active_reqs * max_seq_len * h * d_k * precision
  _kv_cache_size = max_active_reqs * max_seq_len * h * d_k * precision;
  // 确保分配的地址空间不超过HBM大小
  ast(_base_addr + _kv_cache_size < Config::global_config.HBM_size);

  addr_type next_addr = _base_addr;
  // 每个块存储的序列长度数量 / 每个块的序列长度 (一个块包含 32 * d_k 个元素)
  // = HBM中的KV cache块数量
  uint64_t num_kv_cache_entries =
      max_active_reqs * max_seq_len * h / _kv_cache_entry_size;

  // 初始化空闲列表
  for (int i = 0; i < num_kv_cache_entries; ++i) {
    _kv_cache.push_back(next_addr);
    next_addr += _kv_cache_entry_size * d_k *
                 precision; // 增加地址指针: 32 seq_len * d_k * precision
  }
}

void KVCacheAlloc::init_pim_layout(addr_type base_addr) {
  // = DRAM PIM row 中的矩阵行数
  constexpr uint32_t row_per_bank = 32768;
  //字节偏移量 X  // rank bit, bg bit, bank bit, ch bit, col bit = 1 + 2 + 2 + 5
  //+ 10 = 20
  constexpr uint32_t row_offset = 20;
  constexpr uint64_t mask =
      ~((1 << row_offset) - 1); // 掩码: 0x1111(64-21)0000(21), 用于对齐到行起始
  _dram_row_size = Config::global_config.dram_page_size;               // 1024
  _num_ele_per_row = _dram_row_size / Config::global_config.precision; // 512
  _bank_per_ch = Config::global_config.dram_banks_per_ch;
  _dram_channels = Config::global_config.dram_channels;

  base_addr = base_addr & mask; // 获取当前地址所在行的起始地址 (去除低位偏移)
  base_addr =
      base_addr +
      (1 << row_offset); // 移动到下一行的起始地址 (确保从一个新的完整行开始)

  _base_addr = base_addr;
  _base_row = base_addr >> row_offset; // 获取行索引 (去除低位偏移)

  // _rows: channel -> row idx (双端队列，存储每个通道的空闲行索引)
  uint32_t free_rows_size = row_per_bank - _base_row;
  for (int i = 0; i < _dram_channels; ++i) {
    _rows.push_back(
        std::make_shared<std::deque<uint64_t>>()); // 为每个通道创建一个deque
    for (int j = 0; j < free_rows_size; ++j) {
      if (_base_row + j < row_per_bank)
        _rows[i]->push_back(_base_row + j); // 将空闲的行索引加入队列
    }
  }
}

// NPU分配: 分配空间 [bank per ch, d_k], 并返回地址
// 当重复此操作 h 次时，我们要分配的空间为 [h, bank per ch, d_k]
addr_type KVCacheAlloc::allocate() {
  ast(_mode == RunMode::NPU_ONLY);
  ast(_kv_cache.size() > 0);
  addr_type addr = _kv_cache.front();
  _kv_cache.pop_front();
  return addr;
}

// PIM分配: 从指定通道分配一个空闲行
addr_type KVCacheAlloc::allocate(uint64_t ch) {
  ast(_mode == RunMode::NPU_PIM);
  ast(_rows[ch]->size() > 0);
  addr_type row = _rows[ch]->front();
  _rows[ch]->pop_front();
  return row; // return free row (返回空闲行索引)
}

// NPU释放: 释放指定的地址回KV Cache空闲列表
void KVCacheAlloc::free(addr_type addr) {
  ast(_mode == RunMode::NPU_ONLY);
  _kv_cache.push_back(addr);
}

// PIM释放: 释放指定通道的行回空闲列表
void KVCacheAlloc::free(uint32_t ch, uint64_t row) {
  ast(_mode == RunMode::NPU_PIM);
  _rows[ch]->push_back(row);
}
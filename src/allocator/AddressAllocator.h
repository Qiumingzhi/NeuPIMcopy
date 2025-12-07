#pragma once
#include "../Common.h"

// Used in NPU + PIM to allocate weights, and in NPU only to allocate all
// tensors. 权重分配器 (Weight Allocator) 在 "NPU+PIM"
// 混合模式下用于分配权重，在 "NPU Only" 模式下用于分配所有张量。
class WgtAlloc : public Singleton<WgtAlloc> {
private:
  friend class
      Singleton; // WgtAlloc是一个单例模式（Singleton）类这意味着在整个程序运行期间，
                 // WgtAlloc的实例只有一个，它的构造函数
                 // WgtAlloc()也只会被调用一次。当你调用allocate
                 // 时，你是在操作那个已经存在的、唯一的对象。
  WgtAlloc();
  ~WgtAlloc() = default;

public:
  addr_type _base_addr; // 权重分配的基地址
  uint64_t _top_addr;   // 当前已分配到的最高地址 (类似堆指针)

  addr_type allocate(uint64_t size);
  addr_type get_next_aligned_addr();
};

// 激活值分配器 (Activation Allocator)
// 用于分配中间激活值 (Activation) 的内存。
class ActAlloc : public Singleton<ActAlloc> {
private:
  friend class Singleton;
  ActAlloc();
  ~ActAlloc() = default;

public:
  addr_type _base_addr;    // 激活缓冲区的基地址
  addr_type _top_addr;     // 当前分配指针
  uint64_t _act_buf_size;  // 固定大小的激活缓冲区大小
  uint64_t _act_buf_limit; // 缓冲区上限地址 (_base_addr + _act_buf_size)

  void init(addr_type base_addr);
  addr_type allocate(uint64_t size);
  addr_type
  get_next_aligned_addr(); // aligned limit addr + alignment of ActAlloc buf
  void flush();
};

// KV Cache 分配器 (KV Cache Allocator)
// 核心类：负责管理 Key-Value Cache 的内存分配。
// 针对 NPU 和 PIM 有两种完全不同的布局策略。
class KVCacheAlloc : public Singleton<KVCacheAlloc> {
private:
  friend class Singleton;
  KVCacheAlloc();
  ~KVCacheAlloc() = default;

public:
  RunMode _mode;
  addr_type _base_addr;

  // for NPU layout (NPU 模式下的布局)
  // NPU 倾向于连续突发访问，因此采用线性分配，将 dim_k 维度的元素连续存储。
  uint64_t _kv_cache_size;  // fixed (固定的 KV Cache 总大小)
  uint64_t _kv_cache_limit; // _base_addr + _kv_cache_size (地址上限)
  uint64_t _kv_cache_entry_size; // 32 (bank per ch) - 这里的 32 是分配粒度（以
                                 // seq_len 为单位）
  std::deque<addr_type>
      _kv_cache; // base_addr of each entry (记录每个空闲块的基地址)

  // for PIM layout (PIM 模式下的布局)
  // PIM 利用 Bank 并行性，因此采用基于 Channel 和 Row 的分配策略。
  uint64_t _dram_channels;
  uint64_t _base_row;      // _base_addr >> row_offset (起始行索引)
  uint32_t _dram_row_size; // DRAM row size (1024KB) (行大小)
  uint32_t _num_ele_per_row; // DRAM row size / precision (每行能存多少个元素)
  uint32_t _bank_per_ch; // 每个 Channel 的 Bank 数量

  // channel -> free rows base index
  // 这是一个二维结构，第一维是 Channel，第二维是该 Channel
  // 下可用的空闲行索引列表。 PIMTensor 会根据 Channel ID 向这里申请空闲行。
  std::vector<Ptr<std::deque<uint64_t>>> _rows;

  void init(addr_type base_addr);

  // 初始化 NPU 布局：线性切分 memory pool
  void init_npu_layout(addr_type base_addr);

  // 初始化 PIM 布局：按 Channel 划分行资源
  void init_pim_layout(addr_type base_addr);

  // NPU 分配接口：从 _kv_cache 队列取一个地址
  addr_type allocate();

  // PIM 分配接口：指定 Channel，从对应的 _rows[ch] 队列取一个行索引
  addr_type allocate(uint64_t ch);

  void free(addr_type addr);
  void free(uint32_t ch, uint64_t row);
};
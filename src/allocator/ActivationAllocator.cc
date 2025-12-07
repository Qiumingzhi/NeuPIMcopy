#include "AddressAllocator.h"

ActAlloc::ActAlloc()
    : _base_addr(0), _top_addr(0), _act_buf_size(0), _act_buf_limit(0) {}

// 初始化激活值分配器
// 设置基地址、分配缓冲区大小，并计算缓冲区上限。
// 该缓冲区是从 HBM
// 中预分配的一块固定大小的区域，专门用于存放推理过程中的临时激活值。
void ActAlloc::init(addr_type base_addr) {
  _base_addr = base_addr;
  _top_addr = base_addr;
  _act_buf_size = Config::global_config.HBM_act_buf_size;
  _act_buf_limit = _base_addr + _act_buf_size;
}

// 分配内存函数
// 这是一个简单的 Bump Pointer 分配器（指针递增）。
// 每次分配后，_top_addr 向后移动 size 大小，并对齐。
addr_type ActAlloc::allocate(uint64_t size) {
  // 检查是否超出激活缓冲区上限（Overflow check）
  ast(_top_addr + size < _act_buf_limit);
  uint32_t alignment = AddressConfig::alignment;

  addr_type result = _top_addr;
  _top_addr += size;

  // 对齐处理：如果当前地址未对齐，补齐到下一个对齐边界
  if (_top_addr & (alignment - 1)) {
    _top_addr += alignment - (_top_addr & (alignment - 1));
  }
  return result;
}

// 获取激活缓冲区之后的下一个对齐地址
// 通常用于确定下一块内存区域（如 KV Cache 或其他）的起始位置。
addr_type ActAlloc::get_next_aligned_addr() {
  ast(_base_addr > 0);
  ast(_act_buf_size > 0);

  return AddressConfig::align(_act_buf_limit) + AddressConfig::alignment;
}

// 清空分配器 (Flush)
// 将分配指针重置回基地址。
// 这意味着所有之前分配的激活值都被视为“释放”，可以被新一轮计算覆盖。
// 通常在每个推理 Step 结束后调用。
void ActAlloc::flush() { _top_addr = _base_addr; }
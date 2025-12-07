#include "AddressAllocator.h"

WgtAlloc::WgtAlloc()
    : _base_addr(0), _top_addr(0) {
} //这两个参数只会被初始化一次 单例模式（Singleton）类

// 权重分配函数
// 在 DRAM 中分配指定大小的空间。
// 这里的分配单位是 "unit"，它由 DRAM 请求大小乘以通道数决定。
// 这意味着分配是按“全通道宽度的突发请求”为最小单位进行的，以最大化利用带宽。
addr_type WgtAlloc::allocate(uint64_t size) {
  // unit: 最小分配单元（Byte）= 单次DRAM请求大小 * 通道数
  addr_type unit =
      Config::global_config.dram_req_size * Config::global_config.dram_channels;
  // unit: 64 Bytes * 32 = 2KB

  addr_type result = _top_addr;

  // 计算需要多少个 unit，并累加到 _top_addr (类似 sbrk 指针移动)
  // (size + unit - 1) / unit 实现了向上取整
  _top_addr += (size + unit - 1) / unit;

  // if (_top_addr & (AddressConfig::alignment - 1)) {
  //     _top_addr += AddressConfig::alignment - (_top_addr &
  //     (AddressConfig::alignment - 1));
  // }

  return result;
}

// 获取下一个对齐的可用地址
addr_type WgtAlloc::get_next_aligned_addr() {
  ast(_top_addr > 0); //自定义的断言函数
  // 对当前 _top_addr 进行特定的对齐操作，并增加一个对齐步长
  return AddressConfig::align(_top_addr) + AddressConfig::alignment;
}
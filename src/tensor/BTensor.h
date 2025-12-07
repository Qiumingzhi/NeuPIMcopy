#pragma once

#include "../Common.h"

class Operation;

// BTensor（Base Tensor）是整个 NeuPIMs内存模拟
// 和数据流图中的抽象基类（Abstract Base Class）。
// 它定义了所有类型的张量（Tensor）都必须遵守的“契约”和通用行为。

/*
1. 数据流图（Dataflow Graph）节点
NeuPIMs 是一个基于 Trace（踪迹）或
Graph（图）的模拟器。BTensor是这张依赖图中的数据节点。

连接操作（Operations）：
_src_node (Source Node): 谁生产了我？指向一个 Operation 对象（比如 MatMul
或add），表示这个张量是该操作的输出。

_child_nodes (Child Nodes):谁在使用我？一个列表，
指向那些以此张量为输入的Operation对象。这用于构建依赖关系：只有当我被生产出来（
_src_node 完成），我的 _child_nodes才能开始执行。

状态管理：
_produced: 一个简单的布尔值。false 表示还没算好，
true 表示数据已就绪，下游节点可以启动了。这是模拟器调度逻辑的核心检查点。
*/

/*
2. 内存地址映射接口（Memory Mapping Interface）
这是BTensor最独特和核心的部分。由于 NeuPIMs 需要模拟 NPU 和 PIM
两种完全不同的硬件，数据的物理存储方式（Layout）截然不同。
BTensor通过纯虚函数屏蔽了这些差异：

virtual addr_type get_addr(std::vector<uint32_t> indexes) = 0;
灵魂接口。它接受一个高维逻辑索引（比如 [batch=0, head=2, seq=100,
dim=64]），返回一个具体的物理内存地址。

多态性：
NPUTensor 实现：可能会按照 NHWC或 NCHW 等标准连续布局计算地址，
或者加上 Bank 交叉存取偏移。

PIMTensor实现：非常复杂。它会查找这个数据在哪个 Channel，哪个
Bank，哪一行（Row），然后返回一个对应的 PIM 寻址地址。
作用：让上层模拟组件（如计算单元）不需要关心数据到底存在哪、怎么存的，只管发逻辑读写请求。

virtual std::vector<addr_type> get_all_addrs() = 0;
直接拿到这个张量占用的所有物理地址列表。主要用于统计内存占用、预热缓存或生成详细的内存访问
Trace。

virtual void add_token() = 0; 这是为了KV Cache 专门设计的接口。
在自回归生成（LLM Inference）中，序列长度是动态增长的。每次生成一个新 Token，KV
Cache 就变大一点。
具体的子类（如PIMTensor）需要在这个函数里实现动态内存申请逻辑
（比如发现当前行满了，就自动去KVCacheAllocator申请一个新的DRAM行）。


NPUTensor:
运行在主机 NPU 或 Systolic Array 上。它内部通常组合了NPUTensor2D
（普通矩阵）或 NPUTensorKV （KV Cache）来具体管理地址。
PIMTensor:
运行在内存（DRAM）中。它重度依赖 Channel 和 Row
的概念，数据是按行打散分布的。

*/

class BTensor {
public:
  BTensor() = default;
  ~BTensor() = default;

  // 添加依赖于该 Tensor 的子操作节点（即使用该 Tensor 作为输入的操作）
  void add_child_node(Ptr<Operation> op);
  // 清除所有子操作节点
  void clear_child_nodes();

  // 获取 Tensor 的唯一 ID
  uint32_t get_id() { return _id; }
  // 获取 Tensor 的名称
  std::string get_name() { return _name; }
  // 获取 Tensor 的维度信息
  std::vector<uint32_t> get_dims() { return _dims; }
  // 标记该 Tensor 已经被生产（计算完成或数据已就绪）
  void set_produced() { _produced = true; }
  // 检查该 Tensor 是否已就绪
  bool get_produced() { return _produced; }
  // 获取生产该 Tensor 的源操作节点
  Ptr<Operation> get_src_node() { return _src_node; }
  // 获取依赖该 Tensor 的子节点数量
  uint32_t num_child_nodes() { return _child_nodes.size(); }
  // 根据索引获取特定的子操作节点
  Ptr<Operation> get_child_node(uint32_t id) { return _child_nodes[id]; }
  // 获取所有依赖该 Tensor 的子操作节点列表
  std::vector<Ptr<Operation>> get_child_nodes() { return _child_nodes; }

  // 纯虚函数：根据多维索引获取数据在内存中的物理地址
  virtual addr_type get_addr(std::vector<uint32_t> indexes) = 0;
  // 纯虚函数：获取该 Tensor 占用的所有物理地址
  virtual std::vector<addr_type> get_all_addrs() = 0;
  // 纯虚函数：用于 KV Cache 等动态 Tensor，增加一个 Token 的容量
  virtual void add_token() = 0;

  bool _produced;                           // 是否已生产完成
  uint32_t _id;                             // Tensor ID
  std::string _name;                        // Tensor 名称
  std::vector<uint32_t> _dims;              // 维度 [d1, d2, ...]
  Ptr<Operation> _src_node;                 // 生产此 Tensor 的操作节点
  std::vector<Ptr<Operation>> _child_nodes; // 消费此 Tensor 的操作节点列表
  uint32_t _precision;                      // 数据精度（字节数）
};
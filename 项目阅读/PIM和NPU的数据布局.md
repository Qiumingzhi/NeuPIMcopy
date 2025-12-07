# NeuPIMs PIM 与 NPU 数据布局对比分析

NeuPIMs 项目中，PIM（Process-In-Memory）与 NPU（Neural Processing Unit）采用了截然不同的内存数据布局策略，以适应各自的计算特性。核心差异主要体现在 **KV Cache 分配** 和 **地址映射** 上。

## 1. 核心代码位置

*   **KV Cache 分配逻辑**: `src/allocator/KVCacheAllocator.cc`
*   **张量封装**: `src/tensor/PIMTensor.cc` vs `src/tensor/NPUTensor.cc`
*   **物理地址映射**: `src/Common.cc` (Namespace `AddressConfig`)

## 2. 详细布局差异

### A. NPU 数据布局 (NPU Layout)

NPU 的设计目标是利用高带宽突发传输（Burst Access）来加速计算。

*   **分配策略**: **线性连续分配 (Linear Contiguous Allocation)**
*   **核心函数**: `KVCacheAlloc::init_npu_layout`
*   **布局特点**:
    *   **粒度**: 按 `_kv_cache_entry_size` (通常为 32) 个 Sequence Length 为一组进行分配。
    *   **连续性**: 为了提高缓存命中率和总线效率，将 `d_k`（Embedding per Head）维度的元素在物理内存中**连续存储**。
    *   **地址生成**: 维护一个线性的 `_kv_cache` 队列，按请求顺序依次发放起始物理地址。
    *   **目的**: 确保在进行矩阵乘法（如 Attention）加载 KV 时，能够一次性连续加载一个完整的向量块。

### B. PIM 数据布局 (PIM Layout)

PIM 的设计目标是利用多个存储体（Bank）的**并行性**，将计算下沉到存储单元内部。

*   **分配策略**: **基于通道和行的分配 (Channel-Row Based Allocation)**
*   **核心函数**: `KVCacheAlloc::init_pim_layout`
*   **布局特点**:
    *   **Channel 隔离**: 显式管理每个 DRAM Channel 的空闲行（Free Rows）。代码中维护了 `_rows` 向量（`std::vector<Ptr<std::deque<uint64_t>>>`），分别记录每个 Channel 的可用行索引。
    *   **行对齐**: 分配单位通常对齐到 DRAM Row（例如 1024KB/行）。
*   **KV 差异化存储 (`PIMTensor.cc`)**:
    *   **Key 矩阵**: 采用 **跨 Bank 存储**。为了支持并行比较/乘法，数据被打散到同一个 Channel 下的不同 Bank 中。
        *   分配次数: `ceil( seq_len / bank_per_ch )`
    *   **Value 矩阵**: 采用 **跨列存储**。数据在行内的 Column 维度上连续，但在行间可能跨越。
        *   分配次数: `ceil( seq_len / num_ele_per_row )`

## 3. 物理地址编码 (Physical Address Encoding)

PIM 的特殊性还在于它不仅传输数据，还通过地址线传输指令参数。

*   **通用映射**: `src/Common.cc` 中的 `AddressConfig::make_address` 负责将 (Channel, Rank, BankGroup, Bank, Row, Col) 映射为线性物理地址。
*   **PIM 指令编码**:
    *   `AddressConfig::encode_pim_header`: 将 PIM 指令的参数（如 `num_comps` 计算次数、`num_readres` 读取结果数）编码到物理地址的 **Row**、**Bank** 和 **Rank** 位中。
    *   这意味着对于 PIM 操作，物理地址不再仅仅指向数据存储位置，还承载了控制信号。

## 总结

| 特性 | NPU 布局 | PIM 布局 |
| :--- | :--- | :--- |
| **分配器函数** | `init_npu_layout` | `init_pim_layout` |
| **分配方式** | 线性队列 (`_kv_cache`) | Channel-Row 映射表 (`_rows`) |
| **数据连续性** | `d_k` 维度连续 (优化 Burst) | 跨 Bank/Channel 打散 (优化并行度) |
| **地址用途** | 纯数据寻址 | 数据寻址 + 指令参数编码 |
| **典型应用** | 传统矩阵乘法、卷积 | 存内点积、存内归约 |

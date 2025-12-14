# NeuPIMs GPT 推理算子维度分析

本文档分析了 NeuPIMs 项目中 GPT 模型在推理阶段（Inference）各个算子的输入张量、权重张量和输出张量的维度变化。

## 符号定义

| 符号 | 描述 | 对应配置变量 | 典型含义 |
| :--- | :--- | :--- | :--- |
| **B** | Batch Size (当前批次的 Token 数量) | `batch_size` / `num_rows` | 批处理大小 |
| **E** | Embedding Size (嵌入维度) | `model_n_embd` | 768 (GPT-2 Small) |
| **H** | Number of Heads (注意力头数) | `model_n_head` | 12 |
| **n_tp** | Tensor Parallelism (张量并行度) | `n_tp` | 如 1, 2, 4, 8 |
| **dk** | Head Dimension (单头维度) | `E / H` | 64 |
| **T** | Sequence Length (KV Cache 长度) | - | 历史上下文长度 |
| **V** | Vocab Size (词表大小) | `model_vocab_size` | 50257 |

> **注意**: 在 NeuPIMs 的实现中，`MatMul` 算子通常假定权重张量在计算前已转置（`_is_transposed = true`），但为了清晰起见，下表中的权重维度以逻辑形状 `[In_Features, Out_Features]` 表示。

---

## 1. Attention Block (注意力层)

Attention 层主要包含 QKV 生成、存内计算（PIM MHA）和输出投影（Projection）。

| 算子 (Operator) | 输入张量维度 (Input) | 权重张量维度 (Weight) | 输出张量维度 (Output) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **LayerNorm 1** | `[B, E]` | `[E]` (Scale/Bias) | `[B, E]` | 归一化输入 |
| **QKVGen (MatMul)** | `[B, E]` | `[E, 3 * E / n_tp]` | `[B, 3 * E / n_tp]` | 生成 Q, K, V。由于张量并行，每个 Rank 只生成一部分头。 |
| **Split (Implicit)** | `[B, 3 * E / n_tp]` | - | 3 x `[B, E / n_tp]` | 逻辑上将 QKV 分离，准备送入 PIM。此时 `E / n_tp` 等价于 `(H / n_tp) * dk`。 |
| **PIM MHA (Logit)** | Q: `[H/n_tp, 1, dk]`<br>K: `[H/n_tp, dk, T]` | - | `[H/n_tp, 1, T]` | 计算 Attention Scores (QK^T)。在 PIM 中完成。`B` 维被展开处理。 |
| **PIM MHA (Attend)** | Score: `[H/n_tp, 1, T]`<br>V: `[H/n_tp, T, dk]` | - | `[H/n_tp, 1, dk]` | 计算 Attention Output (Score * V)。在 PIM 中完成。 |
| **Projection (MatMul)** | `[B, E / n_tp]` | `[E / n_tp, E]` | `[B, E]` | 将多头的输出投影回原始嵌入维度。输入来自 PIM 结果的拼接。 |
| **Residual (Add)** | `[B, E]`, `[B, E]` | - | `[B, E]` | Attention 输出 + 输入残差 |

**特定说明：**
*   **QKVGen**: 输出通道数除以 `n_tp`，因为 QKV 权重被切分到了不同的计算节点上。
*   **PIM MHA**: 这是 NeuPIMs 的核心。计算发生在内存侧，处理的是按头（Head）组织的张量。NeuPIMs 将 `[B, E/n_tp]` 视为多个 `[1, dk]` 的请求集合（针对每个 Head）。
*   **Projection**: 输入维度是 `E / n_tp`，因为它接收的是当前 TP Rank 负责的那部分 Head 的计算结果。输出维度恢复为 `E`，这通常意味着需要进行 All-Reduce 操作（但在仿真器简单代码中可能通过 Result 聚合体现）。

---

## 2. FeedForward Block (前馈网络层)

FFN 层紧接在 Attention 层之后。

| 算子 (Operator) | 输入张量维度 (Input) | 权重张量维度 (Weight) | 输出张量维度 (Output) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **LayerNorm 2** | `[B, E]` | `[E]` | `[B, E]` | 归一化 Attention 层的输出 |
| **FC1 (MatMul)** | `[B, E]` | `[E, 4 * E / n_tp]` | `[B, 4 * E / n_tp]` | 升维投影 (通常 4 倍)。由于 TP，输出维度被切分。 |
| **Gelu** | `[B, 4 * E / n_tp]` | - | `[B, 4 * E / n_tp]` | 激活函数 |
| **FC2 (MatMul)** | `[B, 4 * E / n_tp]` | `[4 * E / n_tp, E]` | `[B, E]` | 降维投影回 `E`。 |
| **Residual (Add)** | `[B, E]`, `[B, E]` | - | `[B, E]` | FFN 输出 + Attention 输出残差 |

**特定说明：**
*   **FC1**: 类似于 QKVGen，权重按列切分（Column Parallel），每个 Rank 计算一部分特征，输出宽度减小为 `4E / n_tp`。
*   **FC2**: 权重按行切分（Row Parallel），接收切分后的输入 `4E / n_tp`，并将其投影回完整的 `E`。这通常需要随后进行 All-Reduce Sum。

---

## 3. Language Model Head (LM Head)

最后一层，用于生成下一个 Token 的概率分布。

| 算子 (Operator) | 输入张量维度 (Input) | 权重张量维度 (Weight) | 输出张量维度 (Output) | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **LayerNorm (Final)**| `[B, E]` | `[E]` | `[B, E]` | 最终归一化 (部分模型架构有) |
| **LmHead (MatMul)** | `[B, E]` | `[E, V]` | `[B, V]` | 投影到词表大小。通常在 Top-Level 模型定义中。 |

---

## 总结

NeuPIMs 仿真器通过 `StageProgram.cc` 编排上述流程。关键的维度变换逻辑在于：
1.  **SysArray (SA) 阶段**: 处理 `[B, E]` 到 `[B, E]` 或者切分维度的矩阵乘法。
2.  **PIM 阶段**: 处理 `[Head, T, dk]` 格式的 Attention 计算，利用内存高带宽优势处理 KV Cache。
3.  **Tensor Parallelism**: 贯穿始终，主要体现为 `QKVGen`、`Projection`、`FC1`、`FC2` 的权重和中间激活张量的维度除以 `n_tp`。

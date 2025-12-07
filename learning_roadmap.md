# NeuPIMs 项目深度掌握学习路线图

要想达到“随心所欲修改”的程度，建议按照以下 **5 个阶段** 深入源码。不要试图一次看懂所有代码，而是顺着**数据流**和**控制流**抽丝剥茧。

## 第一阶段：宏观架构与数据流 (The Big Picture)
**目标**：理解“输入是什么，输出是什么，中间经历了哪几个大步骤”。

1.  **入口点 (`src/main.cc`)**
    *   看它如何解析参数，如何初始化 `SimulationConfig`。
    *   看它如何创建 `Simulator` 和 `Model`。
    *   **关键问题**：配置文件的参数是如何传递给各个组件的？

2.  **模型构建 (`src/Model.cc` & `src/ModelProgram.cc`)**
    *   `Model.cc`：负责加载权重参数（虽然是模拟的），定义模型的静态结构。
    *   `ModelProgram.cc`：**核心中的核心**。它负责“画图”。
    *   **任务**：找到 `init_program` 函数，看它是如何把 `NeuPIMSAttend`、`SplitDecoding` 这些积木搭在一起的。

3.  **执行循环 (`src/Simulator.cc`)**
    *   `Simulator::run` 和 `Simulator::cycle` 是整个系统的心跳。
    *   **任务**：搞清楚 `Simulator` 是如何每一拍（cycle）都去推一下 `Scheduler`、`Core` 和 `Dram` 的。

---

## 第二阶段：计算图与指令生成 (The Brain)
**目标**：学会如何添加一个新的算子（Operation）。

1.  **算子基类 (`src/operations/Operation.h`)**
    *   理解 `get_outputs` 接口：它定义了数据依赖关系。
    *   理解 `_tiles` 和 `initialize_instructions`：这是从“图节点”到“硬件指令”的桥梁。

2.  **典型算子分析 (`src/operations/NeuPIMSAttend.cc`)**
    *   **精读**这个文件。它是 PIM 模拟的灵魂。
    *   看它如何计算 `calculate_loops`（分块策略）。
    *   看它如何在 `initialize_instructions` 中生成 `PIM_GWRITE`, `PIM_COMP` 等指令。
    *   **练习**：尝试复制一个 `NeuPIMSAttend`，改名为 `MyAttend`，并让 `ModelProgram` 调用它。

---

## 第三阶段：调度与控制 (The Nervous System)
**目标**：理解任务是如何被分发和执行的。

1.  **调度器 (`src/scheduler/Scheduler.cc`)**
    *   它是“大脑”。它决定了哪些 Tile 可以被发射（Issue）。
    *   关注 `allocate_requests`（请求分配）和 `cycle`（流水线推进）。
    *   理解 `Stage` (A, B, C...) 的概念，这是流水线并行（Pipeline Parallelism）的关键。

2.  **核心执行 (`src/Core.cc` & `src/SystolicWS.cc`)**
    *   它是“手脚”。
    *   `Core::issue`：接收任务。
    *   `SystolicWS::cycle`：模拟脉动阵列的计算延迟。
    *   **关键点**：看它如何处理 `_ld_inst_queue`（加载队列）和 `_ex_inst_queue`（执行队列）。

---

## 第四阶段：内存与 PIM 模拟 (The Soul)
**目标**：理解 PIM 指令是如何在内存侧生效的。

1.  **内存接口 (`src/Dram.cc`)**
    *   它是模拟器和 DRAM 库之间的边界。
    *   看它如何把 `MemoryAccess` 转换成底层事务。

2.  **PIM 核心 (`extern/NewtonSim/src/NewtonSim.cc` & `neupims_controller.cc`)**
    *   这是“魔改”发生的地方。
    *   `NewtonSim.cc`：封装层。
    *   `neupims_controller.cc`：**最硬核的部分**。它实现了 PIM 指令的时序控制。如果你要修改 PIM 的硬件行为（比如增加新的 PIM 指令），必须动这里。

---

## 第五阶段：实战演练 (Hands-on Exercises)
光看不练假把式。建议按顺序完成以下修改任务：

1.  **Level 1 (配置修改)**：
    *   修改 `gpt3-7B.json`，把 `n_tp` 改回 4，并尝试修改 `SplitDecoding.cc` 里的代码，让它能正确处理分片后的维度（修复我们之前遇到的 crash）。

2.  **Level 2 (增加统计)**：
    *   在 `NeuPIMSAttend.cc` 里，统计一下一共生成了多少条 `PIM_COMP` 指令，并在程序结束时打印出来。

3.  **Level 3 (修改调度)**：
    *   在 `Scheduler.cc` 里，尝试修改 `allocate_requests` 的逻辑，比如强制让所有请求都分配到 Channel 0（虽然性能会很差，但能验证你对调度的控制力）。

4.  **Level 4 (新增算子)**：
    *   实现一个新的 `MyRelu` 算子（参考 `Gelu`），并在 `ModelProgram` 中把它插入到 `NeuPIMSAttend` 之后。

5.  **Level 5 (修改硬件行为)**：
    *   在 `extern/NewtonSim` 里，尝试增加一种新的 PIM 指令类型 `PIM_MY_OP`，并让它在 DRAM 控制器里产生一定的延迟。

---

## 必备工具
*   **GDB / VSCode Debugger**：你必须学会单步调试。看着变量在 `cycle()` 里怎么变，比看代码一万遍都管用。
*   **spdlog**：项目里到处都是 `spdlog::info`。学会利用它打印关键路径的日志。
*   **Graphviz**：继续利用你已经修好的 `debug_graph` 工具，可视化你的修改对计算图的影响。

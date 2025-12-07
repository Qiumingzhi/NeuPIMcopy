#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

typedef uint64_t cycle_type;

enum class CoreType {
  SYSTOLIC_OS,
  SYSTOLIC_WS
}; // 核心类型：脉动阵列 输出驻留 (OS) 或 权重驻留 (WS) 目前只实现了一种

enum class DramType {
  DRAM,
  NEWTON,
  NEUPIMS
}; // DRAM类型：普通DRAM, Newton架构, 或 NeuPIMs架构

enum class IcntType {
  SIMPLE,
  BOOKSIM2
}; // 互连网络类型：简单模型 或 BookSim2仿真

enum class RunMode { NPU_ONLY, NPU_PIM }; // 运行模式：仅NPU 或 NPU+PIM异构

struct SimulationConfig {
  // gpt model config (GPT模型配置)
  std::string model_name;    // 模型名称
  uint32_t model_params_b;   // 模型参数量 (Bytes或数量)
  uint32_t model_block_size; // 模型块大小 (通常指上下文窗口大小)
  uint32_t model_vocab_size; // 词表大小
  uint32_t model_n_layer;    // 层数
  uint32_t model_n_head;     // 注意力头数
  uint32_t model_n_embd;     // 嵌入维度

  /* Custom Config (自定义配置) */
  RunMode run_mode;         // NPU (运行模式)
  bool sub_batch_mode;      // 是否开启子批处理模式
  bool ch_load_balancing;   // 是否开启通道负载均衡
  bool kernel_fusion;       // 是否开启算子融合
  uint32_t max_batch_size;  // 最大批大小
  uint32_t max_active_reqs; // max size of (ready_queue + running_queue) in
                            // scheduler (调度器中 就绪+运行 队列的最大请求数)
  uint32_t max_seq_len;      // 最大序列长度
  uint64_t HBM_size;         // HBM size in bytes (HBM总容量，字节)
  uint64_t HBM_act_buf_size; // HBM activation buffer size in bytes
                             // (HBM激活值缓冲区大小，字节)

  /* Core config (核心配置 - NPU脉动阵列) */
  uint32_t num_cores;   // 核心数量
  CoreType core_type;   // 核心类型 (OS/WS)
  uint32_t core_freq;   // 核心频率
  uint32_t core_width;  // 脉动阵列宽度
  uint32_t core_height; // 脉动阵列高度

  uint32_t n_tp; // Tensor Parallelism degree (张量并行度)

  uint32_t vector_core_count; // 向量核心数量
  uint32_t vector_core_width; // 向量核心宽度 (SIMD宽度)

  /* Vector config (向量单元配置及延迟) */
  uint32_t process_bit; // 处理位宽

  cycle_type layernorm_latency;   // LayerNorm 延迟
  cycle_type softmax_latency;     // Softmax 延迟
  cycle_type add_latency;         // 加法延迟
  cycle_type mul_latency;         // 乘法延迟
  cycle_type exp_latency;         // 指数运算延迟
  cycle_type gelu_latency;        // GELU激活 延迟
  cycle_type add_tree_latency;    // 加法树延迟
  cycle_type scalar_sqrt_latency; // 标量开方延迟
  cycle_type scalar_add_latency;  // 标量加法延迟
  cycle_type scalar_mul_latency;  // 标量乘法延迟

  /* SRAM config (片上SRAM配置) */
  uint32_t sram_width;      // SRAM宽度
  uint32_t sram_size;       // SRAM大小
  uint32_t spad_size;       // Scratchpad Memory (SPAD) 大小
  uint32_t accum_spad_size; // 累加器 SPAD 大小

  /* DRAM config (DRAM内存配置) */
  DramType dram_type;     // DRAM 类型
  uint32_t dram_freq;     // DRAM 频率
  uint32_t dram_channels; // DRAM 通道数
  uint32_t dram_req_size; // DRAM 请求大小 (粒度)

  /* PIM config (PIM存内计算配置) */
  std::string pim_config_path; // PIM 配置文件路径
  uint32_t dram_page_size;     // DRAM row buffer size (in bytes)
                               // (DRAM行缓冲区大小，字节)
  uint32_t dram_banks_per_ch;  // 每个通道的 Bank 数量
  uint32_t pim_comp_coverage;  // # params per PIM_COMP command (每条 PIM
                               // 计算指令覆盖的参数量)

  /* Log config (日志配置) */
  std::string operation_log_output_path; // 操作日志输出路径
  std::string log_dir;                   // 日志目录

  /* Client config (客户端/负载配置) */
  uint32_t request_input_seq_len;   // 请求输入序列长度
  uint32_t request_interval;        // 请求间隔
  uint32_t request_total_cnt;       // 总请求数量
  std::string request_dataset_path; // 请求数据集路径

  /* ICNT config (互连网络配置) */
  IcntType icnt_type;           // 互连类型
  std::string icnt_config_path; // 互连配置文件路径
  uint32_t icnt_freq;           // 互连频率
  uint32_t icnt_latency;        // 互连延迟

  /* Sheduler config (调度器配置) */
  std::string scheduler_type; // 调度器类型

  /* Other configs (其他配置) */
  uint32_t precision; // 精度
  std::string layout; // 数据布局

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  } // 地址对齐 (按 DRAM 请求大小对齐)
};

namespace Config {
extern SimulationConfig global_config;
}
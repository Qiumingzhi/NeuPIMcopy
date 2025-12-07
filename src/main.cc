#include "Simulator.h"
#include "Common.h"
#include "allocator/AddressAllocator.h"
#include "helper/CommandLineParser.h"
#include "operations/Operation.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    // parse command line argumnet
    CommandLineParser cmd_parser = CommandLineParser();
    cmd_parser.add_command_line_option<std::string>("config",
                                                    "Path for hardware configuration file");
    cmd_parser.add_command_line_option<std::string>("mem_config",
                                                    "Path for memory configuration file");
    cmd_parser.add_command_line_option<std::string>("cli_config",
                                                    "Path for client configuration file");
    cmd_parser.add_command_line_option<std::string>("model_config",
                                                    "Path for model configuration file");
    cmd_parser.add_command_line_option<std::string>("sys_config",
                                                    "Path for system configuration file");
    cmd_parser.add_command_line_option<std::string>("log_dir",
                                                    "Path for experiment result log directory");

    cmd_parser.add_command_line_option<std::string>("models_list", "Path for the models list file");
    cmd_parser.add_command_line_option<std::string>(
        "log_level", "Set for log level [trace, debug, info], default = info");
    cmd_parser.add_command_line_option<std::string>("mode", "choose one_model or two_model");

    try {
        cmd_parser.parse(argc, argv);
    } catch (const CommandLineParser::ParsingError &e) {
        spdlog::error("Command line argument parrsing error captured. Error message: {}", e.what());
        throw(e);
    }
    std::string model_base_path = "./models";
    std::string level = "info";
    cmd_parser.set_if_defined("log_level", &level);
    if (level == "trace")
        spdlog::set_level(spdlog::level::trace);
    else if (level == "debug")
        spdlog::set_level(spdlog::level::debug);
    else if (level == "info")
        spdlog::set_level(spdlog::level::info);

    std::string config_path;
    cmd_parser.set_if_defined("config", &config_path);

    json config_json;
    std::ifstream config_file(config_path);
    config_file >> config_json;
    config_file.close();
    Config::global_config = initialize_config(config_json);

    std::string mem_config_path;
    cmd_parser.set_if_defined("mem_config", &mem_config_path);
    std::string cli_config_path;
    cmd_parser.set_if_defined("cli_config", &cli_config_path);
    std::string model_config_path;
    cmd_parser.set_if_defined("model_config", &model_config_path);
    std::string sys_config_path;
    cmd_parser.set_if_defined("sys_config", &sys_config_path);
    std::string log_dir_path;
    cmd_parser.set_if_defined("log_dir", &log_dir_path);

    initialize_memory_config(mem_config_path);
    initialize_client_config(cli_config_path);
    initialize_model_config(model_config_path);
    initialize_system_config(sys_config_path); //这几个函数都定义在Common.h中

    Config::global_config.log_dir = log_dir_path; // Config 在项目中是一个命名空间 在src/SimulationConfig.h
 

    Operation::initialize(Config::global_config);

    auto simulator = std::make_unique<Simulator>(Config::global_config);
    AddressConfig::alignment = Config::global_config.dram_req_size;
    //alignment (对齐大小)：设置为 DRAM 的请求粒度（通常是 64 字节）。
    //作用：模拟器在分配地址时，会保证每个数据块的起始地址都是 64 的倍数，模拟真实的内存对齐要求。

    // todo: assert log2
    AddressConfig::channel_mask = Config::global_config.dram_channels - 1;
    //channel_mask (通道掩码)：设置为 DRAM 的通道数减 1。
    //作用：用于快速计算地址的通道部分，通过位掩码操作实现。
    //原理：如果通道数是 2 的幂（比如 16），那么 16 - 1 = 15 (二进制 00001111)。
    //用法：channel_id = address & channel_mask。这是一种比取模运算 (%) 快得多的位运算技巧。

    // todo: magic number
    AddressConfig::channel_offset = 10;  // 64B req_size -> 6bit + 16 groups of columns -> 4bit
    //channel_offset (通道偏移量)：决定了地址中的哪几位用来表示通道 ID。
    //解释：
    // 10 表示从第 10 位开始看通道 ID。
    // 低 6 位（0-5）：64B req_size -> 用于表示 64 字节内的偏移。
    // 中间 4 位（6-9）：16 groups of columns -> 用于列地址的一部分。
    // 第 10 位开始：才是通道选择位。
    // 目的：这定义了地址交错 (Interleaving) 的粒度。这里采用了“细粒度交错”，使得相邻的数据块分布在不同的通道上，以最大化并行度。
    
    
    
    spdlog::info("DRAM address alignment {}", AddressConfig::alignment);

    std::string model_name = Config::global_config.model_name;
    std::string input_name = "input";
    spdlog::info("model name: {}", model_name);
    auto model = std::make_shared<Model>(Config::global_config, model_name);

    /* Allocator initialization after weight allocating */
    // if (Config::global_config.run_mode == RunMode::NPU_PIM) {
    //     ActAlloc::init(WgtAlloc::get_next_aligned_addr());
    //     KVCacheAlloc::init(ActAlloc::get_next_aligned_addr());
    // }
    ActAlloc::GetInstance()->init(WgtAlloc::GetInstance()->get_next_aligned_addr());
    KVCacheAlloc::GetInstance()->init(ActAlloc::GetInstance()->get_next_aligned_addr());

    printf("Launching model\n");
    simulator->launch_model(model);
    spdlog::info("Launch model: {}", model_name);
    simulator->run(model_name);

    MemoryAccess::log_count();

    std::string yellow = "\033[1;33m";
    std::string red = "\033[1;31m";
    std::string color = Config::global_config.kernel_fusion ? yellow : red;
    std::string prefix = Config::global_config.kernel_fusion ? "fused" : "naive";
    spdlog::info("{}mode: {} {}{}", color, prefix,
                 Config::global_config.run_mode == RunMode::NPU_ONLY ? "NPU-only" : "NPU+PIM",
                 "\033[0m");
    return 0;
}

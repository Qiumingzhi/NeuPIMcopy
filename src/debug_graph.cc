#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

#include "Common.h"
#include "Model.h"
#include "ModelProgram.h"
#include "BatchedRequest.h"
#include "helper/CommandLineParser.h"
#include "operations/Operation.h"
#include "allocator/AddressAllocator.h"
#include "tensor/PIMTensor.h"

// Mock Client to generate a request
std::shared_ptr<InferRequest> create_dummy_request(uint32_t id, uint32_t seq_len, uint32_t output_len) {
    auto req = std::make_shared<InferRequest>();
    req->id = id;
    req->input_size = seq_len;
    req->output_size = output_len;
    req->is_initiated = false;
    req->generated = 0;
    req->channel = 0; // Default channel
    return req;
}

int main(int argc, char **argv) {
    // 1. Parse Command Line Arguments
    CommandLineParser cmd_parser = CommandLineParser();
    cmd_parser.add_command_line_option<std::string>("config", "Path for hardware configuration file");
    cmd_parser.add_command_line_option<std::string>("model_config", "Path for model configuration file");
    cmd_parser.add_command_line_option<std::string>("mem_config", "Path for memory configuration file");
    cmd_parser.add_command_line_option<std::string>("sys_config", "Path for system configuration file");
    
    try {
        cmd_parser.parse(argc, argv);
    } catch (const CommandLineParser::ParsingError &e) {
        spdlog::error("Command line argument parsing error: {}", e.what());
        return -1;
    }

    std::string config_path;
    cmd_parser.set_if_defined("config", &config_path);
    std::string model_config_path;
    cmd_parser.set_if_defined("model_config", &model_config_path);
    std::string mem_config_path;
    cmd_parser.set_if_defined("mem_config", &mem_config_path);
    std::string sys_config_path;
    cmd_parser.set_if_defined("sys_config", &sys_config_path);

    if (config_path.empty() || model_config_path.empty()) {
        spdlog::error("Please provide at least --config and --model_config");
        return -1;
    }

    // 2. Initialize Configuration
    spdlog::info("Loading configuration from: {}", config_path);
    json config_json;
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        spdlog::error("Failed to open config file: {}", config_path);
        return -1;
    }
    config_file >> config_json;
    config_file.close();
    
    Config::global_config = initialize_config(config_json);

    // Initialize other configs if provided
    if (!mem_config_path.empty()) initialize_memory_config(mem_config_path);
    if (!model_config_path.empty()) initialize_model_config(model_config_path);
    if (!sys_config_path.empty()) initialize_system_config(sys_config_path);
    
    // Initialize Operation static members
    Operation::initialize(Config::global_config);
    
    // Initialize Address Alignment
    AddressConfig::alignment = Config::global_config.dram_req_size;
    spdlog::info("DRAM address alignment {}", AddressConfig::alignment);

    // 3. Create Model
    std::string model_name = Config::global_config.model_name;
    if (model_name.empty()) {
        spdlog::error("Model name is empty! Check your model config file.");
        return -1;
    }
    spdlog::info("Creating Model: {}", model_name);
    auto model = std::make_shared<Model>(Config::global_config, model_name);
    
    // 4. Initialize Allocators (Crucial step to avoid FPE/Assertion failures)
    spdlog::info("Initializing Allocators...");
    ActAlloc::GetInstance()->init(WgtAlloc::GetInstance()->get_next_aligned_addr());
    KVCacheAlloc::GetInstance()->init(ActAlloc::GetInstance()->get_next_aligned_addr());

    // 5. Create Dummy Request and Populate KV Cache
    // Create a batch of requests to trigger ModelProgram initialization
    std::vector<std::shared_ptr<InferRequest>> reqs;
    uint32_t seq_len = 128;
    reqs.push_back(create_dummy_request(0, seq_len, 129)); 
    
    // Populate KV Cache for each request
    // Note: ModelProgram uses full model_n_head for Query in standalone MHA mode, 
    // so we match it here to avoid dimension mismatch assertions.
    uint32_t nh = Config::global_config.model_n_head; // / Config::global_config.n_tp;
    uint32_t dk = Config::global_config.model_n_embd / Config::global_config.model_n_head;
    
    for (auto& req : reqs) {
        for (int i = 0; i < Config::global_config.model_n_layer; ++i) {
             std::vector<uint32_t> dim_key{nh, dk, seq_len};
             std::vector<uint32_t> dim_value{nh, seq_len, dk};
             
             auto k = std::make_shared<PIMTensor>(
                name_gen(std::to_string(req->id), "KEY", std::to_string(i)), 0, dim_key, PIMTensorKVType::KEY, true);
             auto v = std::make_shared<PIMTensor>(
                name_gen(std::to_string(req->id), "VALUE", std::to_string(i)), 0, dim_value, PIMTensorKVType::VALUE, true);
             
             req->K_cache.push_back(k);
             req->V_cache.push_back(v);
        }
        req->is_initiated = true; // Mark as initiated to simulate decoding phase if needed
    }

    auto batched_req = std::make_shared<BatchedRequest>(reqs);

    // 6. Create ModelProgram (This builds the graph)
    spdlog::info("Initializing ModelProgram (Building Graph)...");
    auto model_program = std::make_unique<ModelProgram>(model, batched_req);

    // 7. Inspect the Graph
    spdlog::info("Graph Construction Complete.");
    spdlog::info("==================================================");
    spdlog::info("Inspecting Operations in Model:");

    auto exec_ops = model_program->get_executable_operations();
    spdlog::info("Number of executable operations found: {}", exec_ops.size());

    // Debug: Trace from inputs
    spdlog::info("Tracing from inputs...");
    // We know ModelProgram creates "query" tensors. 
    // But we don't have direct access to them easily from here unless we inspect the model's internal maps if exposed,
    // or we can look at the KV cache tensors which we do have access to.
    
    if (!reqs.empty()) {
        auto req = reqs[0];
        if (!req->K_cache.empty()) {
            auto k_tensor = req->K_cache[0];
            spdlog::info("Checking K_cache[0] (ID: {}) children:", k_tensor->get_id());
            auto children = k_tensor->get_child_nodes();
            if (children.empty()) {
                spdlog::info("  No children found for K_cache[0]. Graph might be disconnected.");
            } else {
                for (auto child : children) {
                    spdlog::info("  -> Child Op: {} (ID: {}, Type: {})", child->get_name(), child->get_id(), child->get_optype());
                    spdlog::info("     Executable: {}", child->check_executable());
                }
            }
        }
    }

    for (const auto& op : exec_ops) {
        spdlog::info("Op ID: {}, Name: {}, Type: {}", op->get_id(), op->get_name(), op->get_optype());
        
        auto children = op->get_child_nodes();
        if (!children.empty()) {
            spdlog::info("  -> Feeds into {} children:", children.size());
            for (const auto& child : children) {
                spdlog::info("     - {} (ID: {})", child->get_name(), child->get_id());
            }
        }
    }

    spdlog::info("==================================================");

    // 8. Export to Graphviz
    std::string dot_filename = "graph.dot";
    spdlog::info("Exporting graph to {}...", dot_filename);
    
    std::ofstream dot_file(dot_filename);
    if (dot_file.is_open()) {
        dot_file << "digraph ComputationGraph {\n";
        dot_file << "    rankdir=LR;\n";
        dot_file << "    node [shape=box, style=filled, color=lightblue];\n";
        
        // Export all operations, not just executable ones
        for (const auto& [id, op] : model_program->_op_map) {
            // Node definition
            dot_file << "    op_" << op->get_id() << " [label=\"" << op->get_name() << "\\n" << op->get_optype() << "\"];\n";
            
            // Edges to children
            auto children = op->get_child_nodes();
            for (const auto& child : children) {
                dot_file << "    op_" << op->get_id() << " -> op_" << child->get_id() << ";\n";
            }
        }
        
        // Also add the input tensors/requests if possible to see the start
        if (!reqs.empty() && !reqs[0]->K_cache.empty()) {
             auto k_tensor = reqs[0]->K_cache[0];
             dot_file << "    tensor_" << k_tensor->get_id() << " [label=\"" << k_tensor->get_name() << "\\n(Input Tensor)\", shape=ellipse, color=lightgrey];\n";
             
             for (auto child : k_tensor->get_child_nodes()) {
                 dot_file << "    tensor_" << k_tensor->get_id() << " -> op_" << child->get_id() << ";\n";
             }
        }

        dot_file << "}\n";
        dot_file.close();
        spdlog::info("Graph exported successfully to {}", dot_filename);
    } else {
        spdlog::error("Failed to open {} for writing", dot_filename);
    }

    return 0;
}

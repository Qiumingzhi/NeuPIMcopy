#pragma once

#include <vector>

#include "BatchedRequest.h"
#include "Common.h"
#include "Logger.h"
#include "Model.h"
#include "Stat.h"
#include "operations/Operation.h"
#include "tensor/BTensor.h"

class StageProgram {
public:
  StageProgram(std::shared_ptr<Model> model,
               Ptr<BatchedRequest> batched_request, StagePlatform stage_type,
               Stage stage);
  void init_program();
  Ptr<Operation> add_op(Ptr<Operation> op);
  std::vector<Ptr<BTensor>> get_outputs(Ptr<Operation> op,
                                        std::vector<Ptr<BTensor>> inputs);

  bool check_exist_in_executable(uint32_t op_id);
  void finish_operation(uint32_t id);
  void find_executable_node(Ptr<BTensor> tensor);
  std::vector<std::shared_ptr<Operation>> get_executable_operations() {
    return _executable_operations;
  }
  bool check_finish();
  std::vector<OperationStat> list_operation_stat();
  void finish_operation_tile(Tile &tile);
  void log();

  std::string _name;

  // todo: from BatchedRequest
  std::shared_ptr<Model> _model;
  std::shared_ptr<BatchedRequest> _breq;
  //含义: 这是一个指向 BatchedRequest对象的智能指针。作用: 它存储了当前 Stage
  //需要处理的一批推理请求（即 Sub-Batch）。
  robin_hood::unordered_map<uint32_t, Ptr<Operation>>
      _op_map; // 含义: 这是一个哈希表（使用高效的 robin_hood 库实现），键是操作
               // ID (uint32_t)，值是指向 Operation 对象的指针。作用:
               // 它是所有已创建算子的注册表。

  // todo: Constructor
  // std::map<uint32_t, Ptr<Operation>> _operation_map;
  std::map<uint32_t, Ptr<BTensor>> _tensor_map;
  std::vector<std::shared_ptr<Operation>> _executable_operations;

  // Sub-batch interleaving
  StagePlatform _stage_platform;
  Stage _stage;

  void init_SA_program();
  void init_PIM_program();

  bool enable_proj_ffns();
  bool enable_qkv_gen();
  bool skip_pim_stage();

  // Layer Block
  std::vector<Ptr<BTensor>> projection_block(std::vector<Ptr<BTensor>> inputs);
  std::vector<Ptr<BTensor>> ffn_block(std::vector<Ptr<BTensor>> inputs);
  std::vector<Ptr<BTensor>> qkv_gen_block(std::vector<Ptr<BTensor>> inputs);
};
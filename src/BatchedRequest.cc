#include "BatchedRequest.h"

#include "./tensor/BTensor.h"

BatchedRequest::BatchedRequest(
    std::vector<std::shared_ptr<InferRequest>> reqs) {
  _reqs.assign(reqs.begin(), reqs.end());
  _batch_size = _reqs.size();
}

uint32_t BatchedRequest::get_num_reqs() { return _reqs.size(); }

uint32_t BatchedRequest::get_num_rows() {
  //这段代码计算了当前 Batch 中所有请求的总行数（即总 Token
  //数），这对于确定矩阵运算的维度至关重要。
  /*核心逻辑：

  解码阶段 (Decoding Phase): req->is_initiated
  为真。此时模型是自回归生成的，每次只处理最新的那个 Token，所以每个请求贡献 1
  行。
  预填充阶段 (Prefill Phase): req->is_initiated
  为假。此时模型需要一次性处理用户输入的所有 Prompt Token，所以每个请求贡献
  input_size 行。*/

  uint32_t num_rows = 0;
  for (auto req : _reqs) {
    num_rows += req->is_initiated ? 1 : req->input_size;
  }
  return num_rows;
}

std::vector<uint32_t> BatchedRequest::get_num_rows_breakdown() {
  std::vector<uint32_t> num_rows_breakdown;
  for (auto req : _reqs) {
    num_rows_breakdown.push_back(req->is_initiated ? 1 : req->input_size);
  }
  return num_rows_breakdown;
}

bool BatchedRequest::is_initiated(uint32_t index) {
  ast(index < _reqs.size());
  return _reqs[index]->is_initiated;
}

std::pair<Ptr<BTensor>, Ptr<BTensor>>
BatchedRequest::get_cache(uint32_t layer, uint32_t index) {
  return std::make_pair(_reqs[index]->K_cache[layer],
                        _reqs[index]->V_cache[layer]);
}

// void BatchedRequest::store_cache(uint32_t layer, uint32_t index, Ptr<BTensor>
// key,
//                                  Ptr<BTensor> value) {
//     // _breq[index]->KCache[layer].push_back(key);
//     // _breq[index]->VCache[layer].push_back(value);
// }

// std::vector<std::shared_ptr<BatchedTensor>> get_cache(uint32_t layer,
// std::string type) {}
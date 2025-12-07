#pragma once

#include "BTensor.h"
#include "NPUTensor2D.h"
#include "NPUTensorKV.h"

/*

    NPUTensor
    本身不直接管理一个巨大的内存块，而是把自己看作是多个更小的、基础的张量块
    (NPUTensorInner) 的集合。

    为什么这样做？因为 Transformer 模型中的张量通常是 3 维的（[Batch/Head,
   Length, Dim]）。NPU 可能会把每一个 Head 的数据分开存储，或者把 Batch 分开。

    如果是 2D 张量（权重矩阵），_inners 可能只有 1 个 NPUTensor2D。
    如果是 3D 张量（激活值），_inners 可能有 Head数量那么多个NPUTensor2D，
        每个存一个 Head 的数据。
    如果是 KV Cache，_inners也是多个NPUTensorKV。
*/

class NPUTensor : public BTensor {
public:
  NPUTensor() = default;

  //三个重载版本，对应不同的使用场景：
  NPUTensor(std::string name, std::vector<uint32_t> dims,
            NPUTensorBufType buf_type, bool produced);
  // 普通 2D/3D 张量
  // (NPUTensorBufType)：用于权重（Weight）或普通激活（Activation）。如果输入是
  // 3D [h, l, d]，它会创建 h 个 NPUTensor2D，每个大小是 [l, d]。

  NPUTensor(std::string name, std::vector<uint32_t> dims,
            NPUTensorKVType kv_type, bool produced);
  // KV Cache 张量 (NPUTensorKVType)：用于 Key 或 Value 缓存。
  // 同样，如果是 [h, l, d]，创建 h 个 NPUTensorKV。

  NPUTensor(std::string name, Ptr<NPUTensor2D> tensor, bool produced);
  // 包装现有 Inner Tensor：直接把一个现成的 NPUTensor2D 包装成 NPUTensor

  ~NPUTensor() = default;

  std::vector<uint32_t> get_dims();
  virtual addr_type get_addr(std::vector<uint32_t> indexes);
  //地址计算 (get_addr)：由于数据分散在_inners里，
  // get_addr需要做一个路由（Routing）工作
  virtual std::vector<addr_type> get_all_addrs();

  virtual void set_transposed();
  virtual void unset_transposed();
  virtual void add_token() override; // for KV
  std::vector<addr_type> get_row_addrs(uint32_t row_idx);

  std::vector<Ptr<NPUTensor>>
  split_by_row(std::vector<uint32_t> row_dims); // for 2D

  std::vector<Ptr<NPUTensorInner>> _inners;

  bool _is_transposed;
};
debug_graph.cc用于验证 ModelProgram 是否构建了计算图。

计算图看起来比较小（只有几个节点）主要有两个原因：

配置文件设置： 
您使用的模型配置文件 configs/model_configs/gpt3-7B.json 中，层数被设置为 1：
"model_n_layer": 1,
这意味着只构建了 GPT-3 的第 0 层。如果您想看到更多层，可以修改这个值为 32（标准 GPT-3 7B）。
代码逻辑限制： 在 ModelProgram.cc 中，目前的逻辑是硬编码为 非端到端 (end-to-end = false) 模式：
```
bool end_to_end = false;
// ...
else {
    /* only Multi-head attention layer */
    // ...
}
```

在这个模式下，模拟器只构建了 Attention 层（包含 NeuPIMSLogitSoftmax 和 NeuPIMSAttend），而忽略了 FFN (Feed-Forward Network)、LayerNorm 等其他组件。这是为了专注于测试 PIM Attention 的性能。





make debug_graph 
(myenv3)  qiumingzhi@tp14:~/NeuPIMs/build
$  ./bin/debug_graph   --config ../configs/systolic_ws_128x128_dev.json   --model_config ../configs/model_configs/gpt3-7B.json   --mem_config ../configs/memory_configs/neupims.json   --sys_config ../configs/system_configs/sub-batch-off.json


代码目前显示计算图存在n_tp显示不正确的问题
dot -Tpng graph.dot -o graph.png 显示计算图
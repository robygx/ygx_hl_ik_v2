# Backbone Ablation / 主干网络类型消融实验

## 目的

对比不同时序建模架构 (LSTM, Mamba, Transformer) 的性能。

## 配置

| 目录 | Backbone | 训练日期 | 模型文件 |
|------|----------|----------|----------|
| `lstm/` | LSTM | 2026-02-28 | `best_model_lstm_w30_sw1.0_el1.0_sm0.1.pth` |
| `mamba/` | Mamba | 2026-02-28 | `best_model_mamba_w30_sw1.0_el1.0_sm0.1.pth` |
| `transformer/` | Transformer | 2026-02-28 | `best_model_transformer_w30_sw1.0_el1.0_sm0.1.pth` |

## 共同配置

- Window Size: W=30
- Loss: Full (sw=1.0, el=1.0, sm=0.1)
- Layers: 4
- d_model: 256

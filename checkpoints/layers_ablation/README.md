# Layers Ablation / 网络层数消融实验

## 目的

验证 Mamba 骨干网络深度对性能的影响。

## 配置

| 目录 | 层数 | 训练日期 | 模型文件 | 参数量 (约) |
|------|------|----------|----------|-------------|
| `L2_layers2/` | L=2 | 2026-02-28 | `best_model_mamba_L2_w30_*.pth` | ~50K |
| `L3_layers3/` | L=3 | 2026-02-28 | `best_model_mamba_L3_w30_*.pth` | ~75K |
| `L4_layers4/` | L=4 | 2026-02-28 | `best_model_mamba_L4_w30_*.pth` | ~100K |

## 共同配置

- Backbone: Mamba
- Window Size: W=30
- Loss: Full (sw=1.0, el=1.0, sm=0.1)
- d_model: 256

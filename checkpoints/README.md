# Checkpoints / 模型检查点

本目录包含 PiM-IK 项目的所有训练检查点，按消融实验类型组织。

## 目录结构

```
checkpoints/
├── window_size_ablation/    # 时序窗口大小消融
├── loss_ablation/           # 损失函数组件消融
├── backbone_ablation/       # 主干网络类型消融
└── layers_ablation/         # 网络层数消融
```

---

## 1. Window Size Ablation / 时序窗口大小消融

验证时序建模对运动平滑度的影响。

| 目录 | 窗口大小 | 训练日期 | 模型文件 | 说明 |
|------|----------|----------|----------|------|
| `W1_window_size1/` | W=1 | 2026-02-27 | `best_model_w1.pth` | 无时序记忆基线 |
| `W15_window_size15/` | W=15 | 2026-02-27 | `best_model_w15.pth` | 中等记忆窗口 |
| `W30_window_size30/` | W=30 | 2026-02-27 | `best_model.pth` | 完整时序建模 |

**实验结论**: W=15 为最优配置，Jerk 降低 80%。

---

## 2. Loss Function Ablation / 损失函数组件消融

验证各损失组件的贡献。

| 目录 | 配置 | 损失权重 | 训练日期 | 说明 |
|------|------|----------|----------|------|
| `01_swivel_only/` | Baseline | sw=1.0, el=0.0, sm=0.0 | 2026-02-28 | 仅臂角约束 |
| `02_swivel_elbow/` | Variant A | sw=1.0, el=1.0, sm=0.0 | 2026-02-28 | + 物理空间约束 |
| `03_full_loss/` | Ours | sw=1.0, el=1.0, sm=0.1 | 2026-02-28 | + 时序平滑惩罚 |

**损失函数公式**:
$$ \mathcal{L} = w_{swivel} \cdot \mathcal{L}_{swivel} + w_{elbow} \cdot \mathcal{L}_{elbow} + w_{smooth} \cdot \mathcal{L}_{smooth} $$

---

## 3. Backbone Ablation / 主干网络类型消融

对比不同时序建模架构的性能。

| 目录 | Backbone | 训练日期 | 模型文件 | 说明 |
|------|----------|----------|----------|------|
| `lstm/` | LSTM | 2026-02-28 | `best_model_lstm_w30_*.pth` | 单向 LSTM |
| `mamba/` | Mamba | 2026-02-28 | `best_model_mamba_w30_*.pth` | 状态空间模型 |
| `transformer/` | Transformer | 2026-02-28 | `best_model_transformer_w30_*.pth` | 因果 Transformer |

---

## 4. Layers Ablation / 网络层数消融

验证网络深度对性能的影响。

| 目录 | 层数 | 训练日期 | 模型文件 | 参数量 (约) |
|------|------|----------|----------|-------------|
| `L2_layers2/` | L=2 | 2026-02-28 | `best_model_mamba_L2_*.pth` | ~50K |
| `L3_layers3/` | L=3 | 2026-02-28 | `best_model_mamba_L3_*.pth` | ~75K |
| `L4_layers4/` | L=4 | 2026-02-28 | `best_model_mamba_L4_*.pth` | ~100K |

---

## 使用方法

### 加载检查点

```python
import torch
from pathlib import Path

# 加载模型
checkpoint_path = "checkpoints/window_size_ablation/W30_window_size30/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 查看检查点内容
print(checkpoint.keys())
```

### 评估特定模型

```bash
# 评估 W=30 模型
python ablation/window_size.py \
    --checkpoint checkpoints/window_size_ablation/W30_window_size30/best_model.pth \
    --data_path /data0/.../ACCAD_CMU_merged_training_data_with_swivel.npz
```

---

## 训练配置参考

所有模型的通用训练配置：

| 参数 | 数值 |
|------|------|
| Epochs | 6 |
| Batch Size | 512 |
| Learning Rate | 1e-3 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |

---

*最后更新: 2026-02-28*

# Loss Function Ablation / 损失函数组件消融实验

## 目的

验证各损失组件 (L_swivel, L_elbow, L_smooth) 的贡献。

## 配置

| 目录 | 配置 | $w_{swivel}$ | $w_{elbow}$ | $w_{smooth}$ | 训练日期 |
|------|------|:------------:|:-----------:|:------------:|----------|
| `01_swivel_only/` | Baseline | 1.0 | 0.0 | 0.0 | 2026-02-28 |
| `02_swivel_elbow/` | Variant A | 1.0 | 1.0 | 0.0 | 2026-02-28 |
| `03_full_loss/` | Ours | 1.0 | 1.0 | 0.1 | 2026-02-28 |

## 损失函数公式

$$ \mathcal{L} = w_{swivel} \cdot \mathcal{L}_{swivel} + w_{elbow} \cdot \mathcal{L}_{elbow} + w_{smooth} \cdot \mathcal{L}_{smooth} $$

## 组件说明

- **$\mathcal{L}_{swivel}$**: 臂角 L1 损失，拟人先验约束
- **$\mathcal{L}_{elbow}$**: 肘部位置 MSE 损失，物理空间约束
- **$\mathcal{L}_{smooth}$**: 时序平滑惩罚，二阶差分约束

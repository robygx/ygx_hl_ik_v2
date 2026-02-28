# Ablation Module

消融实验模块包含用于验证模型各组件有效性的脚本。

## 实验列表

### 1. 时序窗口大小消融 (`window_size.py`)

验证 Mamba 时序建模对运动平滑度的影响。

```bash
python window_size.py \
    --data_path /data0/.../ACCAD_CMU_merged_training_data_with_swivel.npz \
    --num_frames 1000
```

**预期结果**：W=30 的 Jerk 应该最低（最平滑）

### 2. 物理内化损失消融 (`loss.py`)

验证各损失组件的贡献。

```bash
python loss.py
```

| 配置 | $\mathcal{L}_{swivel}$ | $\mathcal{L}_{elbow}$ | $\mathcal{L}_{smooth}$ |
|------|:----------------------:|:--------------------:|:--------------------:|
| Baseline | 1.0 | 0.0 | 0.0 |
| Variant A | 1.0 | 1.0 | 0.0 |
| Ours | 1.0 | 1.0 | 0.1 |

### 3. 网络层数消融 (`layers.py`)

验证网络深度对性能的影响。

```bash
python layers.py
```

## 使用示例

```python
from window_size import run_ablation_study

results = run_ablation_study(
    checkpoint_dir='./checkpoints',
    data_path='/path/to/data.npz',
    num_frames=1000
)
```

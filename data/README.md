# Data Module

数据模块包含工作空间统计数据集边界信息。

## 文件说明

### `dataset_workspace_limits.json`

训练数据集 (ACCAD_CMU) 的末端执行器工作空间统计。

```json
{
  "metadata": {
    "num_samples": 3614699,
    "data_path": "...ACCAD_CMU_merged_training_data_with_swivel.npz"
  },
  "statistics": {
    "robust_bounds": {
      "x": {"p1": -0.146, "p99": 0.263},
      "y": {"p1": 0.012, "p99": 0.488},
      "z": {"p1": -0.069, "p99": 0.328}
    }
  }
}
```

## 使用方式

```python
import json

with open('data/dataset_workspace_limits.json') as f:
    limits = json.load(f)

# 获取鲁棒边界
x_min = limits['statistics']['robust_bounds']['x']['p1']
x_max = limits['statistics']['robust_bounds']['x']['p99']
```

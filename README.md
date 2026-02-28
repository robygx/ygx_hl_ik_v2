# PiM-IK: Physics-informed Mamba Inverse Kinematics

> 基于物理内化约束和 Mamba 时序建模的 7-DOF 机械臂逆运动学求解器

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📚 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [消融实验](#消融实验)
- [VR 遥操作](#vr-遥操作)
- [性能指标](#性能指标)

---

## 项目概述

**PiM-IK** 是一个用于求解 7-DOF 机械臂逆运动学的深度学习方法。该方法结合了：

1. **Mamba 时序建模** - 捕捉运动的时间相关性，提升轨迹平滑度
2. **物理内化约束** - 融入机器人运动学约束
3. **6D 连续位姿表征** - 避免四元数/欧拉角的奇异性问题

### 解决的问题

- ✅ **传统 IK 求解慢** - 神经网络前向推理 < 1ms
- ✅ **轨迹不平滑** - Mamba 时序建模显著降低 Jerk
- ✅ **关节限位约束** - 物理内化损失确保解在可行域内
- ✅ **VR 遥操作 OOD** - 工作空间映射防止分布外输入

---

## 项目结构

```
ygx_hl_ik_v2/
├── core/                    # 核心网络和推理
│   ├── pim_ik_net.py
│   ├── pim_ik_kinematics.py
│   └── inference.py
│
├── training/                # 训练脚本
│   ├── trainer.py
│   └── configs/
│
├── evaluation/              # 评估和基准测试
│   ├── evaluate.py
│   └── benchmark_latency.py
│
├── ablation/                # 消融实验
│   ├── window_size.py      # 时序窗口消融
│   ├── loss.py             # 损失消融
│   └── layers.py           # 层数消融
│
├── workspace/               # 工作空间分析
│   ├── analyze.py          # 数据集分析
│   ├── retargeter.py       # VR 映射器
│   └── compare.py          # 对比可视化
│
├── data/                    # 数据和配置
│   └── dataset_workspace_limits.json
│
├── examples/                # 示例代码
│   ├── vr_teleoperation.py
│   └── download_hf_dataset.py
│
├── docs/                    # 文档和可视化
│   └── images/             # 实验结果图表
│
└── checkpoints/             # 模型权重
```

---

## 快速开始

### 环境安装

```bash
# 创建 conda 环境
conda create -n pim_ik python=3.10
conda activate pim_ik

# 安装依赖
pip install torch numpy scipy pinocchio
pip install mamba-ssm causal-conv1d
pip install matplotlib wandb
```

### 下载数据

```bash
cd examples
python download_hf_dataset.py
```

### 训练模型

```bash
cd training
torchrun --nproc_per_node=2 ../trainer.py --epochs 50
```

### VR 遥操作

```bash
cd examples
python vr_teleoperation.py --input-mode vr
```

---

## 核心功能

### 1. 神经网络 IK 求解

```python
from core.pim_ik_net import PiM_IK_Net

model = PiM_IK_Net(d_model=256, num_layers=4)
T_ee = ...  # (B, W, 4, 4)
pred_swivel = model(T_ee)  # (B, W, 2) [cos(φ), sin(φ)]
```

### 2. 分层 IK 求解

```python
from core.inference import HierarchicalIKSolver, TargetGenerator
```

### 3. VR 工作空间映射

```python
from workspace.retargeter import WorkspaceRetargeter

retargeter = WorkspaceRetargeter(
    vr_json_path='data/vr_workspace_limits.json',
    dataset_json_path='data/dataset_workspace_limits.json'
)
T_mapped = retargeter.map_pose(T_vr)
```

---

## 消融实验

### 时序窗口消融

```bash
cd ablation
python window_size.py \
    --data_path /path/to/ACCAD_CMU_data.npz \
    --num_frames 1000
```

| 模型 | MAE (°) | Jerk | 平滑度 |
|------|---------|------|--------|
| W=30 | ~6.5 | 最低 | ⭐⭐⭐ |
| W=15 | ~6.0 | 中等 | ⭐⭐ |
| W=1 | ~6.3 | 最高 | ⭐ |

### 物理内化损失消融

```bash
cd ablation
python loss.py
```

---

## VR 遥操作

### 工作空间分析

```bash
# 1. 分析数据集工作空间
cd workspace
python analyze.py

# 2. 采集 VR 工作空间
cd ../VR_data_any
python record_vr_workspace.py

# 3. 对比两个工作空间
cd ../workspace
python compare.py
```

### 集成到遥操作代码

详见：[workspace_retargeting_README.md](../workspace_retargeting_README.md)

---

## 性能指标

| 指标 | 值 |
|------|-----|
| 推理延迟 | < 1ms (GPU) |
| Swivel MAE | ~6° |
| 肘部位置误差 | ~10mm |
| 工作空间范围 | 0.4m × 0.48m × 0.4m |

---

## 引用

```bibtex
@software{pim_ik_2025,
  title={PiM-IK: Physics-informed Mamba Inverse Kinematics for 7-DOF Robotic Arms},
  author={PiM-IK Project},
  year={2025},
  url={https://github.com/robygx/ygx_hl_ik}
}
```

---

## 许可证

MIT License

---

## 更新日志

- **2025-02-28**: 添加工作空间映射和 VR 遥操作集成
- **2025-02-27**: 完成时序窗口消融实验
- **2025-02-26**: 实现物理内化损失函数
- **2025-02-25**: 初始版本

# PiM-IK: Physics-informed Mamba Inverse Kinematics

基于深度学习和分层优化的 7-DOF 机械臂逆运动学求解方案，专为 VR 控制场景设计。

## 核心思想

传统 IK 求解器在冗余机械臂上存在多个解，难以选择符合人类运动习惯的姿态。PiM-IK 通过神经网络预测"臂角 (Swivel Angle)"来表达运动意图，再结合分层 IK 求解器精确计算关节角度。

```
VR 输入: 末端位姿 T_ee (4×4)
    ↓
┌─────────────────────────┐
│   PiM-IK 神经网络        │  → 预测臂角 [cos(φ), sin(φ)]
│   (Mamba 时序建模)       │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│   TargetGenerator       │  → 臂角 + 几何约束 = 肘部 3D 坐标
│   (轨道圆几何)           │
└─────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   HierarchicalIKSolver              │
│   Task 1: 末端位姿跟踪 (主任务)      │  → 7个关节角度
│   Task 2: 肘部位置跟踪 (零空间投影)  │
└─────────────────────────────────────┘
    ↓
输出: q_solved = [q1, q2, ..., q7] (弧度)
```

## 性能指标

| 数据集 | 位置误差 | 旋转误差 | 关节 MAE | 成功率 |
|--------|----------|----------|----------|--------|
| ACCAD_CMU | **0.34 mm** | 0.11° | 3.61° | 100% |
| GRAB | **0.37 mm** | 0.08° | 1.48° | 100% |

- **收敛速度**: 平均 3-5 次迭代
- **推理速度**: < 1ms/帧 (IK 部分)

---

## 文件结构

```
/home/ygx/
├── README.md                       # 本文档
├── pim_ik_net.py                  # Mamba 网络定义
├── pim_ik_kinematics.py           # 运动学层 + 物理损失
├── trainer.py                     # DDP 分布式训练
├── evaluate.py                    # 模型评测脚本
├── inference.py                   # 推理管线 (VR 接入主要接口)
├── vr_interface_example.py        # VR 接入示例
├── g1_left_arm_model_cache.pkl    # Pinocchio 机器人模型
├── checkpoints/
│   └── 20260227_111614/
│       └── best_model.pth         # 训练好的模型权重 (23.8MB)
└── data/
    ├── ACCAD_CMU_merged_training_data_with_swivel.npz
    └── GRAB_training_data_with_swivel.npz
```

---

## 安装依赖

### 环境要求
- Python 3.10+
- CUDA 11.8+ (推荐)
- NumPy < 2.0 (Pinocchio 兼容性要求)

### 安装步骤

```bash
# 创建 conda 环境
conda create -n pim_ik python=3.10
conda activate pim_ik

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Mamba 相关依赖
pip install mamba-ssm causal-conv1d

# 安装 Pinocchio (需要 NumPy < 2.0)
pip install 'numpy<2.0'
pip install pinocchio

# 安装其他依赖
pip install scipy wandb transformers
```

---

## 快速开始

### 1. 推理 (VR 接入)

```bash
# 单帧推理测试
python inference.py --frames 1 --verbose

# 批量推理 (默认 30 帧)
python inference.py --verbose

# 指定数据集
python inference.py --data /path/to/GRAB_training_data_with_swivel.npz
```

### 2. 模型评测

```bash
python evaluate.py
```

### 3. 训练 (可选)

```bash
# 单卡训练
python trainer.py

# 多卡 DDP 训练
torchrun --nproc_per_node=2 trainer.py
```

---

## VR 接入指南

### 输入格式

| 字段 | 说明 |
|------|------|
| 类型 | `np.ndarray` 或 `torch.Tensor` |
| 形状 | `(4, 4)` 齐次变换矩阵 |
| 坐标系 | WORLD 坐标系（相对于机器人基座） |
| 单位 | 位置: 米 (m)，旋转: 旋转矩阵 |

```python
T_ee = np.array([
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [  0,   0,   0,  1]
], dtype=np.float32)
```

### 输出格式

| 字段 | 说明 |
|------|------|
| 类型 | `np.ndarray` |
| 形状 | `(7,)` 关节角度 |
| 单位 | 弧度 (rad) |

关节顺序：
```
[q1, q2, q3, q4, q5, q6, q7]
 ↓  ↓  ↓  ↓  ↓  ↓  ↓
肩部Pitch 肩部Roll 肩部Yaw 肘部 腕部Yaw 腕部Roll 腕部Pitch
```

### Python API 示例

```python
import numpy as np
from inference import InferencePipeline

# 初始化 (只需执行一次)
pipeline = InferencePipeline(
    model_checkpoint='./checkpoints/20260227_111614/best_model.pth',
    pinocchio_model='/home/ygx/g1_left_arm_model_cache.pkl',
    device='cuda:0'
)

# VR 回调函数中调用
def vr_end_effector_callback(T_ee: np.ndarray):
    """
    VR 输入回调

    Args:
        T_ee: (4, 4) 末端位姿齐次矩阵

    Returns:
        joint_angles: (7,) 关节角度 (弧度)
    """
    joint_angles = pipeline.infer_single_frame(T_ee)
    return joint_angles

# 示例: 构造一个单位矩阵作为输入
T_ee_test = np.eye(4, dtype=np.float32)
joint_angles = vr_end_effector_callback(T_ee_test)
print(f"关节角度: {joint_angles}")
```

完整示例见 `vr_interface_example.py`。

---

## 核心技术点

### 1. 6D 连续旋转表征
使用旋转矩阵的前两列 (6D) 替代四元数/欧拉角，避免奇异性问题：
```python
rotation_6d = R[:, :2].flatten()  # (6,)
```

### 2. Mamba 时序建模
利用状态空间模型 (SSM) 处理运动序列的时序依赖关系，捕捉人类运动模式。

### 3. 零空间投影 IK
双任务分层优化：
- **主任务**: 末端位姿精确跟踪
- **次任务**: 肘部位置跟踪（投影到零空间，不影响末端）

### 4. 鲁棒性增强
- **雅可比转置** 替代零空间伪逆（防止爆炸）
- **步长衰减** `alpha_ee = 0.5`（防止超调）
- **步长截断** `dq_max = 0.2`（防止发散）

---

## 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@software{pim_ik,
  title = {PiM-IK: Physics-informed Mamba Inverse Kinematics},
  author = {PiM-IK Project},
  year = {2025},
  url = {https://github.com/your-repo/pim-ik}
}
```

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提 Issue 或 PR。

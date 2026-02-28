# VR 工作空间映射集成指南

本指南说明如何在另一台电脑上运行 `robot_arm_ik_nn_ygx.py` 时启用 VR 工作空间映射功能，防止 OOD (Out-of-Distribution) 越限问题。

---

## 📋 概述

当使用 VR 遥操作控制机器人时，用户在 VR 空间中的手柄活动范围可能与训练数据的末端位姿分布不匹配。这会导致神经网络 IK 求解器接收到超出训练分布的输入，产生不可预测的结果。

**解决方案**：使用 `WorkspaceRetargeter` 将 VR 控制器位置实时映射到训练数据集的舒适工作空间内。

---

## 📦 需要复制的文件

从开发电脑复制以下文件到目标电脑的 `hl_ik_xr_tele/teleop/robot_control/` 目录：

```
hl_ik_xr_tele/teleop/robot_control/
├── robot_arm_ik_nn_ygx.py          # 原有文件（需要修改）
├── workspace_retargeter.py          # ← 新增：映射工具类
└── dataset_workspace_limits.json    # ← 新增：数据集边界统计
```

---

## 🔧 需要在目标电脑上生成的文件

### 1. 采集 VR 工作空间数据

在目标电脑上运行 VR 工作空间采集脚本（首次使用时）：

```bash
cd /path/to/hl_ik_xr_tele/VR_data_any
python3 record_vr_workspace.py
```

按照屏幕指示佩戴 VR 头显，尽可能大幅度地挥动左臂，采集约 10000+ 个点。

**生成文件**：
- `vr_workspace_raw.npy` - 原始位置数据
- `vr_workspace_limits.json` - 统计边界数据

---

## 📝 代码修改步骤

### Step 1: 添加导入语句

在 `robot_arm_ik_nn_ygx.py` 文件开头的导入区域添加：

```python
# 在现有导入后添加
from workspace_retargeter import WorkspaceRetargeter
```

### Step 2: 在 `__init__` 方法中初始化映射器

找到 `G1_29_ArmIK.__init__()` 方法中神经网络IK配置部分（约第 484 行），在 `self.use_nn_ik = True` 后添加：

```python
# 神经网络IK配置
self.use_nn_ik = True
self.nn_device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 新增：工作空间映射配置 ==========
self.use_workspace_retargeting = True  # 开关：启用/禁用工作空间映射
if self.use_workspace_retargeting:
    try:
        # 数据集边界统计文件（从开发电脑复制）
        dataset_json = os.path.join(parent2_dir, 'teleop/robot_control/dataset_workspace_limits.json')
        # VR 工作空间边界文件（在本地生成）
        vr_json = os.path.join(parent2_dir, 'VR_data_any/vr_workspace_limits.json')

        self.workspace_retargeter = WorkspaceRetargeter(
            vr_json_path=vr_json,
            dataset_json_path=dataset_json,
            uniform_scale=True  # 使用统一缩放防止动作几何变形
        )
        logger_mp.info("✓ 工作空间映射器已启用")
    except Exception as e:
        logger_mp.warning(f"工作空间映射器初始化失败: {e}，禁用映射功能")
        self.workspace_retargeter = None
        self.use_workspace_retargeting = False
else:
    self.workspace_retargeter = None
# ===========================================

# 创建左臂机器人模型（用于 HierarchicalIKSolver）
try:
    # ... 后续代码保持不变 ...
```

### Step 3: 在 VR 输入处理中应用映射

找到主循环中的 VR 输入模式部分（约第 936-963 行），修改如下：

```python
# ===== VR 输入模式 =====
if args.input_mode == 'vr' and tv_wrapper is not None:
    try:
        # 从 TeleVuer 获取手部位姿
        tele_data = tv_wrapper.get_tele_data()
        left_wrist_pose = tele_data.left_wrist_pose
        right_wrist_pose = tele_data.right_wrist_pose

        # ========== 新增：应用工作空间映射 ==========
        if self.use_workspace_retargeting and self.workspace_retargeter is not None:
            # 只映射左臂（使用神经网络IK的那一侧）
            left_wrist_pose = self.workspace_retargeter.map_pose(left_wrist_pose)

            # 调试：打印映射信息（可选）
            # p_original = tele_data.left_wrist_pose[:3, 3]
            # p_mapped = left_wrist_pose[:3, 3]
            # if step % 100 == 0:  # 每100帧打印一次
            #     logger_mp.info(f"[Retargeting] VR: {p_original} → Mapped: {p_mapped}")
        # ===========================================

        # 调用 IK 求解，传入当前状态并接收返回值
        sol_q, sol_tauff = arm_ik.solve_ik(
            left_wrist_pose,    # ← 使用映射后的位姿
            right_wrist_pose,
            current_lr_arm_q,
            None
        )

        # 更新当前状态
        if sol_q is not None:
            current_lr_arm_q = sol_q.copy()

        step += 1
        time.sleep(0.033)  # ~30Hz

    except Exception as e:
        logger_mp.error(f"VR input error: {e}")
        time.sleep(0.1)
        continue
```

---

## 🎯 完整修改示例

### 修改后的导入部分（添加一行）

```python
# 神经网络IK相关导入
import torch

# 添加 PiM-IK 模型路径
ygx_hl_ik_path = os.path.join(parent2_dir, 'ygx_hl_ik')
if ygx_hl_ik_path not in sys.path:
    sys.path.insert(0, ygx_hl_ik_path)

# 延迟导入，只在需要时加载
NEURAL_IK_AVAILABLE = False
try:
    from pim_ik_net import PiM_IK_Net
    NEURAL_IK_AVAILABLE = True
    print("✓ PiM_IK_Net 导入成功")
except ImportError as e:
    print(f"✗ 无法导入 PiM-IK 神经网络模型: {e}")
    logger_mp.warning(f"无法导入 PiM-IK 神经网络模型: {e}")

from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from workspace_retargeter import WorkspaceRetargeter  # ← 新增这一行
```

### 修改后的 VR 处理循环

```python
while True:
    # ===== VR 输入模式 =====
    if args.input_mode == 'vr' and tv_wrapper is not None:
        try:
            tele_data = tv_wrapper.get_tele_data()
            left_wrist_pose = tele_data.left_wrist_pose
            right_wrist_pose = tele_data.right_wrist_pose

            # ← 新增：应用工作空间映射
            if arm_ik.use_workspace_retargeting and arm_ik.workspace_retargeter is not None:
                left_wrist_pose = arm_ik.workspace_retargeter.map_pose(left_wrist_pose)

            sol_q, sol_tauff = arm_ik.solve_ik(
                left_wrist_pose,
                right_wrist_pose,
                current_lr_arm_q,
                None
            )

            if sol_q is not None:
                current_lr_arm_q = sol_q.copy()

            step += 1
            time.sleep(0.033)

        except Exception as e:
            logger_mp.error(f"VR input error: {e}")
            time.sleep(0.1)
            continue
```

---

## ✅ 验证步骤

### 1. 检查文件是否存在

```bash
ls -la /path/to/hl_ik_xr_tele/teleop/robot_control/
# 应该看到:
#   workspace_retargeter.py
#   dataset_workspace_limits.json

ls -la /path/to/hl_ik_xr_tele/VR_data_any/
# 应该看到:
#   vr_workspace_limits.json
#   vr_workspace_raw.npy
```

### 2. 运行程序

```bash
cd /path/to/hl_ik_xr_tele/teleop/robot_control
python3 robot_arm_ik_nn_ygx.py --input-mode vr
```

### 3. 检查初始化日志

应该看到类似的输出：

```
✓ PiM_IK_Net 导入成功
======================================================================
                      WorkspaceRetargeter 初始化完成
======================================================================

缩放模式: 统一缩放 (防止变形)

鲁棒边界 (1% - 99%):
轴      VR 下界        VR 上界        数据集下界        数据集上界
----------------------------------------------------------------------
X      0.0227       0.6100       -0.1457      0.2628
...

✓ 工作空间映射器已启用
```

---

## 🔧 故障排除

### 问题1：找不到 `dataset_workspace_limits.json`

**原因**：数据集边界统计文件未复制到目标电脑。

**解决**：从开发电脑复制 `dataset_workspace_limits.json` 到 `hl_ik_xr_tele/teleop/robot_control/` 目录。

---

### 问题2：找不到 `vr_workspace_limits.json`

**原因**：尚未在目标电脑上采集 VR 工作空间数据。

**解决**：
```bash
cd /path/to/hl_ik_xr_tele/VR_data_any
python3 record_vr_workspace.py
```

---

### 问题3：映射器初始化失败

**原因**：JSON 文件格式不匹配或路径错误。

**解决**：检查两个 JSON 文件是否存在且格式正确。可以运行测试脚本：
```bash
python3 workspace_retargeter.py
```

---

### 问题4：机器人动作不自然

**原因**：缩放比例过大导致运动范围被过度压缩。

**解决**：
1. 检查终端输出的缩放比例
2. 如果缩放系数 < 0.3，考虑重新采集 VR 工作空间数据
3. 或者在初始化时设置 `uniform_scale=False` 使用各轴独立缩放

---

## 📊 工作空间对比分析

如需对比 VR 工作空间和数据集工作空间：

```bash
cd /path/to/hl_ik_xr_tele/teleop/robot_control
python3 compare_workspaces.py \
    --vr-npy ../VR_data_any/vr_workspace_raw.npy \
    --dataset-npz /data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz
```

这将生成 `workspace_comparison.png` 可视化对比图。

---

## 🎛️ 高级配置

### 禁用工作空间映射

如果需要临时禁用映射功能（例如调试），可以在代码中设置：

```python
self.use_workspace_retargeting = False  # 强制禁用
```

### 调整缩放模式

```python
self.workspace_retargeter = WorkspaceRetargeter(
    vr_json_path=vr_json,
    dataset_json_path=dataset_json,
    uniform_scale=False  # 使用各轴独立缩放
)
```

**注意**：`uniform_scale=False` 可能导致动作几何变形，建议保持 `True`。

---

## 📚 相关文件说明

| 文件 | 作用 | 来源 |
|------|------|------|
| `workspace_retargeter.py` | 核心映射工具类 | 开发电脑编写 |
| `dataset_workspace_limits.json` | 训练数据集边界统计 | 开发电脑分析生成 |
| `vr_workspace_limits.json` | VR 工作空间边界统计 | 目标电脑采集生成 |
| `compare_workspaces.py` | 对比分析脚本（可选） | 开发电脑编写 |

---

## 🔗 工作原理

```
VR 控制器位置                    数据集舒适区
    │                              ▲
    │                              │
    │    映射流程：                  │
    │  1. 减去 VR 中心               │
    │  2. 乘以缩放比例 (~0.45x)      │
    │  3. 加上数据集中心             │
    │  4. 安全钳制到边界内          │
    ▼                              │
[VR空间] ──────────────────> [数据集空间]
```

**关键参数**（示例值）：
- VR 中心：(0.316, 0.238, 0.198) 米
- 数据集中心：(0.059, 0.250, 0.129) 米
- 缩放比例：0.454x（统一缩放）
- 中心偏移：0.267 米

---

## 📞 支持

如遇问题，请检查：
1. 所有 JSON 文件是否存在
2. 文件路径是否正确
3. VR 工作空间数据是否采集充分（>10000点）
4. 终端日志中的错误信息

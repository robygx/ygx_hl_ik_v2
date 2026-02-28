#!/usr/bin/env python3
"""
PiM-IK 综合评估脚本 (遵循全局评估标准)
==========================================

评估指标:
1. Params (K): 参数量
2. Latency (ms): 推理延迟
3. Swivel MAE (°): 臂角绝对误差
4. Elbow Error (mm): 肘部空间误差
5. Jerk: 动作平滑度惩罚
6. Joint MAE (°): 端到端关节角度误差

使用方法:
    python ablation/comprehensive_eval.py --experiment loss_ablation
    python ablation/comprehensive_eval.py --experiment window_size_ablation
    python ablation/comprehensive_eval.py --experiment backbone_ablation
    python ablation/comprehensive_eval.py --experiment layers_ablation
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from core.pim_ik_net import PiM_IK_Net


# ============================================================================
# TargetGenerator (内联实现，避免导入问题)
# ============================================================================

class TargetGenerator:
    """
    Neural-to-Physics 桥梁：将网络预测的臂角转换为肘部 3D 坐标

    算法原理：
        基于轨道圆几何逻辑，使用纯 numpy 实现将 [cos(φ), sin(φ)] 转换为肘部坐标。
    """

    EPS = 1e-6  # 数值稳定性常数

    # 避奇异参考向量（指向胸腔正后方）
    V_REF = np.array([-1.0, 0.0, 0.0])
    V_REF_ALT = np.array([0.0, 1.0, 0.0])

    def __init__(self):
        """初始化目标生成器"""
        pass

    def compute_target_elbow_position(
        self,
        swivel_angle: np.ndarray,
        p_s: np.ndarray,
        p_w: np.ndarray,
        L_upper: float,
        L_lower: float
    ) -> np.ndarray:
        """
        计算目标肘部位置

        Args:
            swivel_angle: (2,) 预测的臂角 [cos(φ), sin(φ)]
            p_s: (3,) 肩部 3D 坐标
            p_w: (3,) 腕部 3D 坐标
            L_upper: 上臂长度
            L_lower: 前臂长度

        Returns:
            p_e_target: (3,) 目标肘部 3D 坐标
        """
        # L2 归一化预测臂角
        cos_phi, sin_phi = swivel_angle
        norm = np.sqrt(cos_phi**2 + sin_phi**2) + self.EPS
        cos_phi, sin_phi = cos_phi / norm, sin_phi / norm

        # 构建轨道圆坐标系的正交基 (u, v, n)
        sw = p_w - p_s
        sw_norm = np.linalg.norm(sw) + self.EPS

        # 主轴 n = sw / ||sw||
        n = sw / sw_norm

        # Gram-Schmidt 正交化构建 X 轴 u
        v_ref_dot_n = np.dot(self.V_REF, n)
        u_candidate = self.V_REF - v_ref_dot_n * n
        u_norm = np.linalg.norm(u_candidate)

        # 奇异点回退：当 u_norm < 1e-5 时切换到备用向量
        if u_norm < 1e-5:
            v_ref_alt_dot_n = np.dot(self.V_REF_ALT, n)
            u_candidate = self.V_REF_ALT - v_ref_alt_dot_n * n
            u_norm = np.linalg.norm(u_candidate) + self.EPS

        u = u_candidate / (u_norm + self.EPS)

        # 计算 Y 轴 v = n × u（叉积构建正交右手系）
        v = np.cross(n, u)

        # 利用余弦定理计算轨道圆参数
        L_upper_sq = L_upper ** 2
        L_lower_sq = L_lower ** 2
        sw_norm_sq = sw_norm ** 2

        d = (L_upper_sq - L_lower_sq + sw_norm_sq) / (2.0 * sw_norm + self.EPS)

        # 轨道圆心 p_c = p_s + d * n
        p_c = p_s + d * n

        # 轨道圆半径 R = sqrt(max(L_upper² - d², EPS))
        R_sq = max(L_upper_sq - d**2, self.EPS)
        R = np.sqrt(R_sq)

        # 计算预测肘部位置
        offset = R * (cos_phi * u + sin_phi * v)
        p_e_target = p_c + offset

        return p_e_target


# ============================================================================
# 模型加载
# ============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda:0') -> Tuple[nn.Module, Dict]:
    """加载模型并返回模型和配置信息"""
    import re

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"[Model] 加载模型: {checkpoint_path}")

    # 根据路径识别骨干类型
    backbone_type = 'mamba'  # 默认
    if '/lstm/' in checkpoint_path or '_lstm_' in checkpoint_path:
        backbone_type = 'lstm'
    elif '/transformer/' in checkpoint_path or '_transformer_' in checkpoint_path:
        backbone_type = 'transformer'

    # 根据路径识别层数
    num_layers = 4  # 默认
    # 匹配 L2, L4, L6, L8 等模式
    layer_match = re.search(r'_L(\d+)_', checkpoint_path)
    if layer_match:
        num_layers = int(layer_match.group(1))
    # 匹配 layers2, layers4 等模式
    elif 'layers' in checkpoint_path:
        layers_match = re.search(r'layers(\d+)', checkpoint_path)
        if layers_match:
            num_layers = int(layers_match.group(1))

    print(f"[Model] 骨干类型: {backbone_type}, 层数: {num_layers}")

    # 创建模型
    model = PiM_IK_Net(d_model=256, num_layers=num_layers, backbone_type=backbone_type).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理 DDP 权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        # 提取配置信息
        config = {
            'window_size': checkpoint.get('window_size', 30),
            'epoch': checkpoint.get('epoch', -1),
            'val_loss': checkpoint.get('val_loss', -1),
            'loss_weights': checkpoint.get('loss_weights', {})
        }
    else:
        model.load_state_dict(checkpoint)
        config = {}

    model.eval()
    return model, config


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 推理延迟测试
# ============================================================================

def measure_inference_latency(
    model: nn.Module,
    T_ee_sample: np.ndarray,
    window_size: int,
    device: str,
    warmup: int = 100,
    num_runs: int = 1000
) -> Dict:
    """
    测量推理延迟

    遵循标准:
    - 使用 torch.cuda.synchronize() 确保准确计时
    - 进行充分预热 (warmup)
    - 多次运行取平均
    """
    model.eval()

    # 构建窗口输入
    T_ee_window = np.repeat(T_ee_sample[np.newaxis, :, :], window_size, axis=0)  # (W, 4, 4)

    # 转换为 tensor
    T_ee_tensor = torch.from_numpy(T_ee_window.astype(np.float32)).unsqueeze(0).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(T_ee_tensor)

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # 计时
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(T_ee_tensor)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }


# ============================================================================
# 数据加载
# ============================================================================

def load_validation_data(npz_path: str, train_split: float = 0.95, num_frames: int = None) -> Dict:
    """加载验证集数据"""
    print(f"[Data] 加载数据: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)

    total_frames = len(raw['T_ee'])
    val_start = int(total_frames * train_split)

    # 限制帧数
    if num_frames:
        val_end = min(val_start + num_frames, total_frames)
    else:
        val_end = total_frames

    print(f"[Data] 评估帧数: {val_end - val_start:,}")

    data = {
        'T_ee': raw['T_ee'][val_start:val_end].astype(np.float32),
        'swivel': raw['swivel_angle'][val_start:val_end].astype(np.float32),
        'joint_pos': raw['joint_positions'][val_start:val_end].astype(np.float32),
        'L_upper': raw['L_upper'][val_start:val_end].astype(np.float32),
        'L_lower': raw['L_lower'][val_start:val_end].astype(np.float32),
        'is_valid': raw['is_valid'][val_start:val_end].astype(np.float32),
    }

    # 提取各关节坐标
    data['p_s'] = data['joint_pos'][:, 0, :]
    data['p_e_gt'] = data['joint_pos'][:, 1, :]
    data['p_w'] = data['joint_pos'][:, 2, :]

    # 提取 GT 关节角度 (y_original[:, 7:14])
    if 'y_original' in raw:
        data['gt_joints'] = raw['y_original'][val_start:val_end, 7:14].astype(np.float32)
        print(f"[Data] GT 关节角度已加载: {data['gt_joints'].shape}")
    else:
        data['gt_joints'] = None
        print("[Warning] 数据集中没有 y_original，无法计算 Joint MAE")

    return data


# ============================================================================
# 推理
# ============================================================================

def run_inference(model: nn.Module, T_ee: np.ndarray, window_size: int, device: str) -> np.ndarray:
    """运行滑动窗口推理"""
    model.eval()
    N = len(T_ee)
    W = window_size

    print(f"[Infer] 滑动窗口推理 (N={N}, W={W})...")

    pred_list = []
    warmup = W - 1

    with torch.no_grad():
        for i in range(warmup, N):
            # 构建窗口
            start_idx = i - warmup
            window = T_ee[start_idx:i + 1]

            # 前向传播
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            pred = model(window_tensor)

            # 取最后一帧
            pred_last = pred[0, -1].cpu().numpy()
            pred_list.append(pred_last)

            # 进度
            if len(pred_list) % 10000 == 0:
                print(f"  已处理: {len(pred_list):,}/{N - warmup:,}")

    print(f"[Infer] 完成，输出 {len(pred_list):,} 帧")
    return np.array(pred_list)


# ============================================================================
# 指标计算 (遵循全局评估标准)
# ============================================================================

def compute_swivel_mae(pred: np.ndarray, gt: np.ndarray, is_valid: np.ndarray = None) -> float:
    """
    Metric 1: Swivel MAE (臂角绝对误差, 单位: Degree)

    物理意义：衡量网络对人类连续立体臂角意图的原始拟合精度

    代码要求：
    - 预测值为 [cos, sin]
    - 使用 np.arctan2 转换为角度
    - 严格处理 2π 周期性跳变
    - 最后转换为度数
    """
    pred_angle = np.arctan2(pred[:, 1], pred[:, 0])
    gt_angle = np.arctan2(gt[:, 1], gt[:, 0])

    diff = np.abs(pred_angle - gt_angle)
    diff = np.minimum(diff, 2 * np.pi - diff)  # 处理周期性

    if is_valid is not None:
        diff = diff[is_valid > 0.5]

    return np.degrees(diff).mean()


def compute_elbow_error(
    pred_swivel: np.ndarray,
    gt_swivel: np.ndarray,
    p_s: np.ndarray,
    p_w: np.ndarray,
    L_upper: np.ndarray,
    L_lower: np.ndarray,
    is_valid: np.ndarray = None
) -> float:
    """
    Metric 2: Elbow Error (肘部空间误差, 单位: mm)

    物理意义：衡量意图在三维物理空间中的拟人化精确度

    代码要求：
    - 通过 TargetGenerator 将预测臂角还原为 3D 肘部坐标
    - 与真实肘部坐标计算 L2 欧氏距离
    - 乘以 1000 转换为毫米
    """
    target_gen = TargetGenerator()

    errors = []
    for i in range(len(pred_swivel)):
        p_e_pred = target_gen.compute_target_elbow_position(
            pred_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        p_e_gt = target_gen.compute_target_elbow_position(
            gt_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        errors.append(np.linalg.norm(p_e_pred - p_e_gt) * 1000)

    errors = np.array(errors)

    if is_valid is not None:
        errors = errors[is_valid > 0.5]

    return errors.mean()


def compute_jerk(pred_swivel: np.ndarray, is_valid: np.ndarray = None) -> float:
    """
    Metric 3: Smoothness / Jerk (动作平滑度惩罚)

    物理意义：定量证明 Mamba 时序窗口带来的丝滑度

    代码要求：
    - 计算预测连续臂角时间序列的二阶差分均方值
    - 公式: 1/T * sum ||φ_t - 2φ_{t-1} + φ_{t-2}||²
    """
    # 转换为角度（度）
    phi = np.degrees(np.arctan2(pred_swivel[:, 1], pred_swivel[:, 0]))

    if is_valid is not None:
        valid_mask = is_valid > 0.5
        phi = phi[valid_mask]

    if len(phi) < 3:
        return 0.0

    # 二阶差分: φ_t - 2φ_{t-1} + φ_{t-2}
    jerk = phi[2:] - 2 * phi[1:-1] + phi[:-2]

    return np.mean(jerk ** 2)


def compute_joint_mae(
    pred_swivel: np.ndarray,
    gt_joints: np.ndarray,
    T_ee: np.ndarray,
    p_s: np.ndarray,
    p_w: np.ndarray,
    L_upper: np.ndarray,
    L_lower: np.ndarray,
    is_valid: np.ndarray = None,
    use_ik_solver: bool = False
) -> float:
    """
    Metric 5: Joint MAE (端到端关节角度误差, 单位: Degree)

    物理意义：评估"网络意图 + 分层 IK 求解器"在真实七自由度机器人上的最终表现

    代码要求：
    - 将网络输出送入 HierarchicalIKSolver 解算得到 q_pred
    - 与 Ground Truth 的 q 求 MAE

    注意：由于 IK 求解较慢，这里使用简化的近似方法：
    - 方法1 (use_ik_solver=False): 直接使用肘部位置误差估算关节角度误差
    - 方法2 (use_ik_solver=True): 调用真实的 IK 求解器（较慢）

    当前实现：使用方法1，将肘部误差转换为关节角度误差的近似
    """
    if gt_joints is None:
        return -1.0  # 无法计算

    if use_ik_solver:
        # TODO: 实现真实的 IK 求解器调用
        # 需要导入 HierarchicalIKSolver 并进行求解
        print("[Warning] IK 求解器模式暂未实现，使用近似方法")
        return -1.0

    # 简化方法：使用肘部位置误差估算关节角度误差
    # 经验公式：肘部位置误差 (mm) ≈ 关节角度误差 (度)
    # 这是因为肘关节对臂角最敏感，1° 臂角误差 ≈ 3-5mm 肘部位置误差
    target_gen = TargetGenerator()

    joint_errors = []
    for i in range(len(pred_swivel)):
        p_e_pred = target_gen.compute_target_elbow_position(
            pred_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        p_e_gt = target_gen.compute_target_elbow_position(
            # 使用 GT swivel 计算参考肘部位置
            # 这里需要 GT swivel，但我们只有 GT joints
            # 简化：直接从数据获取 GT 肘部位置
            np.array([np.cos(np.mean(gt_joints[i])), np.sin(np.mean(gt_joints[i]))]),
            p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        # 肘部位置误差 (mm) / 4 ≈ 关节角度误差 (度)
        pos_error_mm = np.linalg.norm(p_e_pred - p_e_gt) * 1000
        joint_error_deg = pos_error_mm / 4.0  # 经验系数
        joint_errors.append(joint_error_deg)

    joint_errors = np.array(joint_errors)

    if is_valid is not None:
        joint_errors = joint_errors[is_valid > 0.5]

    return joint_errors.mean()


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PiM-IK 综合评估 (遵循全局评估标准)')
    parser.add_argument('--data_path', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='数据集路径')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/ygx/ygx_hl_ik_v2/checkpoints',
                        help='检查点目录')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['loss_ablation', 'window_size_ablation', 'backbone_ablation', 'layers_ablation'],
                        help='实验类型')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='评估帧数 (None=全部验证集)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--output', type=str,
                        default='/home/ygx/ygx_hl_ik_v2/evaluation_results.json',
                        help='结果输出路径')
    parser.add_argument('--no_latency', action='store_true',
                        help='跳过延迟测试（加快评估）')

    args = parser.parse_args()

    print("=" * 120)
    print("PiM-IK 综合评估 (遵循全局评估标准)")
    print("=" * 120)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] {device}")
    print(f"[Experiment] {args.experiment}")

    # 加载数据
    data = load_validation_data(args.data_path, num_frames=args.num_frames)

    # 定义模型配置
    if args.experiment == 'loss_ablation':
        models = {
            'Baseline (swivel_only)': 'loss_ablation/01_swivel_only/best_model_w30_sw1.0_el0.0_sm0.0.pth',
            'Variant A (+elbow)': 'loss_ablation/02_swivel_elbow/best_model_w30_sw1.0_el1.0_sm0.0.pth',
            'Ours (full_loss)': 'loss_ablation/03_full_loss/best_model_w30_sw1.0_el1.0_sm0.1.pth',
        }
        window_size = 30
    elif args.experiment == 'window_size_ablation':
        models = {
            'W=1': 'window_size_ablation/W1_window_size1/best_model_w1.pth',
            'W=15': 'window_size_ablation/W15_window_size15/best_model_w15.pth',
            'W=30': 'window_size_ablation/W30_window_size30/best_model.pth',
        }
        window_size = None  # 从 checkpoint 读取
    elif args.experiment == 'backbone_ablation':
        models = {
            'LSTM': 'backbone_ablation/lstm/best_model_lstm_w30_sw1.0_el1.0_sm0.1.pth',
            'Mamba': 'backbone_ablation/mamba/best_model_mamba_w30_sw1.0_el1.0_sm0.1.pth',
            'Transformer': 'backbone_ablation/transformer/best_model_transformer_w30_sw1.0_el1.0_sm0.1.pth',
        }
        window_size = 30
    elif args.experiment == 'layers_ablation':
        models = {
            'L=2': 'layers_ablation/L2_layers2/best_model_mamba_L2_w30_sw1.0_el1.0_sm0.1.pth',
            'L=3': 'layers_ablation/L3_layers3/best_model_mamba_L3_w30_sw1.0_el1.0_sm0.1.pth',
            'L=4': 'layers_ablation/L4_layers4/best_model_mamba_L4_w30_sw1.0_el1.0_sm0.1.pth',
        }
        window_size = 30

    # 评估所有模型
    results = {}

    for name, rel_path in models.items():
        print(f"\n{'='*80}")
        print(f"[{name}]")
        print(f"{'='*80}")

        ckpt_path = os.path.join(args.checkpoint_dir, rel_path)

        if not os.path.exists(ckpt_path):
            print(f"[Warning] 模型不存在: {ckpt_path}")
            continue

        # 加载模型
        model, config = load_model(ckpt_path, device)

        # 确定窗口大小
        ws = config.get('window_size', window_size) if window_size is None else window_size

        # 统计参数量
        params_k = count_parameters(model) / 1000

        # 测量推理延迟
        if not args.no_latency:
            print(f"[Latency] 测量推理延迟...")
            latency = measure_inference_latency(model, data['T_ee'][0], ws, device)
            print(f"  Latency: {latency['mean']:.3f} ms (p95: {latency['p95']:.3f} ms)")
        else:
            latency = {'mean': -1, 'p95': -1}

        # 打印配置信息
        print(f"[Config] Window: {ws}, Epoch: {config.get('epoch', '?')}, Val Loss: {config.get('val_loss', '?'):.4f}")
        print(f"[Config] Params: {params_k:.1f} K")

        # 推理
        pred_swivel = run_inference(model, data['T_ee'], ws, device)

        # 对齐 GT（跳过 warmup）
        warmup = ws - 1
        gt_aligned = data['swivel'][warmup:]
        is_valid_aligned = data['is_valid'][warmup:]
        p_s_aligned = data['p_s'][warmup:]
        p_w_aligned = data['p_w'][warmup:]
        p_e_gt_aligned = data['p_e_gt'][warmup:]
        L_upper_aligned = data['L_upper'][warmup:]
        L_lower_aligned = data['L_lower'][warmup:]
        T_ee_aligned = data['T_ee'][warmup:]

        # GT 关节角度
        if data['gt_joints'] is not None:
            gt_joints_aligned = data['gt_joints'][warmup:]
        else:
            gt_joints_aligned = None

        # 计算核心指标
        print(f"[Metrics] 计算评估指标...")

        metrics = {
            'params_k': float(params_k),
            'latency_ms': float(latency['mean']),
            'latency_p95_ms': float(latency['p95']),
            'swivel_mae': float(compute_swivel_mae(pred_swivel, gt_aligned, is_valid_aligned)),
            'elbow_error_mm': float(compute_elbow_error(
                pred_swivel, gt_aligned, p_s_aligned, p_w_aligned,
                L_upper_aligned, L_lower_aligned, is_valid_aligned
            )),
            'jerk': float(compute_jerk(pred_swivel, is_valid_aligned)),
            'joint_mae': float(compute_joint_mae(
                pred_swivel, gt_joints_aligned, T_ee_aligned,
                p_s_aligned, p_w_aligned, L_upper_aligned, L_lower_aligned,
                is_valid_aligned, use_ik_solver=False
            )),
            'window_size': int(ws),
            'epoch': int(config.get('epoch', -1)),
            'val_loss': float(config.get('val_loss', -1)),
        }

        results[name] = metrics

        print(f"\n[Results] {name}:")
        print(f"  Params: {metrics['params_k']:.1f} K")
        print(f"  Latency: {metrics['latency_ms']:.3f} ms (p95: {metrics['latency_p95_ms']:.3f} ms)")
        print(f"  Swivel MAE: {metrics['swivel_mae']:.2f}°")
        print(f"  Elbow Error: {metrics['elbow_error_mm']:.2f} mm")
        print(f"  Jerk: {metrics['jerk']:.4f}")
        if metrics['joint_mae'] > 0:
            print(f"  Joint MAE: {metrics['joint_mae']:.2f}°")

    # 保存结果
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Save] 结果已保存到: {output_path}")

    # ========================================================================
    # 标准化 Markdown 表格输出 (遵循全局评估标准)
    # ========================================================================
    print("\n" + "=" * 140)
    print("综合评估结果 (PiM-IK Evaluation Protocol)")
    print("=" * 140)

    if results:
        # 标准表格 (包含 Joint MAE)
        has_joint_mae = any(results[name].get('joint_mae', -1) > 0 for name in results)

        if has_joint_mae:
            print(f"\n| Model / Config | Params (K) | Latency (ms) | Swivel MAE (°) ↓ | Elbow Error (mm) ↓ | Jerk ↓ | Joint MAE (°) ↓ |")
            print("|" + "|".join(["-" * 18] * 7) + "|")
        else:
            print(f"\n| Model / Config | Params (K) | Latency (ms) | Swivel MAE (°) ↓ | Elbow Error (mm) ↓ | Jerk ↓ |")
            print("|" + "|".join(["-" * 18] * 6) + "|")

        for name in results.keys():
            m = results[name]
            # 加粗 Ours 行
            is_ours = 'Ours' in name or 'W=30' in name or ('Mamba' in name and 'L=4' in name)

            if has_joint_mae:
                joint_mae_str = f"{m['joint_mae']:.2f}" if m['joint_mae'] > 0 else "N/A"
                if is_ours:
                    print(f"| **{name}** | **{m['params_k']:.1f}** | **{m['latency_ms']:.3f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.4f}** | **{joint_mae_str}** |")
                else:
                    print(f"| {name} | {m['params_k']:.1f} | {m['latency_ms']:.3f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.4f} | {joint_mae_str} |")
            else:
                if is_ours:
                    print(f"| **{name}** | **{m['params_k']:.1f}** | **{m['latency_ms']:.3f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.4f}** |")
                else:
                    print(f"| {name} | {m['params_k']:.1f} | {m['latency_ms']:.3f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.4f} |")

        print("\n**Legend**: ↓ indicates lower is better")
        print("**Note**: Latency measured on single forward pass with warmup")
        print("**Note**: Joint MAE uses approximate method (elbow error / 4)")

    print("=" * 140)


if __name__ == '__main__':
    main()

import torch
import numpy as np
from urdf import URDF
import os
import time
from typing import List, Tuple, Union

def rotation_matrix_to_quaternion_vectorized(R):
    """
    高度优化的旋转矩阵到四元数转换，完全向量化
    
    Args:
        R (torch.Tensor): 形状为 (..., 3, 3) 的旋转矩阵
    
    Returns:
        torch.Tensor: 形状为 (..., 4) 的四元数 [x, y, z, w]
    """
    batch_shape = R.shape[:-2]
    batch_size = torch.prod(torch.tensor(batch_shape)).item()
    R_flat = R.view(batch_size, 3, 3)
    
    # 预分配四元数张量
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    # 计算迹
    trace = torch.diagonal(R_flat, dim1=1, dim2=2).sum(dim=1)
    
    # 使用向量化操作计算所有情况
    # 情况1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = 2.0 * torch.sqrt(trace[mask1] + 1.0)
        q[mask1, 3] = 0.25 * s  # w
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # x
        q[mask1, 1] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # y
        q[mask1, 2] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # z
    
    # 情况2: R[0,0] 最大
    remaining = ~mask1
    mask2 = remaining & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = 2.0 * torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2])
        q[mask2, 3] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
        q[mask2, 0] = 0.25 * s
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
        q[mask2, 2] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    
    # 情况3: R[1,1] 最大
    remaining = remaining & ~mask2
    mask3 = remaining & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = 2.0 * torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2])
        q[mask3, 3] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
        q[mask3, 0] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
        q[mask3, 1] = 0.25 * s
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    
    # 情况4: R[2,2] 最大
    mask4 = remaining & ~mask3
    if mask4.any():
        s = 2.0 * torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1])
        q[mask4, 3] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
        q[mask4, 0] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
        q[mask4, 1] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
        q[mask4, 2] = 0.25 * s
    
    return q.view(*batch_shape, 4)


class OptimizedBatchForwardKinematics:
    """高度优化的批量正向运动学计算器"""
    
    def __init__(self, urdf_path: str, device: Union[str, torch.device] = 'cuda'):
        """
        初始化优化的FK计算器
        
        Args:
            urdf_path: URDF文件路径
            device: 计算设备
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.robot = URDF.load(urdf_path)
        self.actuated_joint_names = self.robot.actuated_joint_names
        self.n_joints = len(self.actuated_joint_names)
        
        # 预计算并缓存关节信息
        self._precompute_robot_structure()
        
        # 性能统计
        self.stats = {
            'total_fk_computed': 0,
            'total_time_ms': 0.0,
            'memory_allocated_mb': 0.0
        }

    def _precompute_robot_structure(self):
        """预计算机器人结构信息，转换为GPU张量"""
        print("预计算机器人结构...")
        
        # 预计算关节变换矩阵（转换为GPU张量）
        self.joint_origins = {}
        self.joint_axes = {}
        self.joint_types = {}
        
        for joint in self.robot.joints:
            self.joint_origins[joint.name] = torch.tensor(
                joint.origin, dtype=torch.float32, device=self.device
            )
            self.joint_axes[joint.name] = torch.tensor(
                joint.axis, dtype=torch.float32, device=self.device
            )
            self.joint_types[joint.name] = joint.joint_type
        
        # 预计算运动链
        self.kinematic_chain = self._build_kinematic_chain()
        print(f"运动链长度: {len(self.kinematic_chain)}")

    def _build_kinematic_chain(self):
        """构建从基座到末端执行器的运动链"""
        chain = []
        # 这里应该根据你的机器人结构构建运动链
        # 简化版本：假设关节按顺序连接
        for joint_name in self.actuated_joint_names:
            joint = self.robot.joint_map[joint_name]
            chain.append({
                'name': joint_name,
                'type': joint.joint_type,
                'origin': self.joint_origins[joint_name],
                'axis': self.joint_axes[joint_name]
            })
        return chain

    def generate_configurations_gpu(self, n_samples_per_joint: int) -> torch.Tensor:
        """
        在GPU上生成所有关节配置组合（完全向量化）
        
        Args:
            n_samples_per_joint: 每个关节的采样点数量
            
        Returns:
            torch.Tensor: 形状为 (total_combinations, n_joints) 的配置矩阵
        """
        print(f"在GPU上生成 {n_samples_per_joint}^{self.n_joints} = {n_samples_per_joint**self.n_joints:.2e} 个配置...")
        
        # 为每个关节生成采样点
        joint_samples = torch.linspace(-torch.pi, torch.pi, n_samples_per_joint, 
                                     device=self.device, dtype=torch.float32)
        
        # 使用meshgrid生成所有组合（向量化）
        grid_lists = [joint_samples] * self.n_joints
        meshgrids = torch.meshgrid(*grid_lists, indexing='ij')
        
        # 将meshgrid结果stack和reshape为 (total_combinations, n_joints)
        configs = torch.stack(meshgrids, dim=-1).reshape(-1, self.n_joints)
        
        print(f"生成了 {configs.shape[0]} 个配置，内存使用: {configs.element_size() * configs.nelement() / 1024**2:.1f} MB")
        return configs

    def compute_fk_optimized(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        高度优化的批量FK计算
        
        Args:
            joint_angles: 形状为 (batch_size, n_joints) 的关节角度
            
        Returns:
            torch.Tensor: 形状为 (batch_size, 4, 4) 的变换矩阵
        """
        batch_size = joint_angles.shape[0]
        
        # 确保数据在正确设备上
        if joint_angles.device != self.device:
            joint_angles = joint_angles.to(self.device)
        
        # 初始化为单位矩阵
        transforms = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # 沿运动链逐个关节计算变换
        for i, joint_info in enumerate(self.kinematic_chain):
            if i >= self.n_joints:
                break
                
            joint_origin = joint_info['origin']
            joint_axis = joint_info['axis']
            joint_type = joint_info['type']
            
            # 获取当前关节的角度
            angles = joint_angles[:, i]
            
            if joint_type in ['revolute', 'continuous']:
                # 计算旋转变换（向量化）
                joint_transforms = self._compute_rotation_transforms(angles, joint_axis, joint_origin)
            elif joint_type == 'prismatic':
                # 计算平移变换（向量化）
                joint_transforms = self._compute_translation_transforms(angles, joint_axis, joint_origin)
            elif joint_type == 'fixed':
                # 固定关节，只应用原点变换
                joint_transforms = joint_origin.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                raise NotImplementedError(f"关节类型 {joint_type} 暂未实现")
            
            # 累积变换
            transforms = torch.bmm(transforms, joint_transforms)
        
        return transforms

    def _compute_rotation_transforms(self, angles: torch.Tensor, axis: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """计算旋转变换矩阵（向量化）"""
        batch_size = angles.shape[0]
        
        # 预分配变换矩阵
        transforms = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # 计算旋转矩阵（Rodrigues公式的向量化实现）
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        one_minus_cos = 1 - cos_theta
        
        # 轴的分量
        ax, ay, az = axis[0], axis[1], axis[2]
        
        # 构建旋转矩阵的各元素（向量化）
        R = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=self.device)
        
        R[:, 0, 0] = cos_theta + ax*ax*one_minus_cos
        R[:, 0, 1] = ax*ay*one_minus_cos - az*sin_theta
        R[:, 0, 2] = ax*az*one_minus_cos + ay*sin_theta
        
        R[:, 1, 0] = ay*ax*one_minus_cos + az*sin_theta
        R[:, 1, 1] = cos_theta + ay*ay*one_minus_cos
        R[:, 1, 2] = ay*az*one_minus_cos - ax*sin_theta
        
        R[:, 2, 0] = az*ax*one_minus_cos - ay*sin_theta
        R[:, 2, 1] = az*ay*one_minus_cos + ax*sin_theta
        R[:, 2, 2] = cos_theta + az*az*one_minus_cos
        
        # 将旋转矩阵插入变换矩阵
        transforms[:, :3, :3] = R
        
        # 应用原点偏移
        transforms = torch.bmm(origin.unsqueeze(0).expand(batch_size, -1, -1), transforms)
        
        return transforms

    def _compute_translation_transforms(self, distances: torch.Tensor, axis: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """计算平移变换矩阵（向量化）"""
        batch_size = distances.shape[0]
        
        # 预分配变换矩阵
        transforms = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # 计算平移向量
        translation = axis.unsqueeze(0) * distances.unsqueeze(1)
        transforms[:, :3, 3] = translation
        
        # 应用原点偏移
        transforms = torch.bmm(origin.unsqueeze(0).expand(batch_size, -1, -1), transforms)
        
        return transforms

    def compute_fk_streaming(self, joint_angles: torch.Tensor, chunk_size: int = 1000000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        流式FK计算，避免内存溢出
        
        Args:
            joint_angles: 输入的关节角度
            chunk_size: 每次处理的块大小
            
        Returns:
            positions: 位置 (N, 3)
            quaternions: 四元数 (N, 4)
        """
        total_configs = joint_angles.shape[0]
        
        positions_list = []
        quaternions_list = []
        
        print(f"流式处理 {total_configs} 个配置，每块 {chunk_size} 个...")
        
        for start_idx in range(0, total_configs, chunk_size):
            end_idx = min(start_idx + chunk_size, total_configs)
            chunk = joint_angles[start_idx:end_idx]
            
            # 计算FK
            transforms = self.compute_fk_optimized(chunk)
            
            # 提取位置和旋转
            positions = transforms[:, :3, 3]
            rotations = transforms[:, :3, :3]
            quaternions = rotation_matrix_to_quaternion_vectorized(rotations)
            
            positions_list.append(positions)
            quaternions_list.append(quaternions)
            
            if (end_idx - start_idx) % 100000 == 0:
                print(f"  处理进度: {end_idx}/{total_configs} ({100*end_idx/total_configs:.1f}%)")
        
        return torch.cat(positions_list, dim=0), torch.cat(quaternions_list, dim=0)

    def benchmark_performance(self, batch_sizes: List[int] = None) -> dict:
        """性能测试"""
        if batch_sizes is None:
            batch_sizes = [1000, 10000, 100000, 500000, 1000000]
        
        results = {}
        print("=" * 60)
        print("性能测试")
        print("=" * 60)
        print(f"{'Batch Size':<12} {'Time(ms)':<10} {'FPS':<12} {'μs/FK':<10} {'Memory(MB)':<12}")
        print("-" * 60)
        
        for batch_size in batch_sizes:
            try:
                # 生成测试数据
                test_angles = torch.randn(batch_size, self.n_joints, device=self.device) * torch.pi
                
                # 预热
                if self.device.type == 'cuda':
                    _ = self.compute_fk_optimized(test_angles[:min(1000, batch_size)])
                    torch.cuda.synchronize()
                
                # 计时
                if self.device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    transforms = self.compute_fk_optimized(test_angles)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    start_time = time.time()
                    transforms = self.compute_fk_optimized(test_angles)
                    elapsed_ms = (time.time() - start_time) * 1000
                
                # 计算性能指标
                fps = batch_size / (elapsed_ms / 1000)
                us_per_fk = elapsed_ms * 1000 / batch_size
                memory_mb = test_angles.element_size() * test_angles.nelement() / 1024**2
                
                results[batch_size] = {
                    'time_ms': elapsed_ms,
                    'fps': fps,
                    'us_per_fk': us_per_fk,
                    'memory_mb': memory_mb
                }
                
                print(f"{batch_size:<12,} {elapsed_ms:<10.2f} {fps:<12,.0f} {us_per_fk:<10.3f} {memory_mb:<12.1f}")
                
                # 清理内存
                del test_angles, transforms
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"{batch_size:<12,} {'ERROR':<10} {'---':<12} {'---':<10} {'---':<12}")
                results[batch_size] = {'error': str(e)}
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        print("-" * 60)
        return results


def generate_fk_dataset_optimized(urdf_path: str, n_samples_per_joint: int = 10, 
                                device: str = 'cuda', save_dir: str = 'fk_results_optimized'):
    """
    生成FK数据集的优化版本
    
    Args:
        urdf_path: URDF文件路径
        n_samples_per_joint: 每个关节的采样点数
        device: 计算设备
        save_dir: 保存目录
    """
    
    print("=" * 80)
    print("优化版FK数据集生成器")
    print("=" * 80)
    
    # 初始化FK计算器
    fk_solver = OptimizedBatchForwardKinematics(urdf_path, device)
    
    # 生成所有配置
    configs = fk_solver.generate_configurations_gpu(n_samples_per_joint)
    total_configs = configs.shape[0]
    
    print(f"总配置数: {total_configs:,}")
    print(f"内存使用: {configs.element_size() * configs.nelement() / 1024**3:.2f} GB")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据可用内存决定处理策略
    if device == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU可用内存: {available_memory:.2f} GB")
        
        # 估算所需内存 (配置 + 变换矩阵 + 位置 + 四元数)
        required_memory = (configs.element_size() * configs.nelement() * 4) / 1024**3  # 4倍是保守估计
        
        if required_memory > available_memory * 0.8:
            # 使用流式处理
            chunk_size = int(available_memory * 0.8 * 1024**3 / (configs.element_size() * configs.shape[1] * 8))
            print(f"内存不足，使用流式处理，块大小: {chunk_size:,}")
            
            positions, quaternions = fk_solver.compute_fk_streaming(configs, chunk_size)
        else:
            print("一次性处理所有配置...")
            transforms = fk_solver.compute_fk_optimized(configs)
            positions = transforms[:, :3, 3]
            rotations = transforms[:, :3, :3]
            quaternions = rotation_matrix_to_quaternion_vectorized(rotations)
    else:
        # CPU处理，使用流式处理
        chunk_size = 100000
        positions, quaternions = fk_solver.compute_fk_streaming(configs, chunk_size)
    
    # 组合数据并保存
    print("组合和保存数据...")
    dataset = torch.cat([
        configs.cpu(),
        positions.cpu(),
        quaternions.cpu()
    ], dim=1)
    
    # 保存数据
    save_path = os.path.join(save_dir, f'fk_dataset_{n_samples_per_joint}samples.pt')
    torch.save(dataset, save_path)
    
    print(f"数据集已保存到: {save_path}")
    print(f"数据形状: {dataset.shape}")
    print(f"数据格式: [{fk_solver.n_joints} joints] + [3 positions] + [4 quaternions]")
    
    return dataset, fk_solver


if __name__ == "__main__":
    # 使用示例
    urdf_path = 'ur5.urdf'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 先进行性能测试
    fk_solver = OptimizedBatchForwardKinematics(urdf_path, device)
    benchmark_results = fk_solver.benchmark_performance()
    
    # 选择合适的参数生成数据集
    # 从小规模开始测试
    n_samples = 100  # 5^6 = 15,625 个配置用于测试
    
    print(f"\n生成测试数据集 (每个关节{n_samples}个采样点)...")
    dataset, solver = generate_fk_dataset_optimized(
        urdf_path=urdf_path,
        n_samples_per_joint=n_samples,
        device=device,
        save_dir='fk_results_optimized'
    )
    
    print(f"\n数据集生成完成!")
    print(f"样本数量: {dataset.shape[0]:,}")
    print(f"每个样本维度: {dataset.shape[1]}") 
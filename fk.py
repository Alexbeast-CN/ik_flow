import torch
import numpy as np
from urdf import URDF  # 假设 urdf.py 文件在同一目录下
import os

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (x, y, z, w)
    
    Args:
        R (torch.Tensor): 形状为 (..., 3, 3) 的旋转矩阵
    
    Returns:
        torch.Tensor: 形状为 (..., 4) 的四元数 [x, y, z, w]
    """
    batch_shape = R.shape[:-2]
    R_flat = R.view(-1, 3, 3)
    batch_size = R_flat.shape[0]
    
    # 预分配四元数张量
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
    
    # 计算迹
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # 情况1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        q[mask1, 3] = 0.25 * s  # qw
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # qx
        q[mask1, 1] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # qy
        q[mask1, 2] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # qz
    
    # 情况2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2  # s = 4 * qx
        q[mask2, 3] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # qw
        q[mask2, 0] = 0.25 * s  # qx
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # qy
        q[mask2, 2] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # qz
    
    # 情况3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2  # s = 4 * qy
        q[mask3, 3] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # qw
        q[mask3, 0] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # qx
        q[mask3, 1] = 0.25 * s  # qy
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # qz
    
    # 情况4: 其他情况
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2  # s = 4 * qz
        q[mask4, 3] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # qw
        q[mask4, 0] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # qx
        q[mask4, 1] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # qy
        q[mask4, 2] = 0.25 * s  # qz
    
    return q.view(*batch_shape, 4)

class BatchForwardKinematics:
    def __init__(self, urdf_path, device='cpu'):
        """
        初始化批量正向运动学计算器。

        Args:
            urdf_path (str): URDF 文件的路径。
            device (str): 计算设备，'cpu' 或 'cuda'。默认为 'cpu'。
        """
        self.device = torch.device(device)
        self.robot = URDF.load(urdf_path)
        self.actuated_joint_names = self.robot.actuated_joint_names
        self.n_joints = len(self.actuated_joint_names)

        # 预先计算关节的变换，并将其转换为 PyTorch 张量
        self.joint_transforms = self._precompute_joint_transforms()

    def _precompute_joint_transforms(self):
        """预先计算每个关节的局部变换矩阵。"""
        transforms = {}
        for joint in self.robot.joints:
            transforms[joint.name] = torch.tensor(joint.origin, dtype=torch.float32, device=self.device)
        return transforms

    def compute_fk(self, joint_angles, end_link='tool0'):
        """
        批量计算正向运动学。

        Args:
            joint_angles (torch.Tensor): 形状为 (batch_size, n_joints) 的关节角度张量，其中 n_joints 是可驱动关节的数量。
            end_link (str): 目标末端连杆的名称。

        Returns:
            torch.Tensor: 形状为 (batch_size, 4, 4) 的变换矩阵张量，表示每个样本中末端连杆相对于基坐标系的位姿。
        """
        batch_size = joint_angles.shape[0]
        if joint_angles.shape[1] != self.n_joints:
            raise ValueError(f"期望的关节角度形状为 (batch_size, {self.n_joints})，但得到了 {joint_angles.shape}")

        # 将关节角度移动到指定设备
        joint_angles = joint_angles.to(self.device)

        # 使用 GPU 版本的 link_fk_batch 计算所有连杆的变换
        end_link_transforms = self.robot.link_fk_batch_gpu(
            cfgs=joint_angles,
            device=self.device,
            link=end_link,
            use_names=False
        )

        return end_link_transforms

    def compute_fk_for_links(self, joint_angles, links):
        """
        批量计算多个连杆的正向运动学。

        Args:
            joint_angles (torch.Tensor): 形状为 (batch_size, n_joints) 的关节角度张量。
            links (list of str): 要计算正向运动学的连杆名称列表。

        Returns:
            dict:  一个字典，其中键是连杆名称，值是形状为 (batch_size, 4, 4) 的变换矩阵张量。
        """
        batch_size = joint_angles.shape[0]
        if joint_angles.shape[1] != self.n_joints:
            raise ValueError(f"期望的关节角度形状为 (batch_size, {self.n_joints})，但得到了 {joint_angles.shape}")

        joint_angles = joint_angles.to(self.device)

        # 使用 GPU 版本的 link_fk_batch 计算所有连杆的变换
        transforms = self.robot.link_fk_batch_gpu(
            cfgs=joint_angles,
            device=self.device,
            links=links,
            use_names=True
        )

        return transforms

# 示例用法：

if __name__ == '__main__':
    # 1. 加载机器人模型
    urdf_path = 'ur5.urdf'  # 替换为你的 URDF 文件路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fk_solver = BatchForwardKinematics(urdf_path, device=device)
    print(f"使用设备：{device}")
    print(f"机器人有 {fk_solver.n_joints} 个关节")

    # 2. 为每个关节创建均匀采样点
    n_samples_per_joint = 20  # 每个关节1000个采样点
    batch_size = int(1e8)
    
    # 假设每个关节的范围是 [-π, π]，你可以根据实际机器人调整
    joint_ranges = []
    for i in range(fk_solver.n_joints):
        joint_samples = torch.linspace(-torch.pi, torch.pi, n_samples_per_joint)
        joint_ranges.append(joint_samples)
    
    total_combinations = n_samples_per_joint ** fk_solver.n_joints
    print(f"为每个关节生成 {n_samples_per_joint} 个均匀采样点...")
    print(f"总共需要处理 {total_combinations:.2e} 个关节配置组合")
    total_batches = int(np.ceil(total_combinations / batch_size))
    print(f"将分 {total_batches} 个批次处理，每批次 {batch_size:,} 个配置")
    
    # 估算总时间和内存使用
    memory_per_batch_gb = batch_size * fk_solver.n_joints * 4 * 2 / (1024**3)  # input + output
    print(f"每批次预估GPU内存使用: {memory_per_batch_gb:.1f} GB")
    
    # 让用户选择处理多少个批次
    max_demo_batches = min(total_batches, 10)  # 默认最多演示10个批次
    print(f"\n注意：总共 {total_batches} 个批次可能需要很长时间处理")
    user_input = input(f"请输入要处理的批次数量 (默认 {max_demo_batches}，输入 'all' 处理全部): ").strip()
    
    if user_input.lower() == 'all':
        batches_to_process = total_batches
    elif user_input.isdigit():
        batches_to_process = min(int(user_input), total_batches)
    else:
        batches_to_process = max_demo_batches
    
    print(f"将处理 {batches_to_process} 个批次...")

    def generate_joint_combinations_batch(batch_idx, batch_size):
        """生成指定批次的关节角度组合"""
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_combinations)
        actual_batch_size = end_idx - start_idx
        
        if actual_batch_size <= 0:
            return None
            
        # 计算多维索引
        joint_angles_batch = torch.zeros(actual_batch_size, fk_solver.n_joints)
        
        for i in range(actual_batch_size):
            global_idx = start_idx + i
            indices = []
            temp_idx = global_idx
            
            # 将线性索引转换为多维索引
            for j in range(fk_solver.n_joints):
                indices.append(temp_idx % n_samples_per_joint)
                temp_idx //= n_samples_per_joint
            
            # 转换为实际的关节角度
            for j in range(fk_solver.n_joints):
                joint_angles_batch[i, j] = joint_ranges[j][indices[j]]
        
        return joint_angles_batch

    # 3. 创建保存目录
    save_dir = "fk_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存到目录: {save_dir}")
    
    # 计算数据格式：DOF个关节值 + 7个位姿值(x,y,z,qx,qy,qz,qw)
    data_dim = fk_solver.n_joints + 7
    print(f"每个FK数据维度: {data_dim} (前{fk_solver.n_joints}个关节值 + 7个位姿值)")
    
    total_processed = 0
    
    print(f"\n开始分批次计算 FK...")
    
    if device == 'cuda':
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        overall_start.record(stream)
    else:
        import time
        overall_start_time = time.time()
    
    for batch_idx in range(batches_to_process):
        print(f"处理批次 {batch_idx + 1}/{batches_to_process}...", end='')
        
        # 生成当前批次的关节角度
        joint_angles_batch = generate_joint_combinations_batch(batch_idx, batch_size)
        if joint_angles_batch is None:
            break
            
        # 计算FK
        transforms = fk_solver.compute_fk(joint_angles_batch, end_link='tool0')
        
        # 提取位置信息 (x, y, z)
        positions = transforms[:, :3, 3]  # 形状: (batch_size, 3)
        
        # 提取旋转矩阵并转换为四元数
        rotations = transforms[:, :3, :3]  # 形状: (batch_size, 3, 3)
        quaternions = rotation_matrix_to_quaternion(rotations)  # 形状: (batch_size, 4) [x,y,z,w]
        
        # 组合数据：[关节角度, x, y, z, qx, qy, qz, qw]
        joint_angles_cpu = joint_angles_batch.cpu()
        positions_cpu = positions.cpu()
        quaternions_cpu = quaternions.cpu()
        
        # 连接所有数据
        batch_data = torch.cat([
            joint_angles_cpu,    # DOF 个关节值
            positions_cpu,       # 3 个位置值
            quaternions_cpu      # 4 个四元数值
        ], dim=1)  # 形状: (batch_size, DOF+7)
        
        # 保存当前批次数据
        batch_filename = f"fk_batch_{batch_idx:04d}.pt"
        batch_filepath = os.path.join(save_dir, batch_filename)
        torch.save(batch_data, batch_filepath)
        
        total_processed += joint_angles_batch.shape[0]
        print(f" 完成并保存到 {batch_filename} ({total_processed:,} 个配置)")
        
        # 清理GPU内存
        del joint_angles_batch, transforms, positions, rotations, quaternions
        del joint_angles_cpu, positions_cpu, quaternions_cpu, batch_data
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    if device == 'cuda':
        overall_end.record(stream)
        torch.cuda.synchronize()
        total_time_ms = overall_start.elapsed_time(overall_end)
        print(f"\n总计算时间: {total_time_ms:.3f} 毫秒")
        print(f"每个 FK 平均用时: {total_time_ms / total_processed:.6f} 毫秒")
    else:
        total_time_s = time.time() - overall_start_time
        print(f"\n总计算时间: {total_time_s * 1000:.3f} 毫秒")
        print(f"每个 FK 平均用时: {total_time_s * 1000 / total_processed:.6f} 毫秒")

    # 4. 完成处理
    if total_processed > 0:
        print(f"\n处理完成！总共计算了 {total_processed:,} 个 FK")
        print(f"所有批次数据已保存到目录: {save_dir}")
        print(f"数据格式: 每行包含 {data_dim} 个值")
        print(f"  - 前 {fk_solver.n_joints} 个值: 关节角度 (弧度)")
        print(f"  - 第 {fk_solver.n_joints+1}-{fk_solver.n_joints+3} 个值: 位置 x, y, z")
        print(f"  - 第 {fk_solver.n_joints+4}-{fk_solver.n_joints+7} 个值: 四元数 qx, qy, qz, qw")
        
        # 提供加载示例代码
        print(f"\n加载数据示例:")
        print(f"import torch")
        print(f"batch_data = torch.load('{save_dir}/fk_batch_0000.pt')")
        print(f"joint_angles = batch_data[:, :{fk_solver.n_joints}]  # 关节角度")
        print(f"positions = batch_data[:, {fk_solver.n_joints}:{fk_solver.n_joints+3}]  # 位置 xyz")
        print(f"quaternions = batch_data[:, {fk_solver.n_joints+3}:]  # 四元数 qx,qy,qz,qw")
    else:
        print("没有处理任何数据")

# 在文件末尾添加性能测试函数

def benchmark_batch_sizes():
    """测试不同batch_size的性能"""
    import time
    
    # 测试配置
    urdf_path = 'ur5.urdf'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        fk_solver = BatchForwardKinematics(urdf_path, device=device)
    except:
        print("无法加载URDF文件，使用模拟数据测试...")
        return
    
    # 不同的batch_size测试
    batch_sizes = [
        1000,      # 1K
        10000,     # 10K  
        100000,    # 100K
        500000,    # 500K
        1000000,   # 1M
        2000000,   # 2M
        5000000,   # 5M
        10000000,  # 10M (当前设置)
    ]
    
    results = []
    
    print("=" * 80)
    print("Batch Size 性能测试")
    print("=" * 80)
    print(f"设备: {device}")
    print(f"关节数: {fk_solver.n_joints}")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'内存(GB)':<10} {'时间(s)':<10} {'FPS':<12} {'每个FK(μs)':<12} {'状态':<10}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        try:
            # 估算内存使用
            memory_gb = batch_size * fk_solver.n_joints * 4 * 2 / (1024**3)
            
            # 检查内存限制
            if device == 'cuda':
                available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if memory_gb > available_memory * 0.8:  # 80% 阈值
                    print(f"{batch_size:<12,} {memory_gb:<10.2f} {'---':<10} {'---':<12} {'---':<12} {'跳过-内存':<10}")
                    continue
            
            # 生成测试数据
            joint_angles = torch.randn(batch_size, fk_solver.n_joints) * torch.pi
            
            # 预热
            if device == 'cuda':
                _ = fk_solver.compute_fk(joint_angles[:min(1000, batch_size)])
                torch.cuda.synchronize()
            
            # 计时测试
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream()
                
                start_event.record(stream)
                transforms = fk_solver.compute_fk(joint_angles)
                end_event.record(stream)
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            else:
                start_time = time.time()
                transforms = fk_solver.compute_fk(joint_angles)
                elapsed_time = time.time() - start_time
            
            # 计算性能指标
            fps = batch_size / elapsed_time
            time_per_fk = elapsed_time * 1e6 / batch_size  # 微秒
            
            results.append({
                'batch_size': batch_size,
                'memory_gb': memory_gb,
                'time_s': elapsed_time,
                'fps': fps,
                'time_per_fk_us': time_per_fk
            })
            
            print(f"{batch_size:<12,} {memory_gb:<10.2f} {elapsed_time:<10.3f} {fps:<12,.0f} {time_per_fk:<12.2f} {'成功':<10}")
            
            # 清理GPU内存
            del joint_angles, transforms
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"{batch_size:<12,} {memory_gb:<10.2f} {'---':<10} {'---':<12} {'---':<12} {'错误':<10}")
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    print("-" * 80)
    
    # 分析结果
    if results:
        print("\n性能分析:")
        print("=" * 50)
        
        # 找到最佳性能
        best_fps = max(results, key=lambda x: x['fps'])
        best_efficiency = min(results, key=lambda x: x['time_per_fk_us'])
        
        print(f"最高FPS: {best_fps['batch_size']:,} (FPS: {best_fps['fps']:,.0f})")
        print(f"最高效率: {best_efficiency['batch_size']:,} (每个FK: {best_efficiency['time_per_fk_us']:.2f}μs)")
        
        # 效率分析
        print(f"\nBatch Size 效率分析:")
        baseline = results[0]
        for result in results:
            speedup = result['fps'] / baseline['fps']
            efficiency = baseline['time_per_fk_us'] / result['time_per_fk_us']
            print(f"  {result['batch_size']:>8,}: 相对FPS提升 {speedup:.2f}x, 效率提升 {efficiency:.2f}x")
        
        # 推荐设置
        print(f"\n推荐设置:")
        
        # 根据内存和性能平衡推荐
        memory_limit = 4.0 if device == 'cuda' else 8.0  # GPU/CPU内存限制
        suitable_results = [r for r in results if r['memory_gb'] <= memory_limit]
        
        if suitable_results:
            recommended = max(suitable_results, key=lambda x: x['fps'])
            print(f"  推荐batch_size: {recommended['batch_size']:,}")
            print(f"  内存使用: {recommended['memory_gb']:.2f} GB")
            print(f"  预期性能: {recommended['fps']:,.0f} FPS")
        else:
            print(f"  当前内存限制({memory_limit}GB)下无法运行任何测试配置")
    
    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_batch_sizes()
    else:
        # 原有的主程序 - 执行现有的FK计算逻辑
        pass

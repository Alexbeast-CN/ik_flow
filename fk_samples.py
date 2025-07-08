import torch
import numpy as np
from urdf import URDF  # 假设 urdf.py 文件在同一目录下
import os
import time


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
    mask2 = (
        (~mask1)
        & (R_flat[:, 0, 0] > R_flat[:, 1, 1])
        & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    )
    if mask2.any():
        s = (
            torch.sqrt(
                1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]
            )
            * 2
        )  # s = 4 * qx
        q[mask2, 3] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s  # qw
        q[mask2, 0] = 0.25 * s  # qx
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s  # qy
        q[mask2, 2] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s  # qz

    # 情况3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask3.any():
        s = (
            torch.sqrt(
                1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]
            )
            * 2
        )  # s = 4 * qy
        q[mask3, 3] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s  # qw
        q[mask3, 0] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s  # qx
        q[mask3, 1] = 0.25 * s  # qy
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s  # qz

    # 情况4: 其他情况
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = (
            torch.sqrt(
                1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]
            )
            * 2
        )  # s = 4 * qz
        q[mask4, 3] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s  # qw
        q[mask4, 0] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s  # qx
        q[mask4, 1] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s  # qy
        q[mask4, 2] = 0.25 * s  # qz

    return q.view(*batch_shape, 4)


class BatchForwardKinematics:
    def __init__(self, urdf_path, device="cpu"):
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
            transforms[joint.name] = torch.tensor(
                joint.origin, dtype=torch.float32, device=self.device
            )
        return transforms

    def compute_fk(self, joint_angles, end_link="tool0"):
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
            raise ValueError(
                f"期望的关节角度形状为 (batch_size, {self.n_joints})，但得到了 {joint_angles.shape}"
            )

        # 将关节角度移动到指定设备
        joint_angles = joint_angles.to(self.device)

        # 使用 GPU 版本的 link_fk_batch 计算所有连杆的变换
        end_link_transforms = self.robot.link_fk_batch_gpu(
            cfgs=joint_angles, device=self.device, link=end_link, use_names=False
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
            raise ValueError(
                f"期望的关节角度形状为 (batch_size, {self.n_joints})，但得到了 {joint_angles.shape}"
            )

        joint_angles = joint_angles.to(self.device)

        # 使用 GPU 版本的 link_fk_batch 计算所有连杆的变换
        transforms = self.robot.link_fk_batch_gpu(
            cfgs=joint_angles, device=self.device, links=links, use_names=True
        )

        return transforms


def print_robot_info(fk_solver):
    """打印机器人信息"""
    device = fk_solver.device
    print(f"使用设备：{device}")
    print(f"机器人有 {fk_solver.n_joints} 个关节")

    # 获取关节限制
    joint_limits = fk_solver.robot.joint_limits
    print(f"关节限制:")
    for i, (joint_name, limits) in enumerate(
        zip(fk_solver.actuated_joint_names, joint_limits)
    ):
        lower, upper = limits
        if lower == -np.inf:
            lower_str = "-∞"
        else:
            lower_str = f"{lower:.3f}"
        if upper == np.inf:
            upper_str = "∞"
        else:
            upper_str = f"{upper:.3f}"
        print(f"  {joint_name}: [{lower_str}, {upper_str}]")


def calculate_sampling_parameters(
    fk_solver,
    n_samples_per_joint=20,
    max_batch_size=int(1e6),
    use_regular_batching=True,
):
    """计算采样参数"""
    joint_limits = fk_solver.robot.joint_limits

    # 计算总组合数
    total_combinations = n_samples_per_joint**fk_solver.n_joints
    print(f"为每个关节生成 {n_samples_per_joint} 个均匀采样点...")
    print(f"总共需要处理 {total_combinations:.2e} 个关节配置组合")

    # 计算实际批次大小和批次数量
    if total_combinations <= max_batch_size:
        actual_batch_size = total_combinations
        total_batches = 1
    else:
        if use_regular_batching:
            # 原算法：找到一个合适的批次大小，使其能整除 n_samples_per_joint
            # 这样可以更容易地在 GPU 上生成网格，但批次大小可能比 max_batch_size 小很多
            factor = n_samples_per_joint
            while factor**fk_solver.n_joints > max_batch_size:
                factor = max(factor // 2, 1)
            actual_batch_size = factor**fk_solver.n_joints
            print(
                f"使用规律性批次模式: factor={factor}, actual_batch_size={actual_batch_size}"
            )
        else:
            # 新算法：直接使用接近 max_batch_size 的值
            actual_batch_size = min(max_batch_size, total_combinations)
            print(f"使用最大批次模式: actual_batch_size={actual_batch_size}")

        total_batches = int(np.ceil(total_combinations / actual_batch_size))

    print(f"实际批次大小: {actual_batch_size:,}")
    print(f"将分 {total_batches} 个批次处理")

    # 估算内存使用
    memory_per_batch_gb = (
        actual_batch_size * fk_solver.n_joints * 4 * 2 / (1024**3)
    )  # input + output
    print(f"每批次预估GPU内存使用: {memory_per_batch_gb:.1f} GB")

    return {
        "n_samples_per_joint": n_samples_per_joint,
        "total_combinations": total_combinations,
        "actual_batch_size": actual_batch_size,
        "total_batches": total_batches,
        "joint_limits": joint_limits,
        "use_regular_batching": use_regular_batching,
    }


def generate_joint_combinations_gpu(fk_solver, batch_idx, batch_size, sampling_params):
    """
    在 GPU 上高效生成关节角度组合

    Args:
        fk_solver: FK求解器
        batch_idx: 批次索引
        batch_size: 当前批次的大小
        sampling_params: 采样参数字典

    Returns:
        torch.Tensor: 形状为 (batch_size, n_joints) 的关节角度张量
    """
    n_samples_per_joint = sampling_params["n_samples_per_joint"]
    total_combinations = sampling_params["total_combinations"]
    joint_limits = sampling_params["joint_limits"]
    use_regular_batching = sampling_params.get("use_regular_batching", True)
    device = fk_solver.device

    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, total_combinations)
    current_batch_size = end_idx - start_idx

    if current_batch_size <= 0:
        return None

    # 为每个关节在其限制范围内创建均匀采样点
    joint_samples = []
    for i in range(fk_solver.n_joints):
        lower, upper = joint_limits[i]
        # 处理无限限制的情况
        if lower == -np.inf:
            lower = -torch.pi
        if upper == np.inf:
            upper = torch.pi

        # 在 GPU 上创建采样点
        samples = torch.linspace(lower, upper, n_samples_per_joint, device=device)
        joint_samples.append(samples)

    # 在 GPU 上生成索引并转换为关节角度
    joint_angles_batch = torch.zeros(
        current_batch_size, fk_solver.n_joints, device=device
    )

    # 生成批次内的所有线性索引
    indices = torch.arange(start_idx, end_idx, device=device)

    # 将线性索引转换为多维索引并查找对应的关节角度
    # 这个方法对规律性和非规律性批次都适用
    for j in range(fk_solver.n_joints):
        joint_indices = indices % n_samples_per_joint
        joint_angles_batch[:, j] = joint_samples[j][joint_indices]
        indices = indices // n_samples_per_joint

    return joint_angles_batch


def process_fk_batch(fk_solver, joint_angles_batch, end_link="tool0"):
    """
    处理一个批次的FK计算

    Args:
        fk_solver: FK求解器
        joint_angles_batch: 关节角度批次
        end_link: 末端连杆名称

    Returns:
        torch.Tensor: 处理后的数据，形状为 (batch_size, DOF+7)
    """
    # 计算FK
    transforms = fk_solver.compute_fk(joint_angles_batch, end_link=end_link)

    # 提取位置信息 (x, y, z)
    positions = transforms[:, :3, 3]  # 形状: (batch_size, 3)

    # 提取旋转矩阵并转换为四元数
    rotations = transforms[:, :3, :3]  # 形状: (batch_size, 3, 3)
    quaternions = rotation_matrix_to_quaternion(
        rotations
    )  # 形状: (batch_size, 4) [x,y,z,w]

    # 组合数据：[关节角度, x, y, z, qx, qy, qz, qw]
    joint_angles_cpu = joint_angles_batch.cpu()
    positions_cpu = positions.cpu()
    quaternions_cpu = quaternions.cpu()

    # 连接所有数据
    batch_data = torch.cat(
        [
            joint_angles_cpu,  # DOF 个关节值
            positions_cpu,  # 3 个位置值
            quaternions_cpu,  # 4 个四元数值
        ],
        dim=1,
    )  # 形状: (batch_size, DOF+7)

    return batch_data


def generate_fk_dataset(fk_solver, sampling_params, end_link="tool0"):
    """
    生成FK数据集

    Args:
        fk_solver: FK求解器
        sampling_params: 采样参数
        end_link: 末端连杆名称

    Returns:
        torch.Tensor: 完整的FK数据集
    """
    device = fk_solver.device
    actual_batch_size = sampling_params["actual_batch_size"]
    total_batches = sampling_params["total_batches"]

    print(f"将处理所有 {total_batches} 个批次...")

    # 用于累积所有批次数据的列表
    all_batch_data = []
    total_processed = 0

    print(f"\n开始分批次计算 FK...")

    # 计时开始
    if device == "cuda":
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        overall_start.record(stream)
    else:
        overall_start_time = time.time()

    for batch_idx in range(total_batches):
        print(f"处理批次 {batch_idx + 1}/{total_batches}...", end="")

        # 在 GPU 上生成当前批次的关节角度
        joint_angles_batch = generate_joint_combinations_gpu(
            fk_solver, batch_idx, actual_batch_size, sampling_params
        )

        if joint_angles_batch is None:
            break

        # 处理FK计算
        batch_data = process_fk_batch(fk_solver, joint_angles_batch, end_link)

        # 累积数据
        all_batch_data.append(batch_data)
        total_processed += joint_angles_batch.shape[0]
        print(f" 完成 ({total_processed:,} 个配置)")

        # 清理GPU内存
        del joint_angles_batch, batch_data
        if device == "cuda":
            torch.cuda.empty_cache()

    # 计时结束
    if device == "cuda":
        overall_end.record(stream)
        torch.cuda.synchronize()
        total_time_ms = overall_start.elapsed_time(overall_end)
        print(f"\n总计算时间: {total_time_ms:.3f} 毫秒")
        if total_processed > 0:
            print(f"每个 FK 平均用时: {total_time_ms / total_processed:.6f} 毫秒")
    else:
        total_time_s = time.time() - overall_start_time
        print(f"\n总计算时间: {total_time_s * 1000:.3f} 毫秒")
        if total_processed > 0:
            print(f"每个 FK 平均用时: {total_time_s * 1000 / total_processed:.6f} 毫秒")

    # 合并所有批次数据
    if all_batch_data:
        final_data = torch.cat(all_batch_data, dim=0)
        print(f"数据合并完成，最终数据形状: {final_data.shape}")
        return final_data, total_processed
    else:
        return None, 0


def save_fk_dataset(data, fk_solver, save_dir="fk_results", filename="fk_dataset.pt"):
    """
    保存FK数据集

    Args:
        data: FK数据集
        fk_solver: FK求解器
        save_dir: 保存目录
        filename: 文件名
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    torch.save(data, filepath)

    data_dim = fk_solver.n_joints + 7
    print(f"数据已保存到: {filepath}")
    print(f"数据形状: {data.shape}")
    print(f"数据格式: 每行包含 {data_dim} 个值")
    print(f"  - 前 {fk_solver.n_joints} 个值: 关节角度 (弧度)")
    print(f"  - 第 {fk_solver.n_joints+1}-{fk_solver.n_joints+3} 个值: 位置 x, y, z")
    print(
        f"  - 第 {fk_solver.n_joints+4}-{fk_solver.n_joints+7} 个值: 四元数 qx, qy, qz, qw"
    )

    # 提供加载示例代码
    print(f"\n加载数据示例:")
    print(f"import torch")
    print(f"data = torch.load('{filepath}')")
    print(f"joint_angles = data[:, :{fk_solver.n_joints}]  # 关节角度")
    print(
        f"positions = data[:, {fk_solver.n_joints}:{fk_solver.n_joints+3}]  # 位置 xyz"
    )
    print(f"quaternions = data[:, {fk_solver.n_joints+3}:]  # 四元数 qx,qy,qz,qw")


def main():
    """主函数"""
    # 1. 配置参数
    urdf_path = "ur5.urdf"  # 替换为你的 URDF 文件路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples_per_joint = 20  # 每个关节100个采样点
    max_batch_size = int(1e6)  # 最大批次大小，避免内存过大
    end_link = "tool0"
    save_dir = "fk_results"

    # 新增：选择批次模式
    # use_regular_batching=True: 保持规律性，批次大小为 factor^n_joints（可能远小于max_batch_size）
    # use_regular_batching=False: 使用接近 max_batch_size 的批次大小
    use_regular_batching = False  # 设置为 False 来使用更大的批次

    # 2. 初始化FK求解器
    fk_solver = BatchForwardKinematics(urdf_path, device=device)

    # 3. 打印机器人信息
    print_robot_info(fk_solver)

    # 4. 计算采样参数
    sampling_params = calculate_sampling_parameters(
        fk_solver, n_samples_per_joint, max_batch_size, use_regular_batching
    )

    total_batches = sampling_params["total_batches"]
    print(f"\n注意：总共 {total_batches} 个批次可能需要很长时间处理")

    # 5. 生成FK数据集（处理所有批次）
    data, total_processed = generate_fk_dataset(
        fk_solver, sampling_params, end_link=end_link
    )

    # 6. 保存数据集
    if data is not None and total_processed > 0:
        save_fk_dataset(data, fk_solver, save_dir)
        print(f"\n处理完成！总共计算了 {total_processed:,} 个 FK")
    else:
        print("没有处理任何数据")


# 示例用法：

if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from planner_robot import PlannerRobot
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class RealNVP(nn.Module):
    """
    Real-valued Non-Volume Preserving (Real NVP) 模型
    用于逆运动学的 Normalizing Flow 实现
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, num_layers: int = 8):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # 创建耦合层 (Coupling layers)
        self.coupling_layers = nn.ModuleList()
        for i in range(num_layers):
            self.coupling_layers.append(
                CouplingLayer(input_dim, hidden_dim, mask_type=(i % 2))
            )
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            reverse: 是否反向计算
            
        Returns:
            z: 变换后的张量
            log_det_jacobian: 雅可比行列式的对数
        """
        if reverse:
            return self._inverse(x)
        else:
            return self._forward(x)
    
    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """正向变换: x -> z"""
        z = x
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        
        for coupling_layer in self.coupling_layers:
            z, log_det = coupling_layer(z)
            log_det_jacobian += log_det
            
        return z, log_det_jacobian
    
    def _inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆向变换: z -> x"""
        x = z
        log_det_jacobian = torch.zeros(z.shape[0], device=z.device)
        
        for coupling_layer in reversed(self.coupling_layers):
            x, log_det = coupling_layer(x, reverse=True)
            log_det_jacobian += log_det
            
        return x, log_det_jacobian
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """从基础分布采样并转换到数据空间"""
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, reverse=True)
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """计算数据点的对数概率密度"""
        z, log_det_jacobian = self.forward(x)
        # 基础分布是标准正态分布
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob_z + log_det_jacobian


class CouplingLayer(nn.Module):
    """耦合层实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, mask_type: int):
        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.mask_type = mask_type
        
        # 创建掩码 (Create mask)
        if mask_type == 0:
            self.mask = torch.tensor([1, 0], dtype=torch.float32)
        else:
            self.mask = torch.tensor([0, 1], dtype=torch.float32)
        
        # 尺度和平移网络 (Scale and translation networks)
        self.scale_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 限制尺度参数
        )
        
        self.translation_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """耦合层变换"""
        mask = self.mask.to(x.device)
        masked_x = x * mask
        
        # 提取被掩码的维度
        if self.mask_type == 0:
            x_masked = x[:, 0:1]  # [batch_size, 1]
        else:
            x_masked = x[:, 1:2]  # [batch_size, 1]
        
        # 计算尺度和平移参数
        scale = self.scale_net(x_masked)
        translation = self.translation_net(x_masked)
        
        if reverse:
            # 逆变换
            if self.mask_type == 0:
                x_new = x.clone()
                x_new[:, 1:2] = (x[:, 1:2] - translation) / torch.exp(scale)
            else:
                x_new = x.clone()
                x_new[:, 0:1] = (x[:, 0:1] - translation) / torch.exp(scale)
            log_det = -scale.sum(dim=1)
        else:
            # 正向变换
            if self.mask_type == 0:
                x_new = x.clone()
                x_new[:, 1:2] = x[:, 1:2] * torch.exp(scale) + translation
            else:
                x_new = x.clone()
                x_new[:, 0:1] = x[:, 0:1] * torch.exp(scale) + translation
            log_det = scale.sum(dim=1)
        
        return x_new, log_det


class InverseKinematicsFlow:
    """
    基于 Normalizing Flow 的逆运动学求解器
    """
    
    def __init__(self, robot: PlannerRobot, device: str = 'cpu'):
        self.robot = robot
        self.device = device
        
        # 创建 Flow 模型
        self.flow_model = RealNVP(input_dim=2, hidden_dim=256, num_layers=8).to(device)
        
        # 数据标准化参数
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.theta1_mean = 0.0
        self.theta1_std = 1.0
        self.theta2_mean = 0.0
        self.theta2_std = 1.0
        
    def generate_training_data(self, num_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据：从关节角度到末端执行器位置的映射
        
        Args:
            num_samples: 样本数量
            
        Returns:
            positions: 末端执行器位置 [num_samples, 2]
            joint_angles: 关节角度 [num_samples, 2]
        """
        print(f"正在生成 {num_samples} 个训练样本...")
        
        # 均匀采样关节角度
        theta1_samples = np.random.uniform(-np.pi, np.pi, num_samples)
        theta2_samples = np.random.uniform(-np.pi, np.pi, num_samples)
        
        positions = []
        joint_angles = []
        
        for i, (theta1, theta2) in enumerate(zip(theta1_samples, theta2_samples)):
            # 计算正向运动学
            (x0, y0), (x1, y1), (x2, y2) = self.robot.forward_kinematics(theta1, theta2)
            
            # 检查是否在工作空间内
            distance = np.sqrt(x2**2 + y2**2)
            if abs(self.robot.L1 - self.robot.L2) <= distance <= (self.robot.L1 + self.robot.L2):
                positions.append([x2, y2])
                joint_angles.append([theta1, theta2])
            
            if (i + 1) % 10000 == 0:
                print(f"已处理 {i + 1} 个样本...")
        
        positions = np.array(positions)
        joint_angles = np.array(joint_angles)
        
        print(f"生成了 {len(positions)} 个有效样本")
        return positions, joint_angles
    
    def normalize_data(self, positions: np.ndarray, joint_angles: np.ndarray):
        """标准化数据"""
        # 计算标准化参数
        self.x_mean, self.x_std = positions[:, 0].mean(), positions[:, 0].std()
        self.y_mean, self.y_std = positions[:, 1].mean(), positions[:, 1].std()
        self.theta1_mean, self.theta1_std = joint_angles[:, 0].mean(), joint_angles[:, 0].std()
        self.theta2_mean, self.theta2_std = joint_angles[:, 1].mean(), joint_angles[:, 1].std()
        
        # 标准化
        positions_norm = np.column_stack([
            (positions[:, 0] - self.x_mean) / self.x_std,
            (positions[:, 1] - self.y_mean) / self.y_std
        ])
        
        joint_angles_norm = np.column_stack([
            (joint_angles[:, 0] - self.theta1_mean) / self.theta1_std,
            (joint_angles[:, 1] - self.theta2_mean) / self.theta2_std
        ])
        
        return positions_norm, joint_angles_norm
    
    def train(self, num_samples: int = 50000, epochs: int = 100, batch_size: int = 512, lr: float = 1e-3):
        """
        训练 Flow 模型
        
        Args:
            num_samples: 训练样本数量
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
        """
        print("开始训练逆运动学 Flow 模型...")
        
        # 生成训练数据
        positions, joint_angles = self.generate_training_data(num_samples)
        
        # 标准化数据
        positions_norm, joint_angles_norm = self.normalize_data(positions, joint_angles)
        
        # 创建数据加载器
        # 注意：这里我们要学习从位置到关节角度的映射
        # 所以输入是位置，输出是关节角度
        dataset = TensorDataset(
            torch.FloatTensor(positions_norm),
            torch.FloatTensor(joint_angles_norm)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr)
        
        # 训练循环
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_pos, batch_angles in dataloader:
                batch_pos = batch_pos.to(self.device)
                batch_angles = batch_angles.to(self.device)
                
                optimizer.zero_grad()
                
                # 计算负对数似然损失
                # 我们要学习 p(joint_angles | position)
                # 这里我们使用条件 Flow，将位置作为条件输入
                # 简化起见，我们直接使用关节角度的分布
                z, log_det_jacobian = self.flow_model(batch_angles)
                
                # 负对数似然
                log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * 2 * np.log(2 * np.pi)
                log_prob = log_prob_z + log_det_jacobian
                loss = -log_prob.mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")
        
        print("训练完成!")
        return losses
    
    def inverse_kinematics(self, target_x: float, target_y: float, num_samples: int = 100) -> Tuple[float, float]:
        """
        使用训练好的模型计算逆运动学
        
        Args:
            target_x: 目标 x 坐标
            target_y: 目标 y 坐标
            num_samples: 采样数量用于寻找最佳解
            
        Returns:
            theta1, theta2: 最佳关节角度
        """
        self.flow_model.eval()
        
        # 标准化目标位置
        target_x_norm = (target_x - self.x_mean) / self.x_std
        target_y_norm = (target_y - self.y_mean) / self.y_std
        
        with torch.no_grad():
            # 从 Flow 模型采样
            samples = self.flow_model.sample(num_samples, device=self.device)
            samples_np = samples.cpu().numpy()
            
            # 反标准化关节角度
            theta1_samples = samples_np[:, 0] * self.theta1_std + self.theta1_mean
            theta2_samples = samples_np[:, 1] * self.theta2_std + self.theta2_mean
            
            # 找到最接近目标位置的解
            best_error = float('inf')
            best_theta1, best_theta2 = 0.0, 0.0
            
            for theta1, theta2 in zip(theta1_samples, theta2_samples):
                # 计算正向运动学
                (x0, y0), (x1, y1), (x2, y2) = self.robot.forward_kinematics(theta1, theta2)
                
                # 计算误差
                error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
                
                if error < best_error:
                    best_error = error
                    best_theta1, best_theta2 = theta1, theta2
        
        return best_theta1, best_theta2
    
    def visualize_training_data(self, num_samples: int = 1000):
        """可视化训练数据分布"""
        positions, joint_angles = self.generate_training_data(num_samples)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 工作空间分布
        axes[0, 0].scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=1)
        axes[0, 0].set_xlabel('X 坐标')
        axes[0, 0].set_ylabel('Y 坐标')
        axes[0, 0].set_title('末端执行器工作空间')
        axes[0, 0].set_aspect('equal')
        
        # 关节角度分布
        axes[0, 1].scatter(joint_angles[:, 0], joint_angles[:, 1], alpha=0.6, s=1)
        axes[0, 1].set_xlabel('θ1 (rad)')
        axes[0, 1].set_ylabel('θ2 (rad)')
        axes[0, 1].set_title('关节角度分布')
        
        # θ1 分布
        axes[1, 0].hist(joint_angles[:, 0], bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('θ1 (rad)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('θ1 分布')
        
        # θ2 分布
        axes[1, 1].hist(joint_angles[:, 1], bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('θ2 (rad)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('θ2 分布')
        
        plt.tight_layout()
        plt.show()


def test_inverse_kinematics():
    """测试逆运动学求解器"""
    print("创建机器人和逆运动学求解器...")
    
    # 创建机器人
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建逆运动学求解器
    ik_solver = InverseKinematicsFlow(robot, device=device)
    
    # 可视化训练数据
    print("\n可视化训练数据分布...")
    ik_solver.visualize_training_data(5000)
    
    # 训练模型
    print("\n开始训练...")
    losses = ik_solver.train(num_samples=30000, epochs=50, batch_size=512, lr=1e-3)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.grid(True)
    plt.show()
    
    # 测试逆运动学
    print("\n测试逆运动学求解...")
    test_targets = [
        (1.0, 1.0),
        (0.5, 1.5),
        (-1.0, 0.5),
        (0.0, 1.8),
        (1.5, -0.5)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (target_x, target_y) in enumerate(test_targets):
        if i >= len(axes):
            break
            
        print(f"\n目标位置: ({target_x:.2f}, {target_y:.2f})")
        
        # 使用 Flow 模型求解
        theta1_pred, theta2_pred = ik_solver.inverse_kinematics(target_x, target_y, num_samples=1000)
        
        # 验证解的准确性
        (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1_pred, theta2_pred)
        error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
        
        print(f"预测关节角度: θ1={np.degrees(theta1_pred):.1f}°, θ2={np.degrees(theta2_pred):.1f}°")
        print(f"实际末端位置: ({x2:.3f}, {y2:.3f})")
        print(f"位置误差: {error:.4f}")
        
        # 可视化结果
        robot.visualize_static(theta1_pred, theta2_pred, ax=axes[i])
        axes[i].plot(target_x, target_y, 'r*', markersize=15, label=f'目标位置')
        axes[i].plot(x2, y2, 'g*', markersize=12, label=f'实际位置')
        axes[i].set_title(f'目标: ({target_x:.1f}, {target_y:.1f})\n误差: {error:.4f}')
        axes[i].legend()
    
    # 隐藏多余的子图
    for i in range(len(test_targets), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_inverse_kinematics() 
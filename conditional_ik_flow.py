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


class ConditionalCouplingLayer(nn.Module):
    """条件耦合层 - 以末端执行器位置为条件"""
    
    def __init__(self, input_dim: int = 2, condition_dim: int = 2, hidden_dim: int = 256, mask_type: int = 0):
        super(ConditionalCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.mask_type = mask_type
        
        # 创建掩码
        if mask_type == 0:
            self.mask = torch.tensor([1, 0], dtype=torch.float32)
        else:
            self.mask = torch.tensor([0, 1], dtype=torch.float32)
        
        # 条件网络的输入维度 = 被掩码的关节角度维度 + 位置条件维度
        condition_input_dim = 1 + condition_dim
        
        # 尺度网络 (条件输入 -> 尺度参数)
        self.scale_net = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 限制尺度参数范围
        )
        
        # 平移网络 (条件输入 -> 平移参数)
        self.translation_net = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        条件耦合层变换
        
        Args:
            x: 关节角度 [batch_size, 2]
            condition: 末端执行器位置 [batch_size, 2]
            reverse: 是否反向变换
        """
        # 提取被掩码的维度
        if self.mask_type == 0:
            x_masked = x[:, 0:1]  # 第一个关节角度
            x_transform = x[:, 1:2]  # 第二个关节角度
        else:
            x_masked = x[:, 1:2]  # 第二个关节角度
            x_transform = x[:, 0:1]  # 第一个关节角度
        
        # 构建条件输入：被掩码的关节角度 + 位置条件
        condition_input = torch.cat([x_masked, condition], dim=1)
        
        # 计算尺度和平移参数
        scale = self.scale_net(condition_input) * 2.0  # 扩大尺度范围
        translation = self.translation_net(condition_input)
        
        if reverse:
            # 逆变换: y = (x - t) / exp(s)
            x_new = x.clone()
            if self.mask_type == 0:
                x_new[:, 1:2] = (x_transform - translation) / torch.exp(scale)
            else:
                x_new[:, 0:1] = (x_transform - translation) / torch.exp(scale)
            log_det = -scale.sum(dim=1)
        else:
            # 正向变换: y = x * exp(s) + t
            x_new = x.clone()
            if self.mask_type == 0:
                x_new[:, 1:2] = x_transform * torch.exp(scale) + translation
            else:
                x_new[:, 0:1] = x_transform * torch.exp(scale) + translation
            log_det = scale.sum(dim=1)
        
        return x_new, log_det


class ConditionalRealNVP(nn.Module):
    """条件 Real NVP 模型用于逆运动学"""
    
    def __init__(self, input_dim: int = 2, condition_dim: int = 2, hidden_dim: int = 256, num_layers: int = 10):
        super(ConditionalRealNVP, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        
        # 创建条件耦合层
        self.coupling_layers = nn.ModuleList()
        for i in range(num_layers):
            self.coupling_layers.append(
                ConditionalCouplingLayer(input_dim, condition_dim, hidden_dim, mask_type=(i % 2))
            )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        条件正向/反向变换
        
        Args:
            x: 关节角度 [batch_size, 2]
            condition: 末端执行器位置 [batch_size, 2] 
            reverse: 是否反向变换
        """
        if reverse:
            return self._inverse(x, condition)
        else:
            return self._forward(x, condition)
    
    def _forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """正向变换: 关节角度 -> 隐空间"""
        z = x
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        
        for coupling_layer in self.coupling_layers:
            z, log_det = coupling_layer(z, condition)
            log_det_jacobian += log_det
        
        return z, log_det_jacobian
    
    def _inverse(self, z: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """逆向变换: 隐空间 -> 关节角度"""
        x = z
        log_det_jacobian = torch.zeros(z.shape[0], device=z.device)
        
        for coupling_layer in reversed(self.coupling_layers):
            x, log_det = coupling_layer(x, condition, reverse=True)
            log_det_jacobian += log_det
        
        return x, log_det_jacobian
    
    def sample_given_condition(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """给定条件采样关节角度"""
        batch_size = condition.shape[0]
        if num_samples > 1:
            # 扩展条件以匹配采样数
            condition = condition.repeat(num_samples, 1)
            total_samples = batch_size * num_samples
        else:
            total_samples = batch_size
        
        # 从标准正态分布采样
        z = torch.randn(total_samples, self.input_dim, device=condition.device)
        
        # 条件逆变换
        x, _ = self.forward(z, condition, reverse=True)
        
        if num_samples > 1:
            # 重新整形为 [batch_size, num_samples, input_dim]
            x = x.reshape(batch_size, num_samples, self.input_dim)
        
        return x
    
    def log_prob_given_condition(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """计算给定条件下的对数概率密度"""
        z, log_det_jacobian = self.forward(x, condition)
        # 基础分布的对数概率
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob_z + log_det_jacobian


class ConditionalInverseKinematicsFlow:
    """改进的条件逆运动学求解器"""
    
    def __init__(self, robot: PlannerRobot, device: str = 'cpu'):
        self.robot = robot
        self.device = device
        
        # 创建条件 Flow 模型
        self.flow_model = ConditionalRealNVP(
            input_dim=2, 
            condition_dim=2, 
            hidden_dim=512, 
            num_layers=12
        ).to(device)
        
        # 数据标准化参数
        self.x_mean = 0.0
        self.x_std = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0
        self.theta1_mean = 0.0
        self.theta1_std = 1.0
        self.theta2_mean = 0.0
        self.theta2_std = 1.0
    
    def generate_training_data(self, num_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        print(f"正在生成 {num_samples} 个训练样本...")
        
        # 使用更智能的采样策略
        positions = []
        joint_angles = []
        
        # 方法1: 在工作空间内均匀采样位置，然后计算对应的关节角度
        for _ in range(num_samples // 2):
            # 在圆环工作空间内采样
            r = np.random.uniform(abs(self.robot.L1 - self.robot.L2), self.robot.L1 + self.robot.L2)
            theta = np.random.uniform(0, 2 * np.pi)
            x, y = r * np.cos(theta), r * np.sin(theta)
            
            # 计算解析逆运动学解
            solutions = self._analytical_ik(x, y)
            for sol in solutions:
                if sol is not None:
                    theta1, theta2 = sol
                    positions.append([x, y])
                    joint_angles.append([theta1, theta2])
        
        # 方法2: 随机采样关节角度
        theta1_samples = np.random.uniform(-np.pi, np.pi, num_samples // 2)
        theta2_samples = np.random.uniform(-np.pi, np.pi, num_samples // 2)
        
        for theta1, theta2 in zip(theta1_samples, theta2_samples):
            (x0, y0), (x1, y1), (x2, y2) = self.robot.forward_kinematics(theta1, theta2)
            positions.append([x2, y2])
            joint_angles.append([theta1, theta2])
        
        positions = np.array(positions)
        joint_angles = np.array(joint_angles)
        
        print(f"生成了 {len(positions)} 个有效样本")
        return positions, joint_angles
    
    def _analytical_ik(self, x: float, y: float) -> List[Tuple[float, float]]:
        """解析逆运动学解"""
        L1, L2 = self.robot.L1, self.robot.L2
        
        # 计算到目标点的距离
        d = np.sqrt(x**2 + y**2)
        
        # 检查是否在工作空间内
        if d > L1 + L2 or d < abs(L1 - L2):
            return [None, None]
        
        # 使用余弦定理计算 theta2
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)  # 数值稳定性
        
        # 两个可能的 theta2 解
        theta2_1 = np.arccos(cos_theta2)
        theta2_2 = -np.arccos(cos_theta2)
        
        solutions = []
        for theta2 in [theta2_1, theta2_2]:
            # 计算对应的 theta1
            k1 = L1 + L2 * np.cos(theta2)
            k2 = L2 * np.sin(theta2)
            
            theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
            solutions.append((theta1, theta2))
        
        return solutions
    
    def normalize_data(self, positions: np.ndarray, joint_angles: np.ndarray):
        """数据标准化"""
        # 计算标准化参数
        self.x_mean = positions[:, 0].mean()
        self.x_std = positions[:, 0].std()
        self.y_mean = positions[:, 1].mean()
        self.y_std = positions[:, 1].std()
        self.theta1_mean = joint_angles[:, 0].mean()
        self.theta1_std = joint_angles[:, 0].std()
        self.theta2_mean = joint_angles[:, 1].mean()
        self.theta2_std = joint_angles[:, 1].std()
        
        # 确保标准差不为零
        if self.x_std == 0:
            self.x_std = 1.0
        if self.y_std == 0:
            self.y_std = 1.0
        if self.theta1_std == 0:
            self.theta1_std = 1.0
        if self.theta2_std == 0:
            self.theta2_std = 1.0
        
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
    
    def train(self, num_samples: int = 100000, epochs: int = 200, batch_size: int = 1024, lr: float = 1e-4):
        """训练条件 Flow 模型"""
        print("开始训练条件逆运动学 Flow 模型...")
        
        # 生成训练数据
        positions, joint_angles = self.generate_training_data(num_samples)
        
        # 标准化数据
        positions_norm, joint_angles_norm = self.normalize_data(positions, joint_angles)
        
        # 创建数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(joint_angles_norm),  # 关节角度作为输入
            torch.FloatTensor(positions_norm)      # 位置作为条件
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和学习率调度器
        optimizer = optim.AdamW(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练循环
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.flow_model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_angles, batch_positions in dataloader:
                batch_angles = batch_angles.to(self.device)
                batch_positions = batch_positions.to(self.device)
                
                optimizer.zero_grad()
                
                # 计算条件对数似然
                log_prob = self.flow_model.log_prob_given_condition(batch_angles, batch_positions)
                loss = -log_prob.mean()
                
                # 添加正则化项
                l2_reg = 0.0
                for param in self.flow_model.parameters():
                    l2_reg += torch.norm(param, p=2)
                loss += 1e-6 * l2_reg
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.flow_model.state_dict(), 'best_ik_flow_model.pth')
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}, 学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        print("训练完成!")
        return losses
    
    def load_model(self, model_path: str = 'best_ik_flow_model.pth') -> bool:
        """
        加载已保存的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否成功加载模型
        """
        try:
            # 加载模型状态字典
            state_dict = torch.load(model_path, map_location=self.device)
            self.flow_model.load_state_dict(state_dict)
            
            # 生成一些训练数据来计算标准化参数
            print("   重新计算数据标准化参数...")
            positions, joint_angles = self.generate_training_data(10000)
            self.normalize_data(positions, joint_angles)
            
            print(f"   成功加载模型: {model_path}")
            return True
        except Exception as e:
            print(f"   加载模型失败: {e}")
            return False
    
    def inverse_kinematics(self, target_x: float, target_y: float, num_samples: int = 50) -> Tuple[float, float]:
        """使用训练好的模型计算逆运动学"""
        self.flow_model.eval()
        
        # 标准化目标位置
        target_x_norm = (target_x - self.x_mean) / self.x_std
        target_y_norm = (target_y - self.y_mean) / self.y_std
        condition = torch.tensor([[target_x_norm, target_y_norm]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # 从条件分布采样
            samples = self.flow_model.sample_given_condition(condition, num_samples=num_samples)
            samples = samples.squeeze(0).cpu().numpy()  # [num_samples, 2]
            
            # 反标准化关节角度
            theta1_samples = samples[:, 0] * self.theta1_std + self.theta1_mean
            theta2_samples = samples[:, 1] * self.theta2_std + self.theta2_mean
            
            # 找到最优解
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
    
    def get_multiple_solutions(self, target_x: float, target_y: float, num_samples: int = 100) -> List[Tuple[float, float]]:
        """获取多个逆运动学解"""
        self.flow_model.eval()
        
        # 标准化目标位置
        target_x_norm = (target_x - self.x_mean) / self.x_std
        target_y_norm = (target_y - self.y_mean) / self.y_std
        condition = torch.tensor([[target_x_norm, target_y_norm]], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # 从条件分布采样
            samples = self.flow_model.sample_given_condition(condition, num_samples=num_samples)
            samples = samples.squeeze(0).cpu().numpy()  # [num_samples, 2]
            
            # 反标准化关节角度
            theta1_samples = samples[:, 0] * self.theta1_std + self.theta1_mean
            theta2_samples = samples[:, 1] * self.theta2_std + self.theta2_mean
            
            # 计算所有解的误差
            solutions = []
            for theta1, theta2 in zip(theta1_samples, theta2_samples):
                (x0, y0), (x1, y1), (x2, y2) = self.robot.forward_kinematics(theta1, theta2)
                error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
                solutions.append((theta1, theta2, error))
            
            # 按误差排序
            solutions.sort(key=lambda x: x[2])
            
            # 返回前几个最佳解
            return [(sol[0], sol[1]) for sol in solutions[:min(10, len(solutions))]]


def test_conditional_ik():
    """测试条件逆运动学求解器"""
    print("创建机器人和条件逆运动学求解器...")
    
    # 创建机器人
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建条件逆运动学求解器
    ik_solver = ConditionalInverseKinematicsFlow(robot, device=device)
    
    # 训练模型
    print("\n开始训练...")
    losses = ik_solver.train(num_samples=80000, epochs=150, batch_size=1024, lr=1e-4)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Conditional Flow Training Loss Curve')
    plt.grid(True)
    plt.show()
    
    # 测试逆运动学
    print("\n测试条件逆运动学求解...")
    test_targets = [
        (1.2, 0.8),
        (0.5, 1.6),
        (-1.4, 0.6),
        (0.2, 1.7),
        (1.6, -0.4)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (target_x, target_y) in enumerate(test_targets):
        if i >= len(axes):
            break
            
        print(f"\n目标位置: ({target_x:.2f}, {target_y:.2f})")
        
        # 获取多个解
        solutions = ik_solver.get_multiple_solutions(target_x, target_y, num_samples=200)
        
        if solutions:
            # 使用最佳解
            theta1_best, theta2_best = solutions[0]
            
            # 验证解的准确性
            (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1_best, theta2_best)
            error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
            
            print(f"最佳解: θ1={np.degrees(theta1_best):.1f}°, θ2={np.degrees(theta2_best):.1f}°")
            print(f"实际末端位置: ({x2:.3f}, {y2:.3f})")
            print(f"位置误差: {error:.4f}")
            print(f"找到 {len(solutions)} 个候选解")
            
            # 可视化结果
            robot.visualize_static(theta1_best, theta2_best, ax=axes[i])
            axes[i].plot(target_x, target_y, 'r*', markersize=20, label='目标位置')
            axes[i].plot(x2, y2, 'g*', markersize=15, label='实际位置')
            
            # 显示其他候选解的末端位置
            if len(solutions) > 1:
                other_x, other_y = [], []
                for theta1, theta2 in solutions[1:min(5, len(solutions))]:
                    (_, _), (_, _), (x_end, y_end) = robot.forward_kinematics(theta1, theta2)
                    other_x.append(x_end)
                    other_y.append(y_end)
                axes[i].plot(other_x, other_y, 'b.', markersize=8, alpha=0.7, label='其他解')
            
            axes[i].set_title(f'目标: ({target_x:.1f}, {target_y:.1f})\n误差: {error:.4f}')
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, '无解', transform=axes[i].transAxes, 
                        ha='center', va='center', fontsize=20)
            axes[i].set_title(f'目标: ({target_x:.1f}, {target_y:.1f})\n无解')
    
    # 隐藏多余的子图
    for i in range(len(test_targets), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_conditional_ik() 
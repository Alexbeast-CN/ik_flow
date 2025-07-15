"""
UR5机器人 IK Flow Pipeline
结合FK数据采样和IK Flow训练的完整流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, List
import warnings
import logging
warnings.filterwarnings('ignore')

# 配置日志
def setup_logging():
    """配置日志系统"""
    # 创建logs目录
    os.makedirs('logs', exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ur5_ik_flow_pipeline.log', encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 获取logger
    logger = logging.getLogger(__name__)
    return logger

from fk_sampler import BatchForwardKinematics, generate_fk_dataset, calculate_sampling_parameters, save_fk_dataset


class UR5Robot:
    """UR5机器人接口，简化版本用于IK Flow"""
    
    def __init__(self, urdf_path="ur5.urdf", device="cpu"):
        self.urdf_path = urdf_path
        self.device = device
        self.fk_solver = BatchForwardKinematics(urdf_path, device)
        self.n_joints = self.fk_solver.n_joints
        
    def forward_kinematics(self, joint_angles):
        """
        单个配置的正向运动学
        
        Args:
            joint_angles: 关节角度，shape为(n_joints,)或(1, n_joints)
            
        Returns:
            position: 末端执行器位置 (x, y, z)
            quaternion: 末端执行器四元数 (qx, qy, qz, qw)
        """
        if isinstance(joint_angles, (list, tuple)):
            joint_angles = np.array(joint_angles)
        
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.reshape(1, -1)
        
        joint_angles_tensor = torch.tensor(joint_angles, dtype=torch.float32, device=self.device)
        
        # 计算FK
        transforms = self.fk_solver.compute_fk(joint_angles_tensor, end_link="tool0")
        
        # 提取位置和旋转
        position = transforms[0, :3, 3].cpu().numpy()  # (3,)
        rotation_matrix = transforms[0, :3, :3].cpu().numpy()  # (3, 3)
        
        # 转换为四元数
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        return position, quaternion
    
    def _rotation_matrix_to_quaternion(self, R):
        """将旋转矩阵转换为四元数"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])


class ConditionalCouplingLayerUR5(nn.Module):
    """适用于UR5的条件耦合层"""
    
    def __init__(self, input_dim=6, condition_dim=7, hidden_dim=512, mask_type=0):
        super(ConditionalCouplingLayerUR5, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.mask_type = mask_type
        
        # 创建掩码 - 交替掩码前半部分和后半部分关节
        if mask_type == 0:
            self.mask = torch.cat([torch.ones(3), torch.zeros(3)])  # [1,1,1,0,0,0]
        else:
            self.mask = torch.cat([torch.zeros(3), torch.ones(3)])  # [0,0,0,1,1,1]
        
        # 被掩码的维度数
        masked_dim = int(self.mask.sum().item())
        
        # 条件网络的输入维度
        condition_input_dim = masked_dim + condition_dim
        
        # 尺度网络
        self.scale_net = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim - masked_dim),
            nn.Tanh()
        )
        
        # 平移网络
        self.translation_net = nn.Sequential(
            nn.Linear(condition_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim - masked_dim)
        )
    
    def forward(self, x, condition, reverse=False):
        """条件耦合变换"""
        # 移动mask到正确设备
        mask = self.mask.to(x.device)
        mask_bool = mask.bool()
        inv_mask_bool = ~mask_bool
        
        # 分离被掩码和待变换的部分
        x_masked = x * mask
        x_transform = x * (1 - mask)
        
        # 提取非零部分
        x_masked_values = x_masked[:, mask_bool]
        x_transform_values = x_transform[:, inv_mask_bool]
        
        # 构建条件输入
        condition_input = torch.cat([x_masked_values, condition], dim=1)
        
        # 计算变换参数
        scale = self.scale_net(condition_input) * 2.0
        translation = self.translation_net(condition_input)
        
        if reverse:
            # 逆变换
            x_new_transform = (x_transform_values - translation) / torch.exp(scale)
            log_det = -scale.sum(dim=1)
        else:
            # 正向变换
            x_new_transform = x_transform_values * torch.exp(scale) + translation
            log_det = scale.sum(dim=1)
        
        # 重构输出
        x_new = x.clone()
        x_new[:, inv_mask_bool] = x_new_transform
        
        return x_new, log_det


class ConditionalRealNVPUR5(nn.Module):
    """适用于UR5的条件Real NVP模型"""
    
    def __init__(self, input_dim=6, condition_dim=7, hidden_dim=512, num_layers=12):
        super(ConditionalRealNVPUR5, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        
        # 创建条件耦合层
        self.coupling_layers = nn.ModuleList()
        for i in range(num_layers):
            self.coupling_layers.append(
                ConditionalCouplingLayerUR5(input_dim, condition_dim, hidden_dim, mask_type=(i % 2))
            )
    
    def forward(self, x, condition, reverse=False):
        """条件正向/反向变换"""
        if reverse:
            return self._inverse(x, condition)
        else:
            return self._forward(x, condition)
    
    def _forward(self, x, condition):
        """正向变换: 关节角度 -> 隐空间"""
        z = x
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)
        
        for coupling_layer in self.coupling_layers:
            z, log_det = coupling_layer(z, condition)
            log_det_jacobian += log_det
        
        return z, log_det_jacobian
    
    def _inverse(self, z, condition):
        """逆向变换: 隐空间 -> 关节角度"""
        x = z
        log_det_jacobian = torch.zeros(z.shape[0], device=z.device)
        
        for coupling_layer in reversed(self.coupling_layers):
            x, log_det = coupling_layer(x, condition, reverse=True)
            log_det_jacobian += log_det
        
        return x, log_det_jacobian
    
    def sample_given_condition(self, condition, num_samples=1):
        """给定条件采样关节角度"""
        batch_size = condition.shape[0]
        if num_samples > 1:
            condition = condition.repeat(num_samples, 1)
            total_samples = batch_size * num_samples
        else:
            total_samples = batch_size
        
        # 从标准正态分布采样
        z = torch.randn(total_samples, self.input_dim, device=condition.device)
        
        # 条件逆变换
        x, _ = self.forward(z, condition, reverse=True)
        
        if num_samples > 1:
            x = x.reshape(batch_size, num_samples, self.input_dim)
        
        return x
    
    def log_prob_given_condition(self, x, condition):
        """计算给定条件下的对数概率密度"""
        z, log_det_jacobian = self.forward(x, condition)
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob_z + log_det_jacobian


class UR5InverseKinematicsFlow:
    """UR5逆运动学Flow求解器"""
    
    def __init__(self, robot, device='cpu'):
        self.robot = robot
        self.device = device
        
        # 创建条件Flow模型
        self.flow_model = ConditionalRealNVPUR5(
            input_dim=6,    # 6个关节
            condition_dim=7, # 位置3 + 四元数4
            hidden_dim=512,
            num_layers=12
        ).to(device)
        
        # 数据标准化参数
        self.joint_mean = torch.zeros(6)
        self.joint_std = torch.ones(6)
        self.pos_mean = torch.zeros(3)
        self.pos_std = torch.ones(3)
        self.quat_mean = torch.zeros(4)
        self.quat_std = torch.ones(4)
    
    def load_fk_dataset(self, fk_data_path):
        """加载FK数据集"""
        logger = logging.getLogger(__name__)
        logger.info(f"加载FK数据集: {fk_data_path}")
        fk_data = torch.load(fk_data_path, map_location='cpu')
        logger.info(f"数据形状: {fk_data.shape}")
        
        # 分离关节角度和位姿数据
        joint_angles = fk_data[:, :6]  # 前6列是关节角度
        positions = fk_data[:, 6:9]    # 接下来3列是位置
        quaternions = fk_data[:, 9:13] # 最后4列是四元数
        
        logger.info(f"关节角度形状: {joint_angles.shape}")
        logger.info(f"位置形状: {positions.shape}")
        logger.info(f"四元数形状: {quaternions.shape}")
        
        return joint_angles, positions, quaternions
    
    def normalize_data(self, joint_angles, positions, quaternions):
        """数据标准化"""
        # 计算标准化参数
        self.joint_mean = joint_angles.mean(dim=0)
        self.joint_std = joint_angles.std(dim=0)
        self.pos_mean = positions.mean(dim=0)
        self.pos_std = positions.std(dim=0)
        self.quat_mean = quaternions.mean(dim=0)
        self.quat_std = quaternions.std(dim=0)
        
        # 避免除零
        self.joint_std = torch.clamp(self.joint_std, min=1e-6)
        self.pos_std = torch.clamp(self.pos_std, min=1e-6)
        self.quat_std = torch.clamp(self.quat_std, min=1e-6)
        
        # 标准化
        joint_angles_norm = (joint_angles - self.joint_mean) / self.joint_std
        positions_norm = (positions - self.pos_mean) / self.pos_std
        quaternions_norm = (quaternions - self.quat_mean) / self.quat_std
        
        return joint_angles_norm, positions_norm, quaternions_norm
    
    def train(self, fk_data_path, epochs=100, batch_size=102400, lr=1e-4):
        """训练IK Flow模型"""
        import datetime
        logger = logging.getLogger(__name__)
        logger.info("开始训练UR5 IK Flow模型...")
        start_time = datetime.datetime.now()
        logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 加载FK数据
        joint_angles, positions, quaternions = self.load_fk_dataset(fk_data_path)
        
        # 标准化数据
        joint_angles_norm, positions_norm, quaternions_norm = self.normalize_data(
            joint_angles, positions, quaternions
        )
        
        # 组合条件：位置 + 四元数
        conditions = torch.cat([positions_norm, quaternions_norm], dim=1)
        
        # 创建数据加载器
        dataset = TensorDataset(joint_angles_norm, conditions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和调度器
        optimizer = optim.AdamW(self.flow_model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练循环
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = datetime.datetime.now()
            logger.info(f"\n===== Epoch {epoch + 1}/{epochs} 开始 =====")
            logger.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            self.flow_model.train()
            epoch_loss = 0.0
            num_batches = 0
            batch_losses = []
            
            for batch_idx, (batch_joints, batch_conditions) in enumerate(dataloader):
                batch_joints = batch_joints.to(self.device)
                batch_conditions = batch_conditions.to(self.device)
                
                optimizer.zero_grad()
                
                # 计算对数似然
                log_prob = self.flow_model.log_prob_given_condition(batch_joints, batch_conditions)
                loss = -log_prob.mean()
                
                # 正则化
                l2_reg = sum(torch.norm(param, p=2) for param in self.flow_model.parameters())
                loss += 1e-6 * l2_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_losses.append(loss.item())
                num_batches += 1
                
                # 每10个batch打印一次
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    avg_batch_loss = np.mean(batch_losses[-10:]) if len(batch_losses) >= 10 else np.mean(batch_losses)
                    logger.info(f"  [Epoch {epoch + 1}/{epochs}] Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg(last 10): {avg_batch_loss:.4f}")
            
            scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model('best_ur5_ik_flow_model.pth')
                logger.info(f"  [Epoch {epoch + 1}] 新最佳模型已保存，平均损失: {avg_loss:.4f}")
            
            epoch_end_time = datetime.datetime.now()
            epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
            logger.info(f"===== Epoch {epoch + 1}/{epochs} 结束 | 平均损失: {avg_loss:.4f} | 用时: {epoch_duration:.1f}秒 =====")
        
        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"训练完成! 总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
        return losses
    def save_model(self, model_path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.flow_model.state_dict(),
            'joint_mean': self.joint_mean,
            'joint_std': self.joint_std,
            'pos_mean': self.pos_mean,
            'pos_std': self.pos_std,
            'quat_mean': self.quat_mean,
            'quat_std': self.quat_std
        }, model_path)
    
    def load_model(self, model_path):
        """加载模型"""
        logger = logging.getLogger(__name__)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.flow_model.load_state_dict(checkpoint['model_state_dict'])
            
            self.joint_mean = checkpoint['joint_mean']
            self.joint_std = checkpoint['joint_std']
            self.pos_mean = checkpoint['pos_mean']
            self.pos_std = checkpoint['pos_std']
            self.quat_mean = checkpoint['quat_mean']
            self.quat_std = checkpoint['quat_std']
            
            logger.info(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def inverse_kinematics(self, target_pos, target_quat, num_samples=50):
        """求解逆运动学"""
        self.flow_model.eval()
        
        # 转换为tensor
        target_pos = torch.tensor(target_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
        target_quat = torch.tensor(target_quat, dtype=torch.float32, device=self.device).reshape(1, 4)
        
        # 标准化条件
        pos_norm = (target_pos - self.pos_mean.to(self.device)) / self.pos_std.to(self.device)
        quat_norm = (target_quat - self.quat_mean.to(self.device)) / self.quat_std.to(self.device)
        condition = torch.cat([pos_norm, quat_norm], dim=1)
        
        with torch.no_grad():
            # 采样
            samples = self.flow_model.sample_given_condition(condition, num_samples=num_samples)
            samples = samples.squeeze(0)  # [num_samples, 6]
            
            # 反标准化
            joint_angles_samples = samples * self.joint_std.to(self.device) + self.joint_mean.to(self.device)
            
            # 找到最优解
            best_error = float('inf')
            best_joint_angles = None
            
            for i in range(joint_angles_samples.shape[0]):
                joint_angles = joint_angles_samples[i].cpu().numpy()
                
                # 计算FK验证
                pred_pos, pred_quat = self.robot.forward_kinematics(joint_angles)
                
                # 计算误差
                pos_error = np.linalg.norm(pred_pos - target_pos.cpu().numpy().flatten())
                quat_error = np.linalg.norm(pred_quat - target_quat.cpu().numpy().flatten())
                total_error = pos_error + 0.1 * quat_error  # 位置误差权重更高
                
                if total_error < best_error:
                    best_error = total_error
                    best_joint_angles = joint_angles
        
        return best_joint_angles, best_error


def run_ur5_ik_flow_pipeline():
    """运行完整的UR5 IK Flow pipeline"""
    # 设置日志
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("UR5 IK Flow Pipeline")
    logger.info("="*60)
    
    # 配置参数
    urdf_path = "ur5.urdf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_samples_per_joint = 20  # 每个关节20个采样点（demo用小数据）
    max_batch_size = int(1e6)
    end_link = "tool0"
    fk_save_dir = "fk_results"
    fk_filename = "ur5_fk_dataset_demo.pt"
    fk_data_path = os.path.join(fk_save_dir, fk_filename)
    
    logger.info(f"使用设备: {device}")
    logger.info(f"每个关节采样点数: {n_samples_per_joint}")
    
    # 步骤1: 生成FK数据集
    logger.info("\n" + "="*40)
    logger.info("步骤1: 生成FK数据集")
    logger.info("="*40)
    
    if not os.path.exists(fk_data_path):
        logger.info("FK数据集不存在，开始生成...")
        
        # 初始化FK求解器
        fk_solver = BatchForwardKinematics(urdf_path, device=device)
        
        # 计算采样参数
        sampling_params = calculate_sampling_parameters(
            fk_solver, 
            n_samples_per_joint=n_samples_per_joint,
            max_batch_size=max_batch_size,
            use_regular_batching=False
        )
        
        # 生成FK数据集
        data, total_processed = generate_fk_dataset(
            fk_solver, sampling_params, end_link=end_link
        )
        
        # 保存数据集
        if data is not None:
            save_fk_dataset(data, fk_solver, fk_save_dir, fk_filename)
            logger.info(f"FK数据集生成完成！总计 {total_processed:,} 个样本")
        else:
            logger.error("FK数据集生成失败！")
            return
    else:
        logger.info(f"找到现有FK数据集: {fk_data_path}")
    
    # 步骤2: 训练IK Flow模型
    logger.info("\n" + "="*40)
    logger.info("步骤2: 训练IK Flow模型")
    logger.info("="*40)
    
    # 创建UR5机器人接口
    ur5_robot = UR5Robot(urdf_path, device=device)
    
    # 创建IK Flow求解器
    ik_solver = UR5InverseKinematicsFlow(ur5_robot, device=device)
    
    # 尝试加载已保存的模型
    model_loaded = ik_solver.load_model('best_ur5_ik_flow_model.pth')
    
    if not model_loaded:
        logger.info("未找到已保存的模型，开始训练...")
        
        # 训练模型
        losses = ik_solver.train(
            fk_data_path=fk_data_path,
            epochs=100,
            batch_size=25600,
            lr=1e-4
        )
        
        # 可视化训练过程
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('UR5 IK Flow Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('ur5_ik_flow_training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        logger.info("模型加载成功，跳过训练步骤")
    
    # 步骤3: 测试IK求解
    logger.info("\n" + "="*40)
    logger.info("步骤3: 测试IK求解")
    logger.info("="*40)
    
    # 定义测试目标位姿
    test_targets = [
        {
            'pos': [0.5, 0.3, 0.8],
            'quat': [0.0, 0.0, 0.0, 1.0],  # 无旋转
            'description': '前方位置'
        },
        {
            'pos': [0.0, 0.6, 0.4],
            'quat': [0.707, 0.0, 0.0, 0.707],  # 绕X轴90度
            'description': '左侧位置'
        },
        {
            'pos': [0.3, -0.4, 0.6],
            'quat': [0.0, 0.707, 0.0, 0.707],  # 绕Y轴90度
            'description': '右后方位置'
        }
    ]
    
    logger.info(f"\n测试 {len(test_targets)} 个目标位姿:")
    logger.info("-" * 80)
    logger.info(f"{'描述':<12} {'目标位置':<20} {'目标四元数':<25} {'误差':<10}")
    logger.info("-" * 80)
    
    for target in test_targets:
        try:
            # 求解IK
            joint_angles, error = ik_solver.inverse_kinematics(
                target['pos'], target['quat'], num_samples=100
            )
            
            # 验证结果
            pred_pos, pred_quat = ur5_robot.forward_kinematics(joint_angles)
            
            logger.info(f"{target['description']:<12} "
                  f"{str(target['pos']):<20} "
                  f"{str([f'{q:.3f}' for q in target['quat']]):<25} "
                  f"{error:<10.4f}")
            
            logger.info(f"  -> 关节角度 (度): {[f'{np.degrees(q):.1f}' for q in joint_angles]}")
            logger.info(f"  -> 实际位置: {[f'{p:.3f}' for p in pred_pos]}")
            logger.info(f"  -> 实际四元数: {[f'{q:.3f}' for q in pred_quat]}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"{target['description']:<12} 求解失败: {e}")
            logger.info("")
    
    logger.info("-" * 80)
    logger.info("\nPipeline完成!")
    logger.info("主要成果:")
    logger.info("1. 生成了UR5的FK数据集")
    logger.info("2. 训练了基于Normalizing Flow的IK模型")
    logger.info("3. 验证了IK求解的准确性")
    logger.info("4. 模型可以处理6DOF机器人的复杂逆运动学问题")


if __name__ == "__main__":
    run_ur5_ik_flow_pipeline() 
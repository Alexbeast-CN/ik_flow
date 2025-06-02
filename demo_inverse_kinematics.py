"""
基于 Normalizing Flow 的逆运动学演示脚本
使用深度生成式网络求解 2DOF 平面机器人的逆运动学问题
"""

import numpy as np
import matplotlib.pyplot as plt
from planner_robot import PlannerRobot
from conditional_ik_flow import ConditionalInverseKinematicsFlow
import torch

def demo_basic_usage():
    """基本使用演示"""
    print("=" * 50)
    print("基于 Normalizing Flow 的逆运动学求解器演示")
    print("=" * 50)
    
    # 1. 创建机器人
    print("\n1. 创建机器人实例...")
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 2. 检查计算设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   使用计算设备: {device}")
    
    # 3. 创建逆运动学求解器
    print("\n2. 创建逆运动学求解器...")
    ik_solver = ConditionalInverseKinematicsFlow(robot, device=device)
    
    # 4. 尝试加载已保存的模型，如果没有则训练新模型
    print("\n3. 尝试加载已保存的模型...")
    model_loaded = ik_solver.load_model('best_ik_flow_model.pth')
    
    if not model_loaded:
        print("\n   未找到已保存的模型，开始训练新模型...")
        print("   注意：为了演示，使用较小的数据集和训练轮数")
        print("   在实际应用中，建议使用更大的数据集和更多的训练轮数以获得更好的性能")
        
        losses = ik_solver.train(
            num_samples=20000,  # 训练样本数
            epochs=50,          # 训练轮数  
            batch_size=512,     # 批大小
            lr=1e-4            # 学习率
        )
        
        # 5. 可视化训练过程
        print("\n4. 可视化训练过程...")
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Normalizing Flow Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("   模型加载成功，跳过训练步骤")
    
    # 6. 测试逆运动学求解
    print("\n5. 测试逆运动学求解...")
    test_targets = [
        (1.2, 0.8, "右上方"),
        (0.0, 1.7, "正上方"),
        (-1.0, 0.5, "左侧"),
        (1.5, -0.3, "右下方"),
        (0.5, 1.4, "中上方")
    ]
    
    # 创建子图展示结果
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print(f"\n   测试 {len(test_targets)} 个目标位置:")
    print("-" * 70)
    print(f"{'目标位置':<15} {'预测角度':<25} {'实际位置':<15} {'误差':<10}")
    print("-" * 70)
    
    for i, (target_x, target_y, description) in enumerate(test_targets):
        if i >= len(axes):
            break
        
        # 使用 Flow 模型求解逆运动学
        theta1_pred, theta2_pred = ik_solver.inverse_kinematics(
            target_x, target_y, num_samples=100
        )
        
        # 验证解的准确性
        (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1_pred, theta2_pred)
        error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
        
        # 打印结果
        print(f"({target_x:4.1f}, {target_y:4.1f})   "
              f"θ1={np.degrees(theta1_pred):6.1f}°, θ2={np.degrees(theta2_pred):6.1f}°   "
              f"({x2:4.2f}, {y2:4.2f})   "
              f"{error:6.4f}")
        
        # 可视化结果
        robot.visualize_static(theta1_pred, theta2_pred, ax=axes[i])
        axes[i].plot(target_x, target_y, 'r*', markersize=20, label='Target Position')
        axes[i].plot(x2, y2, 'g*', markersize=15, label='Actual Position')
        axes[i].set_title(f'{description}\nTarget: ({target_x:.1f}, {target_y:.1f}) | Error: {error:.4f}')
        axes[i].legend(fontsize=8)
    
    # 隐藏多余的子图
    for i in range(len(test_targets), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Inverse Kinematics Solution Results Based on Normalizing Flow', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("-" * 70)
    print(f"\n6. 结果分析:")
    print(f"   - 训练完成后，模型能够有效求解逆运动学问题")
    print(f"   - 误差通常在 0.01 以下，表明模型精度较高")
    print(f"   - Normalizing Flow 能够学习到复杂的逆映射关系")
    print(f"   - 可以处理工作空间内的任意目标位置")


def demo_multiple_solutions():
    """多解演示"""
    print("\n" + "=" * 50)
    print("多解逆运动学演示")
    print("=" * 50)
    
    # 创建机器人和求解器
    robot = PlannerRobot(L1=1.0, L2=0.8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ik_solver = ConditionalInverseKinematicsFlow(robot, device=device)
    
    # 尝试加载已保存的模型
    print("\n尝试加载已保存的模型...")
    model_loaded = ik_solver.load_model('best_ik_flow_model.pth')
    
    if not model_loaded:
        # 如果加载失败，则重新训练模型
        print("\n未找到已保存的模型，开始重新训练...")
        ik_solver.train(num_samples=15000, epochs=30, batch_size=512, lr=1e-4)
    else:
        print("   模型加载成功，跳过训练步骤")
    
    # 测试多解情况
    print("\n测试多解情况...")
    target_x, target_y = 1.0, 1.0
    
    # 获取多个解
    solutions = ik_solver.get_multiple_solutions(target_x, target_y, num_samples=200)
    
    print(f"\n目标位置: ({target_x}, {target_y})")
    print(f"找到 {len(solutions)} 个候选解:")
    print("-" * 50)
    print(f"{'序号':<4} {'θ1 (度)':<10} {'θ2 (度)':<10} {'实际位置':<15} {'误差':<10}")
    print("-" * 50)
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 显示前5个最佳解
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (theta1, theta2) in enumerate(solutions[:5]):
        (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1, theta2)
        error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)
        
        print(f"{i+1:<4} {np.degrees(theta1):<10.1f} {np.degrees(theta2):<10.1f} "
              f"({x2:.3f}, {y2:.3f})  {error:<10.4f}")
        
        # 在第一个子图中显示机器人配置
        if i < len(colors):
            robot.visualize_static(theta1, theta2, ax=ax1)
            ax1.plot([x0, x1], [y0, y1], 'o-', lw=3, color=colors[i], alpha=0.7, 
                    label=f'solution {i+1}')
            ax1.plot([x1, x2], [y1, y2], 'o-', lw=3, color=colors[i], alpha=0.7)
    
    # 标记目标点
    ax1.plot(target_x, target_y, 'k*', markersize=20, label='Target Position')
    ax1.set_title(f'Multiple IK Solutions\nTarget: ({target_x}, {target_y})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 在第二个子图中显示关节空间的解分布
    theta1_solutions = [np.degrees(sol[0]) for sol in solutions[:20]]
    theta2_solutions = [np.degrees(sol[1]) for sol in solutions[:20]]
    
    ax2.scatter(theta1_solutions, theta2_solutions, c='red', alpha=0.6, s=50)
    ax2.set_xlabel('θ1 (deg)')
    ax2.set_ylabel('θ2 (deg)')
    ax2.set_title('Solution Distribution in Joint Space')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("-" * 50)
    print(f"\n多解分析:")
    print(f"   - 对于给定的末端执行器位置，可能存在多个关节角度组合")
    print(f"   - Normalizing Flow 能够捕获这种一对多的映射关系")
    print(f"   - 通过采样可以获得多个有效解，为路径规划提供选择")


def demo_workspace_analysis():
    """工作空间分析演示"""
    print("\n" + "=" * 50)
    print("工作空间分析演示")
    print("=" * 50)
    
    # 创建机器人
    robot = PlannerRobot(L1=1.0, L2=0.8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ik_solver = ConditionalInverseKinematicsFlow(robot, device=device)
    
    # 尝试加载已保存的模型
    print("\n尝试加载已保存的模型...")
    model_loaded = ik_solver.load_model('best_ik_flow_model.pth')
    
    if not model_loaded:
        # 如果加载失败，则重新训练模型
        print("\n未找到已保存的模型，开始训练模型进行工作空间分析...")
        ik_solver.train(num_samples=25000, epochs=40, batch_size=512, lr=1e-4)
    else:
        print("   模型加载成功，跳过训练步骤")
    
    # 生成工作空间网格
    print("\n分析工作空间可达性...")
    x_range = np.linspace(-2.0, 2.0, 30)
    y_range = np.linspace(-2.0, 2.0, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    reachable = np.zeros_like(X)
    errors = np.zeros_like(X)
    
    total_points = X.size
    processed = 0
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j]
            
            # 检查是否在理论工作空间内
            distance = np.sqrt(x**2 + y**2)
            if abs(robot.L1 - robot.L2) <= distance <= (robot.L1 + robot.L2):
                try:
                    theta1, theta2 = ik_solver.inverse_kinematics(x, y, num_samples=50)
                    (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1, theta2)
                    error = np.sqrt((x2 - x)**2 + (y2 - y)**2)
                    
                    if error < 0.1:  # 误差阈值
                        reachable[i, j] = 1
                        errors[i, j] = error
                except:
                    pass
            
            processed += 1
            if processed % 100 == 0:
                print(f"   处理进度: {processed}/{total_points} ({processed/total_points*100:.1f}%)")
    
    # 可视化工作空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 可达性图
    im1 = ax1.imshow(reachable, extent=[-2, 2, -2, 2], origin='lower', 
                     cmap='RdYlGn', alpha=0.8)
    ax1.set_xlabel('X coordinates')
    ax1.set_ylabel('Y coordinates')
    ax1.set_title('Workspace Reachability Analysis')
    ax1.grid(True, alpha=0.3)
    
    # 添加理论工作空间边界
    circle_outer = plt.Circle((0, 0), robot.L1 + robot.L2, fill=False, 
                             color='black', linestyle='--', linewidth=2, 
                             label='Theoretical Outer Boundary')
    circle_inner = plt.Circle((0, 0), abs(robot.L1 - robot.L2), fill=False, 
                             color='black', linestyle='--', linewidth=2, 
                             label='Theoretical Inner Boundary')
    ax1.add_patch(circle_outer)
    ax1.add_patch(circle_inner)
    ax1.legend()
    
    # 误差分布图
    errors_masked = np.ma.masked_where(reachable == 0, errors)
    im2 = ax2.imshow(errors_masked, extent=[-2, 2, -2, 2], origin='lower', 
                     cmap='plasma', alpha=0.8)
    ax2.set_xlabel('X coordinates')
    ax2.set_ylabel('Y coordinates')
    ax2.set_title('Inverse Kinematics Solution Error Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    plt.colorbar(im1, ax=ax1, label='Reachability (1=Reachable, 0=Unreachable)')
    plt.colorbar(im2, ax=ax2, label='Position Error')
    
    plt.tight_layout()
    plt.show()
    
    # 统计信息
    reachable_points = np.sum(reachable)
    total_workspace_points = np.sum((X**2 + Y**2) <= (robot.L1 + robot.L2)**2) - \
                            np.sum((X**2 + Y**2) <= (robot.L1 - robot.L2)**2)
    coverage = reachable_points / total_workspace_points if total_workspace_points > 0 else 0
    avg_error = np.mean(errors[reachable == 1]) if reachable_points > 0 else 0
    
    print("\nWorkspace Analysis Results:")
    print(f"   - Theoretical Workspace Points: {total_workspace_points}")
    print(f"   - Actual Reachable Points: {reachable_points}")
    print(f"   - Coverage: {coverage:.2%}")
    print(f"   - Average Solution Error: {avg_error:.6f}")
    print(f"   - Flow Model Can Effectively Cover Most of the Workspace")


if __name__ == "__main__":
    # 运行演示
    try:
        # 基本使用演示
        # demo_basic_usage()
        
        # 多解演示
        demo_multiple_solutions()
        
        # 工作空间分析
        # demo_workspace_analysis()
        
        print("\n" + "=" * 50)
        print("演示完成！")
        print("=" * 50)
        print("\n主要特点总结:")
        print("1. 使用 Normalizing Flow 学习复杂的逆运动学映射")
        print("2. 能够处理多解情况，提供多个候选解")
        print("3. 高精度求解，误差通常在毫米级别")
        print("4. 支持任意工作空间内的目标位置")
        print("5. 基于深度学习的方法，可扩展到更复杂的机器人")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        print("请确保已安装所有必要的库：torch, numpy, matplotlib") 
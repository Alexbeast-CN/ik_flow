#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlannerRobot 演示脚本 (PlannerRobot Demo Script)

展示如何使用 PlannerRobot 类的各种功能：
1. 正向运动学计算
2. 静态可视化
3. 交互式可视化
4. 动画演示
"""

import numpy as np
import matplotlib.pyplot as plt
from planner_robot import PlannerRobot

def demo_forward_kinematics():
    """演示正向运动学计算 (Demonstrate forward kinematics calculation)"""
    print("=== 演示1: 正向运动学计算 ===")
    
    # 创建机器人实例 (Create robot instance)
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 定义一些测试角度 (Define some test angles)
    test_angles = [
        (0, 0),                    # 完全伸展 (Fully extended)
        (np.pi/2, 0),              # 向上伸展 (Extended upward)
        (np.pi/4, np.pi/4),        # 45度角 (45 degree angles)
        (np.pi, np.pi/2),          # 向左并弯曲 (Left and bent)
    ]
    
    for i, (theta1, theta2) in enumerate(test_angles):
        (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1, theta2)
        print(f"测试 {i+1}:")
        print(f"  输入角度: θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°")
        print(f"  基座位置: ({x0:.3f}, {y0:.3f})")
        print(f"  关节1位置: ({x1:.3f}, {y1:.3f})")
        print(f"  末端位置: ({x2:.3f}, {y2:.3f})")
        print()

def demo_static_visualization():
    """演示静态可视化 (Demonstrate static visualization)"""
    print("=== 演示2: 静态可视化 ===")
    
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 创建2x2的子图 (Create 2x2 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PlannerRobot 静态位置演示 (Static Position Demonstrations)', fontsize=16)
    
    # 四个不同的配置 (Four different configurations)
    configurations = [
        (0, 0, "完全伸展 (Fully Extended)"),
        (np.pi/2, 0, "向上伸展 (Extended Upward)"),
        (np.pi/4, np.pi/4, "45度配置 (45° Configuration)"),
        (np.pi, np.pi/2, "弯曲配置 (Bent Configuration)")
    ]
    
    for i, (theta1, theta2, title) in enumerate(configurations):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        robot.visualize_static(theta1, theta2, ax=ax)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def demo_interactive_visualization():
    """演示交互式可视化 (Demonstrate interactive visualization)"""
    print("=== 演示3: 交互式可视化 ===")
    print("请使用滑块调节机器人的关节角度")
    print("观察机器人形态的实时变化以及末端执行器位置")
    
    robot = PlannerRobot(L1=1.0, L2=0.8)
    robot.visualize_interactive()

def demo_animations():
    """演示各种动画 (Demonstrate various animations)"""
    print("=== 演示4: 动画演示 ===")
    
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 动画1: 简单圆周运动 (Animation 1: Simple circular motion)
    def circular_motion(t):
        theta1 = t * 2 * np.pi
        theta2 = 0
        return theta1, theta2
    
    print("动画1: 简单圆周运动")
    robot.animate_motion(circular_motion, total_frames=100, interval_ms=50)
    robot.clear_path()  # 清除轨迹为下一个动画做准备
    
    # 动画2: 复杂的摆动运动 (Animation 2: Complex swinging motion)
    def swinging_motion(t):
        theta1 = 0.5 * np.sin(t * 2 * np.pi)
        theta2 = 0.8 * np.sin(t * 3 * np.pi)
        return theta1, theta2
    
    print("动画2: 复杂摆动运动")
    robot.animate_motion(swinging_motion, total_frames=150, interval_ms=40)
    robot.clear_path()
    
    # 动画3: 八字形轨迹 (Animation 3: Figure-8 trajectory)
    def figure_eight_motion(t):
        theta1 = np.pi/3 + 0.5 * np.sin(t * 2 * np.pi)
        theta2 = 0.6 * np.sin(t * 4 * np.pi)
        return theta1, theta2
    
    print("动画3: 八字形轨迹")
    robot.animate_motion(figure_eight_motion, total_frames=200, interval_ms=30)

def demo_workspace_analysis():
    """演示工作空间分析 (Demonstrate workspace analysis)"""
    print("=== 演示5: 工作空间分析 ===")
    
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 生成工作空间边界点 (Generate workspace boundary points)
    theta1_range = np.linspace(0, 2*np.pi, 50)
    theta2_range = np.linspace(-np.pi, np.pi, 30)
    
    workspace_x, workspace_y = [], []
    
    for theta1 in theta1_range:
        for theta2 in theta2_range:
            (x0, y0), (x1, y1), (x2, y2) = robot.forward_kinematics(theta1, theta2)
            workspace_x.append(x2)
            workspace_y.append(y2)
    
    # 绘制工作空间 (Plot workspace)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(workspace_x, workspace_y, c='lightblue', alpha=0.6, s=1)
    ax.set_aspect('equal')
    ax.set_xlabel('X 坐标 (X Coordinate)')
    ax.set_ylabel('Y 坐标 (Y Coordinate)')
    ax.set_title('机器人工作空间 (Robot Workspace)')
    ax.grid(True, alpha=0.3)
    
    # 显示理论边界 (Show theoretical boundaries)
    circle_outer = plt.Circle((0, 0), robot.total_reach, fill=False, 
                             color='red', linestyle='--', linewidth=2, label='外边界 (Outer Boundary)')
    circle_inner = plt.Circle((0, 0), abs(robot.L1 - robot.L2), fill=False, 
                             color='red', linestyle='--', linewidth=2, label='内边界 (Inner Boundary)')
    ax.add_patch(circle_outer)
    ax.add_patch(circle_inner)
    ax.legend()
    
    plt.show()

def main():
    """主函数 (Main function)"""
    print("PlannerRobot 类演示程序")
    print("=" * 50)
    
    while True:
        print("\n请选择演示项目:")
        print("1. 正向运动学计算")
        print("2. 静态可视化")
        print("3. 交互式可视化")
        print("4. 动画演示")
        print("5. 工作空间分析")
        print("0. 退出")
        
        try:
            choice = int(input("请输入选择 (0-5): "))
            
            if choice == 0:
                print("程序结束，谢谢使用！")
                break
            elif choice == 1:
                demo_forward_kinematics()
            elif choice == 2:
                demo_static_visualization()
            elif choice == 3:
                demo_interactive_visualization()
            elif choice == 4:
                demo_animations()
            elif choice == 5:
                demo_workspace_analysis()
            else:
                print("无效选择，请重新输入")
        
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break

if __name__ == "__main__":
    main() 
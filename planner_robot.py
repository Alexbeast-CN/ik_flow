import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

class PlannerRobot:
    """2自由度平面机器人规划器 (2-DOF Planar Robot Planner)"""
    
    def __init__(self, L1=1.0, L2=0.8):
        """
        初始化机器人参数 (Initialize robot parameters)
        
        Args:
            L1 (float): 第一个连杆的长度 (Length of the first link)
            L2 (float): 第二个连杆的长度 (Length of the second link)
        """
        self.L1 = L1
        self.L2 = L2
        self.total_reach = L1 + L2  # 最大工作半径 (Maximum reach)
        
        # 存储末端执行器路径的点 (Store points for the end-effector path)
        self.path_x = []
        self.path_y = []
    
    def forward_kinematics(self, theta1, theta2):
        """
        正向运动学计算 (Forward kinematics calculation)
        
        Args:
            theta1 (float): 第一个关节的角度 (rad) (Angle of the first joint in radians)
            theta2 (float): 第二个关节相对于第一个连杆的角度 (rad) (Angle of the second joint relative to the first link in radians)
        
        Returns:
            tuple: ((x0, y0), (x1, y1), (x2, y2)) 三个关键点的坐标
                   x0, y0: 基座位置 (Base position)
                   x1, y1: 第一个关节位置 (First joint position)
                   x2, y2: 末端执行器位置 (End-effector position)
        """
        # 基座位置 (Base position)
        x0, y0 = 0.0, 0.0
        
        # 第一个关节位置 (First joint position, end of link 1)
        x1 = self.L1 * np.cos(theta1)
        y1 = self.L1 * np.sin(theta1)
        
        # 末端执行器位置 (End-effector position, end of link 2)
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        
        return (x0, y0), (x1, y1), (x2, y2)
    
    def visualize_static(self, theta1, theta2, show_path=False, ax=None):
        """
        静态可视化机器人位置 (Static visualization of robot position)
        
        Args:
            theta1 (float): 第一个关节的角度 (rad)
            theta2 (float): 第二个关节的角度 (rad)
            show_path (bool): 是否显示已记录的路径 (Whether to show recorded path)
            ax (matplotlib.axes.Axes): 可选的绘图轴 (Optional axes for plotting)
        
        Returns:
            tuple: (fig, ax) 图形和轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # 设置绘图参数 (Set plot parameters)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-self.total_reach * 1.1, self.total_reach * 1.1)
        ax.set_ylim(-self.total_reach * 1.1, self.total_reach * 1.1)
        ax.set_xlabel("X 坐标 (X Coordinate)")
        ax.set_ylabel("Y 坐标 (Y Coordinate)")
        ax.set_title(f"2自由度平面机器人 (θ1={np.degrees(theta1):.1f}°, θ2={np.degrees(theta2):.1f}°)")
        ax.grid(True, alpha=0.3)
        
        # 计算关键点位置 (Calculate key point positions)
        (x0, y0), (x1, y1), (x2, y2) = self.forward_kinematics(theta1, theta2)
        
        # 绘制机器人 (Draw robot)
        # 基座 (Base)
        ax.plot(x0, y0, 'ko', markersize=12, label='基座 (Base)')
        
        # 连杆1 (Link 1)
        ax.plot([x0, x1], [y0, y1], 'o-', lw=4, color='skyblue', 
                markersize=10, markerfacecolor='blue', label='连杆1 (Link 1)')
        
        # 连杆2 (Link 2)
        ax.plot([x1, x2], [y1, y2], 'o-', lw=4, color='lightcoral', 
                markersize=10, markerfacecolor='red', label='连杆2 (Link 2)')
        
        # 末端执行器特殊标记 (Special marker for end-effector)
        ax.plot(x2, y2, 's', markersize=12, color='green', 
                markerfacecolor='lightgreen', label='末端执行器 (End-effector)')
        
        # 显示路径 (Show path if requested)
        if show_path and len(self.path_x) > 0:
            ax.plot(self.path_x, self.path_y, ':', lw=2, color='green', 
                   alpha=0.7, label='末端轨迹 (End-effector Path)')
        
        # 显示工作空间边界 (Show workspace boundary)
        circle_outer = plt.Circle((0, 0), self.total_reach, fill=False, 
                                 color='gray', linestyle='--', alpha=0.5)
        circle_inner = plt.Circle((0, 0), abs(self.L1 - self.L2), fill=False, 
                                 color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        
        ax.legend(loc='upper right')
        
        return fig, ax
    
    def visualize_interactive(self):
        """
        交互式可视化，允许通过滑块调节关节角度 (Interactive visualization with sliders for joint angles)
        """
        # 创建图形和子图 (Create figure and subplots)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)  # 为滑块留出空间 (Leave space for sliders)
        
        # 初始角度 (Initial angles)
        theta1_init = 0.0
        theta2_init = 0.0
        
        # 初始绘图 (Initial plot)
        self.visualize_static(theta1_init, theta2_init, ax=ax)
        
        # 创建滑块 (Create sliders)
        ax_theta1 = plt.axes([0.2, 0.1, 0.5, 0.03])
        ax_theta2 = plt.axes([0.2, 0.05, 0.5, 0.03])
        
        slider_theta1 = Slider(ax_theta1, 'θ1 (度)', -180, 180, 
                              valinit=np.degrees(theta1_init), valstep=1)
        slider_theta2 = Slider(ax_theta2, 'θ2 (度)', -180, 180, 
                              valinit=np.degrees(theta2_init), valstep=1)
        
        def update_plot(val):
            """更新绘图的回调函数 (Callback function to update plot)"""
            theta1 = np.radians(slider_theta1.val)
            theta2 = np.radians(slider_theta2.val)
            
            # 清除当前绘图 (Clear current plot)
            ax.clear()
            
            # 重新绘制 (Redraw)
            self.visualize_static(theta1, theta2, ax=ax)
            
            # 显示当前末端执行器位置 (Show current end-effector position)
            (x0, y0), (x1, y1), (x2, y2) = self.forward_kinematics(theta1, theta2)
            ax.text(0.02, 0.98, f'末端位置: ({x2:.3f}, {y2:.3f})', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            fig.canvas.draw()
        
        # 连接滑块事件 (Connect slider events)
        slider_theta1.on_changed(update_plot)
        slider_theta2.on_changed(update_plot)
        
        plt.show()
        
        return fig, ax, slider_theta1, slider_theta2
    
    def animate_motion(self, motion_func, total_frames=200, interval_ms=50, save_path=None):
        """
        动画化机器人运动 (Animate robot motion)
        
        Args:
            motion_func (callable): 运动函数，输入归一化时间t(0-1)，输出(theta1, theta2)
            total_frames (int): 总帧数 (Total number of frames)
            interval_ms (int): 每帧间隔毫秒数 (Interval between frames in ms)
            save_path (str): 保存路径，None表示不保存 (Save path, None means don't save)
        """
        # 初始化图形 (Initialize figure)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-self.total_reach * 1.1, self.total_reach * 1.1)
        ax.set_ylim(-self.total_reach * 1.1, self.total_reach * 1.1)
        ax.set_xlabel("X 坐标 (X Coordinate)")
        ax.set_ylabel("Y 坐标 (Y Coordinate)")
        ax.set_title("2自由度平面机器人动画 (2-DOF Planar Robot Animation)")
        ax.grid(True, alpha=0.3)
        
        # 初始化绘图元素 (Initialize plot elements)
        link1_line, = ax.plot([], [], 'o-', lw=4, color='skyblue', 
                             markersize=10, markerfacecolor='blue', label='连杆1 (Link 1)')
        link2_line, = ax.plot([], [], 'o-', lw=4, color='lightcoral', 
                             markersize=10, markerfacecolor='red', label='连杆2 (Link 2)')
        path_line, = ax.plot([], [], ':', lw=2, color='green', label='末端轨迹 (End-effector Path)')
        base_point, = ax.plot([0], [0], 'ko', markersize=12, label='基座 (Base)')
        
        # 重置路径 (Reset path)
        self.path_x, self.path_y = [], []
        
        def init():
            """初始化函数 (Initialization function)"""
            link1_line.set_data([], [])
            link2_line.set_data([], [])
            path_line.set_data([], [])
            return link1_line, link2_line, path_line
        
        def update(frame):
            """动画更新函数 (Animation update function)"""
            t_normalized = frame / total_frames
            theta1, theta2 = motion_func(t_normalized)
            
            # 计算关键点位置 (Calculate key point positions)
            (x0, y0), (x1, y1), (x2, y2) = self.forward_kinematics(theta1, theta2)
            
            # 更新连杆位置 (Update link positions)
            link1_line.set_data([x0, x1], [y0, y1])
            link2_line.set_data([x1, x2], [y1, y2])
            
            # 更新末端执行器轨迹 (Update end-effector path)
            self.path_x.append(x2)
            self.path_y.append(y2)
            path_line.set_data(self.path_x, self.path_y)
            
            return link1_line, link2_line, path_line
        
        # 创建动画 (Create animation)
        ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                     init_func=init, blit=True, interval=interval_ms)
        
        # 添加图例 (Add legend)
        ax.legend(loc='upper right')
        
        # 保存动画 (Save animation if requested)
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='imagemagick', fps=1000/interval_ms)
            elif save_path.endswith('.mp4'):
                ani.save(save_path, writer='ffmpeg', fps=1000/interval_ms)
            print(f"动画已保存到: {save_path}")
        
        plt.show()
        return ani
    
    def clear_path(self):
        """清除记录的路径 (Clear recorded path)"""
        self.path_x, self.path_y = [], []


def example_motion(t):
    """
    示例运动函数 (Example motion function)
    
    Args:
        t (float): 归一化时间 (0-1) (Normalized time)
    
    Returns:
        tuple: (theta1, theta2) 关节角度 (Joint angles)
    """
    theta1 = t * 2 * np.pi  # theta1 旋转一周 (theta1 rotates a full circle)
    theta2 = (np.pi / 2) * np.sin(t * 4 * np.pi)  # theta2 来回摆动两次 (theta2 oscillates twice)
    return theta1, theta2


# 使用示例 (Usage examples)
if __name__ == "__main__":
    # 创建机器人实例 (Create robot instance)
    robot = PlannerRobot(L1=1.0, L2=0.8)
    
    # 示例1: 静态可视化 (Example 1: Static visualization)
    print("示例1: 静态可视化")
    robot.visualize_static(np.pi/4, np.pi/6)
    plt.show()
    
    # 示例2: 交互式可视化 (Example 2: Interactive visualization)
    print("示例2: 交互式可视化 - 使用滑块调节关节角度")
    robot.visualize_interactive()
    
    # 示例3: 动画 (Example 3: Animation)
    print("示例3: 动画演示")
    robot.animate_motion(example_motion, total_frames=200, interval_ms=50)

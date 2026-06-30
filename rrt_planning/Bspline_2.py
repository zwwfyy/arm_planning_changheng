import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt


def bspline_smooth_path(joint_path, degree=3, num_points=100):
    if isinstance(joint_path, list):
        joint_path = np.vstack(joint_path)
    joint_path = np.asarray(joint_path)

    N, dim = joint_path.shape
    t = np.linspace(0, 1, N)
    t_new = np.linspace(0, 1, num_points)

    smooth_path = np.zeros((num_points, dim))
    for j in range(dim):
        spline = InterpolatedUnivariateSpline(t, joint_path[:, j], k=degree)
        smooth_path[:, j] = spline(t_new)

    result = [smooth_path[i, :] for i in range(num_points)]
    return result


def plot_smooth_joint_paths(smooth_path):
    """
    绘制平滑后的六轴关节角相对于时间的变化曲线（全部在一张图中）
    只显示平滑后的轨迹，不显示原始数据
    """
    smooth_path = np.vstack(smooth_path) if isinstance(smooth_path, list) else np.asarray(smooth_path)

    num_points, dim = smooth_path.shape
    t_smooth = np.linspace(0, 1, num_points)

    # 定义不同关节的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    joint_names = [f'Joint {i + 1}' for i in range(dim)]

    plt.figure(figsize=(12, 8))

    # 只绘制平滑轨迹
    for j in range(dim):
        plt.plot(t_smooth, smooth_path[:, j], '-',
                 color=colors[j], linewidth=1.5,
                 label=f'{joint_names[j]}')

    # 坐标轴标签（Times New Roman）
    plt.xlabel('Time (s)', fontsize=20, fontname='Times New Roman')
    plt.ylabel('Joint Angle (rad)', fontsize=20, fontname='Times New Roman')

    # 坐标轴刻度字体（Times New Roman）
    plt.xticks(fontsize=20, fontname='Times New Roman')
    plt.yticks(fontsize=20, fontname='Times New Roman')

    # 图例字体（Times New Roman）
    plt.legend(
        loc='upper right',  # 放在图框内部右上角
        ncol=2,  # 两列 → 自动三排（6条曲线）
        frameon=True,  # 显示边框
        fancybox=False,  # 方形边框（论文更规范）
        edgecolor='black',  # 黑色边框
        columnspacing=1.2,  # 列间距
        handletextpad=0.4,  # 图例线与文字距离
        borderpad=0.4,  # 图例内部边距
        prop={'family': 'Times New Roman', 'size': 18}
    )

    # 坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(-2.5, 2.5)

    plt.show()


if __name__ == "__main__":
    # 生成随机六轴轨迹
    path = [np.array([np.sin(i / 5), np.cos(i / 5), i * 0.05, np.sin(i / 6), np.cos(i / 6), i * 0.03])
            for i in range(20)]

    # 插值
    smooth_path = bspline_smooth_path(path, degree=3, num_points=200)

    # 绘图（只绘制平滑后的轨迹）
    plot_smooth_joint_paths(smooth_path)
import numpy as np
import pybullet as p

class Visualize:
    @staticmethod
    def plot_points(position, color, size):
        if np.array(position).ndim == 1:
            position = [position]
        if np.array(color).ndim == 1:
            color = [color]
        if len(color) == 1:
            num_points = len(position)
            color = color * num_points

        p.addUserDebugPoints(
            pointPositions=position,
            pointColorsRGB=color,
            pointSize=size,
        )

    @staticmethod
    def plot_lines(position1, position2, color, size):
        if np.array(position1).ndim == 1:
            position1 = [position1]
        if np.array(position2).ndim == 1:
            position2 = [position2]

        assert len(position1) == len(position2)

        # 循环绘制连线
        for start, end in zip(position1, position2):
            p.addUserDebugLine(
                lineFromXYZ=start, lineToXYZ=end, lineColorRGB=color, lineWidth=size
            )

    @staticmethod
    def plot_path(path, point_color, line_color, point_size, line_size):
        Visualize.plot_points(path, point_color, point_size)
        if np.array(path).ndim == 1 or len(path) == 1:
            return
        position1 = path[:-1]
        position2 = path[1:]
        Visualize.plot_lines(position1, position2, line_color, line_size)

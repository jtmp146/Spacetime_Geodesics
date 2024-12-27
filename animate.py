import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

fig_size = 5
fix_axes = True
show_frames = False
start = 0
length = 1000
mark_origin = True
mark_horizon = True
source = "trajectory.txt"

with open(source) as data:
    lines = data.readlines()

point_hist = []

for line in lines:
    points = line.split(",")
    point_step = []
    for point in points:
        x, y, z, s = [float(val) for val in point.split(" ")]
        point_step.append([x, y, z, s])
    point_hist.append(point_step)

def update(num):
    ax.cla()
    ax.set_aspect("equal")
    frame = num+start
    points = point_hist[frame]
    if mark_origin:
        ax.scatter(0, 0, 0, s=100, marker="o", color="black")
    if mark_horizon:
        r_s = 2
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r_s * np.outer(np.cos(u), np.sin(v))
        y = r_s * np.outer(np.sin(u), np.sin(v))
        z = r_s * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha = 0.2, color="r")
    if show_frames:
        print(f"Frame {frame}")
    for i in range(len(points)):
        point = points[i]
        # if i != 12 and i != 4:
        #     ax.scatter(point[0], point[1], point[2], s=point[3], marker="o")
        # else:
        ax.scatter(point[0], point[1], point[2], s=point[3], marker="o")
        if num > 0:
            path = [points[i] for points in point_hist[:frame+1]]
            ax.plot([pos[0] for pos in path], [pos[1] for pos in path], [pos[2] for pos in path])

    if fix_axes:
        ax.set_xlim(-fig_size, fig_size)
        ax.set_ylim(-fig_size, fig_size)
        ax.set_zlim(-fig_size, fig_size)

fig = plt.figure(dpi=100)
ax = fig.add_subplot(projection='3d')

ani = FuncAnimation(fig = fig, func = update, frames = length, interval = 10, repeat = False)

plt.show()
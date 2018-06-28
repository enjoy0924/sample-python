import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import pickle as pkl

# plt.close()  # clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig = plt.figure()
ax1_2 = fig.add_subplot(1, 1, 1, projection='3d')


def v_random(v):
    r = numpy.random.uniform(0, 1, size=9)

    return v * r


R = numpy.random.uniform(-1, 1, size=100)
coords = []


def d3_plane(a, b, c):
    X = numpy.arange(1, 10, 1)
    Y = numpy.arange(1, 10, 1)
    X, Y = numpy.meshgrid(X, Y)  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
    Z = a * X + b * Y + c

    # 构建平面
    surf = ax1_2.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax1_2.set_zlim3d(0, 100)  # 设置z坐标轴
    # fig.colorbar(surf, shrink=0.5, aspect=5)  # 图例

    for i in range(100):
        _X_Y = numpy.random.uniform(0, 10, size=2)
        _Z = a * _X_Y[0] + b * _X_Y[1] + c + R[i]

        ax1_2.scatter(_X_Y[0], _X_Y[1], _Z, c='g')

        coords.append([_X_Y[0], _X_Y[1], _Z])
        with open('3d-data.pkl', 'wb') as f:
            pkl.dump(coords, f)


d3_plane(1, 8, 10)

plt.show()

import pickle
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = 3
data_file = '2d-data.pkl'
if dim > 2:
    data_file = '3d-data.pkl'
fid = open(data_file, 'rb')
data = pickle.load(fid)
fid.close()

w_array = numpy.empty(dim - 1)
for d in range(dim - 1):
    w_array[d] = numpy.random.random()

w_dim = [w_array]

w = numpy.array(w_dim)
numpy.reshape(w, (1, (dim - 1)))

bias = numpy.random.random()

learning_rate = 0.05

d2_array = numpy.array(data)


def loss(_x, _y, _w, _b):
    out = numpy.dot(_w, _x)
    return (out + _b - _y) ** 2


def gradient_descent(_x, _y, _b, _ref):
    _w = w.copy()
    shape_x = _w.shape[0]
    shape_y = _w.shape[1]
    for _shape_x in range(shape_x):
        for _shape_y in range(shape_y):
            _w[_shape_x, _shape_y] += learning_rate
            _loss = loss(_x, _y, _w, _b)
            # 计算完之后，恢复至原矩阵
            _w[_shape_x, _shape_y] = w[_shape_x, _shape_y]

            diff = _loss - _ref
            if numpy.abs(diff) > 0.1:  # 误差在变大
                if diff > 0:  # 正向变大
                    w[_shape_x, _shape_y] -= learning_rate
                else:    # 反向变大
                    w[_shape_x, _shape_y] += learning_rate

    _loss = loss(_x, _y, _w, (_b + learning_rate))
    diff = _loss - _ref
    if numpy.abs(diff) > 0.1:
        if diff > 0:
            return _b - learning_rate
        else:
            return _b + learning_rate

    return _b


def d2_view(_w, _b):
    plt.clf()
    plt.scatter(d2_array[:, 0], d2_array[:, 1], c='red')
    canvas_x_range = numpy.linspace(1, 10)
    plt.plot(canvas_x_range, _w[0][0] * canvas_x_range + _b)


def d3_view(_w, _b):
    ax = plt.subplot(111, projection='3d')

    ax.scatter(d2_array[:, 0], d2_array[:, 1], d2_array[:, 2], c='green')

    X = numpy.arange(1, 10, 1)
    Y = numpy.arange(1, 10, 1)
    X, Y = numpy.meshgrid(X, Y)  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度

    Z = _w[0][0] * X + _w[0][1] * Y + _b
    # 构建平面
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    ax.set_zlim3d(0, 100)  # 设置z坐标轴
    # fig.colorbar(surf, shrink=0.5, aspect=5)  # 图例

    # for i in range(100):
    #     _X_Y = numpy.random.uniform(0, 10, size=2)
    #     _Z = a * _X_Y[0] + b * _X_Y[1] + c + R[i]
    #
    #     ax1_2.scatter(_X_Y[0], _X_Y[1], _Z, c='g')
    #
    #     coords.append([_X_Y[0], _X_Y[1], _Z])
    #     with open('3d-data.pkl', 'wb') as f:
    #         pkl.dump(coords, f)


for i in range(1000):
    for dot in d2_array:
        length = len(dot)
        x = dot[0:(length - 1)]
        y = dot[length - 1]
        loss1 = loss(x, y, w, bias)
        bias = gradient_descent(x, y, bias, loss1)
        print("%s, %f" % (str(w), bias))
        if dim > 2:
            d3_view(w, bias)
        else:
            d2_view(w, bias)
        plt.pause(0.5)

plt.show()

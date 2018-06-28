import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def onclick(event):
    ix, iy = event.xdata, event.ydata
    data = [ix, iy]
    global coords
    coords.append(data)
    print('x = %f, y = %f' % (ix, iy))
    with open('2d-data.pkl', 'wb') as f:
        pkl.dump(coords, f)
    ax.scatter(ix, iy, c='red')
    fig.canvas.draw()
    return


coords = []

# x = np.linspace(1, 10)
# w = 2
# b = 5
# y = w * x + b

plt.xlim(0.0, 25.0)
plt.ylim(0.0, 25.0)

fig = plt.figure(1)
ax = fig.add_subplot(111)
# ax.plot(x, y)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

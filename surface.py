import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf

def plot_surface(x_size, y_size, Z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(x_size)
	y = np.arange(y_size)
	X, Y = np.meshgrid(x, y)

	ax.plot_surface(X, Y, Z)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()


if __name__ == '__main__':

	# Compute the gradient of a gaussian filter

	a = np.zeros((21, 21))
	a[10][10] = 1

	g = gaussian_filter(a, 2)

	#plot_surface(31, 31, g)

	# image to convolve
	x = tf.placeholder(tf.float32, [1, None, None, 1])

	# filter to use
	c = tf.placeholder(tf.float32, [None, None, 1, 1])

	# convolution operation
	op = tf.nn.convolution(x, c, 'SAME')

	f = np.array([[-1, 1]])

	with tf.Session() as sess:

		out = sess.run(op, {x: np.reshape(g, [1] + list(g.shape) + [1]), c: np.reshape(f, list(f.shape) + [1, 1])})

		plot_surface(a.shape[0], a.shape[1], out.squeeze())

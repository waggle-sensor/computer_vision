import numpy as np
cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def grad_hist(int[:,::1] tf, int[:,::1] tc, float[:,::1] rf, float[:,::1] rc,
			  int cell_x, int cell_y, int n_bins):
	"""
	:param tf:
	:param tc:
	:param rf:
	:param rc:
	:param cell_x:
	:param cell_y:
	:param n_bins:
	:return:
	Compute Histogram of Oriented Gradients give rounded thetas and interpolated rs
	"""
	cdef:
		int xmax = tf.shape[0]
		int ymax = tf.shape[1]
		int hx, hy
		int x, y, i, j, x_s, y_s
		float[:,:,::1] h

	hx = xmax // cell_x
	hy = ymax // cell_y
	h = np.zeros((hx, hy, n_bins), dtype=np.float32)
	for x in range(hx):
		for y in range(hy):
			for i in range(cell_x):
				for j in range(cell_y):
					x_s = cell_x*x + i
					y_s = cell_y*y + j
					h[x, y, tf[x_s,y_s]] += rf[x_s,y_s]
					h[x, y, tc[x_s,y_s]] += rc[x_s,y_s]

	return np.asarray(h)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sum_pool_frame(int[:,:,::1] frame, int cell_x, int cell_y):
	"""

	:param frame:
	:param cell_x:
	:param cell_y:
	:return:
	Does pooling over cell_x by cell_y regions in a frame
	"""
	cdef:
		int xmax = frame.shape[0]
		int ymax = frame.shape[1]
		int chans = frame.shape[2]
		int hx, hy
		int x, y, i, j, x_s, y_s
		int[:,:,::1] frame_p

	hx = xmax // cell_x
	hy = ymax // cell_y
	frame_p = np.zeros((hx, hy, chans), dtype=np.int32)
	for x in range(hx):
		for y in range(hy):
			for i in range(cell_x):
				for j in range(cell_y):
					x_s = cell_x*x + i
					y_s = cell_y*y + j
					frame_p[x, y, 0] += frame[x_s,y_s, 0]
					frame_p[x, y, 1] += frame[x_s,y_s, 1]
					frame_p[x, y, 2] += frame[x_s,y_s, 2]
	return np.asarray(frame_p)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sum_pool_grad(float[:,::1] grad, int cell_x, int cell_y):
	"""
	Does pooling over cell_x by cell_y regions of gradient magnitude. Basically the same as sum_pool_frame,
	except takes in floats.
	:param grad:
	:param cell_x:
	:param cell_y:
	:return:
	"""
	cdef:
		int xmax = grad.shape[0]
		int ymax = grad.shape[1]
		int chans = grad.shape[2]
		int hx, hy
		int x, y, i, j, x_s, y_s
		float[:, ::1] grad_p

	hx = xmax // cell_x
	hy = ymax // cell_y
	grad_p = np.zeros((hx, hy), dtype=np.float32)
	for x in range(hx):
		for y in range(hy):
			for i in range(cell_x):
				for j in range(cell_y):
					x_s = cell_x*x + i
					y_s = cell_y*y + j
					grad_p[x, y] += grad[x_s,y_s]
	return np.asarray(grad_p)


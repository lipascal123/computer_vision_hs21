import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	# TODO
	# return the least-squares solution
	# you can use np.linalg.lstsq
	X = np.vstack((x,np.ones(x.shape[0]))).T
	k, b = np.linalg.lstsq(X, y, rcond=None)[0]
	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# TODO
	# compute the number of inliers and a mask that denotes the indices of inliers
	num = 0
	mask = np.zeros(x.shape, dtype=bool)

	# compute distance
	y_hat = k*x + b
	diff = abs(y_hat - y)
	# compute inliers
	num = diff[diff<thres_dist]
	num = len(num)
	# compute mask
	idx = np.argwhere(diff<thres_dist)
	mask[idx] = 1

	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# TODO
	#! Note: the first 50 datapoints are outliers
	best_inliers = 0
	k_ransac = None
	b_ransac = None
	# ransac
	# stack x and y for sampling
	# x: first column, y: second column
	data = np.vstack((x, y)).T
	for i in range(iter):
		# sample
		smpl = random.sample(data.tolist(), num_subset)
		# convert to array
		smpl = np.array(smpl)
		x_smpl = smpl[:,0]
		y_smpl = smpl[:,1]
		# comput least square solution
		X_smpl = np.vstack((x_smpl, np.ones(num_subset))).T
		k, b = np.linalg.lstsq(X_smpl, y_smpl, rcond=None)[0]
		# compute inlier
		num_inlr, mask = num_inlier(x, y, k, b, num_subset, thres_dist)

		# udpate model
		if num_inlr > best_inliers:
			k_ransac = k
			b_ransac = b
			inlier_mask = mask
			best_inliers = num_inlr
			# print("k_ransac", k_ransac)
			# print("b_ransac", b_ransac)
			# print("best_inliers", best_inliers)

	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()

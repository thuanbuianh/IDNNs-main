'Calculation of the full plug-in distribuation'

import numpy as np
import multiprocessing
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


def calc_entropy_for_specipic_t(current_ts, px_i):
	"""Calc entropy for specipic t"""
	b2 = np.ascontiguousarray(current_ts).view(
		np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
	unique_array, unique_inverse_t, unique_counts = \
		np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
	p_tgx = unique_counts / float(sum(unique_counts))	# P(t|x)
	p_tgx = np.asarray(p_tgx, dtype=np.float64).T
	H_tgx = px_i * (-np.sum(p_tgx * np.log2(p_tgx)))	# H(t|x) = -P(x)*sum(P(t|x) * log(P(t|x)))
	return H_tgx


def calc_condtion_entropy(px, t_data, unique_inverse_x):
	# Condition entropy of t given x
	# H_tgx_array = np.array(
	# 	Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
	# 	                           for i in range(px.shape[0])))

	H_tgx_array = []
	for i in range(px.shape[0]):
		H_tgx_array.append(calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]))

	H_tgx_array = np.array(H_tgx_array)

	H_TgX = np.sum(H_tgx_array)

	return H_TgX


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array, x, ws_layer):
	"""Calculate the MI based on binning of the data"""
	# H2 = -np.sum(ps2 * np.log2(ps2))	# H(T)
	# H2X = calc_condtion_entropy(px, data, unique_inverse_x)	# H(T|X)
	# H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y) # H(T|Y)
	# IY = H2 - H2Y	# I(T,X) == I(X,T) 
	# IX = H2 - H2X	# I(T,Y) == I(Y,T)

	# Tính MMI dựa theo công thức (7)
	B = np.zeros(x.shape[0])
	As = [np.zeros(x.shape[0]) for i in range(ws_layer.shape[3])]
	for i in range(x.shape[0]):
		for j in range(x.shape[0]):
			B[i][j] = rbf_kernel(x[i], x[j])
			for k in range(len(As)):
				As[k][i][j] = rbf_kernel(ws_layer[i][:][:][k], ws_layer[j][:][:][k])
	# Viết lại công thức (7): I(B; {A1; A2; · · · ; AC}) = S(B) + S(As_scaled) - S(AsB_scaled)
	As_temp = As[0]
	for i in range(1, len(As)):
		As_temp = np.multiply(As_temp, As[i])	
	As_scaled = As_temp / np.matrix.trace(As_temp)

	AsB_temp = np.multiply(As_temp, B)
	AsB_scaled = AsB_temp / np.matrix.trace(AsB_temp)

	# Tính các đại lượng S_alpha theo công thức (2)
	alpha = 1.01
	B_ev = np.linalg.eig(B) # B's eigenvalues 
	S_B = np.log2(np.sum(B_ev**alpha)) / (1-alpha)

	As_scaled_ev = np.linalg.eig(As_scaled)
	S_As_scaled = np.log2(np.sum(As_scaled_ev**alpha)) / (1-alpha)

	AsB_scaled_ev = np.linalg.eig(AsB_scaled)
	S_AsB_scaled = np.log2(np.sum(AsB_scaled_ev**alpha)) / (1-alpha)

	IX = S_B + S_As_scaled - S_AsB_scaled
	IY = IX
	return IX, IY


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
	"""Calculate the p(x|T) and p(y|T)"""
	indexs = unique_inverse == t_index
	p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
	unique_array_internal, unique_counts_internal = \
		np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
	indexes_x = np.where(np.in1d(b1, b[indexs]))
	p_x_ts = np.zeros(len_unique_a)
	p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
	return p_x_ts, p_y_ts

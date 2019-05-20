import numpy as np
import torch
from maskrcnn_benchmark.layers.roi_pool import ROIPool
from maskrcnn_benchmark.layers.roi_align import ROIAlign

if __name__ == '__main__':
	pool_size = (2,2)
	spatial_scale = 1.0
	sampling_ratio = 0
	pooler = ROIPool(pool_size, spatial_scale=spatial_scale)
	pooler2 = ROIAlign(pool_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)

	N = 1
	C = 3
	W = 6
	H = 6
	x = np.arange(N*C*W*H).reshape((N,C,W,H)).astype(np.float32)
	tx = torch.tensor(x, requires_grad=True, device='cuda')
	tx2 = torch.tensor(x, requires_grad=True, device='cuda')

	rois = np.array([
		[0, 1, 1, W-2, H-2]  # batch_ind, start_x, start_y, end_x, end_y
	], dtype=np.float32)
	trois = torch.tensor(rois, device='cuda')

	out = pooler(tx, trois)
	loss = (out ** 2).sum()
	loss.backward()

	trois[:,-2:] += 1
	out2 = pooler2(tx2, trois)
	loss2 = (out2 ** 2).sum()
	loss2.backward()

	print(out)
	print(out2)
	
	# print(tx.grad)

"""
author:	wamawama
email:	wmy19970215@gmail.com

"""

# from wama.utils import *
import numpy as np
import torch
from copy import deepcopy

def devideBySum(image):
	return image / np.sum(image)

def tensor2numpy(tensor):
	return tensor.data.cpu().numpy()

def draw_line(mask, y_left = 6/6, y_right = 3/6, line_width=0.01, line_value=0.4, gpu_device = None):
	"""

	:param mask: 2D with 1 channel, range of pixel value: 0~1
	:param y_left: from 0 to 1 左侧点所在位置，相对坐标
	:param y_right: from 0 to 1 右侧点所在位置，相对坐标
	:param line_width: 线宽，from 0 to 1， 是y轴的长度的百分之n
	:param line_value: xia
	:return:
	"""

	# 首先，提取直线y = ax+b的斜率w和截距b
	b = y_left * mask.shape[0]
	a = ((y_right - y_left)* mask.shape[0]) / mask.shape[1]

	# 构建索引矩阵，dim0， dim1
	index_array_dim0 = np.stack([list(range(mask.shape[0]))]*mask.shape[1],1)  # y轴
	index_array_dim1 = np.stack([list(range(mask.shape[1]))]*mask.shape[0],1).T  # x轴
	if gpu_device is not None:
		index_array_dim0 = torch.tensor(index_array_dim0).to(gpu_device)
		index_array_dim1 = torch.tensor(index_array_dim1).to(gpu_device)
	# show2D(index_array_dim1)

	# 计算所有点到直线的距离
	if gpu_device is not None:
		dis_array = torch.abs((a * index_array_dim1 - index_array_dim0 + b) / np.sqrt(a ** 2 + 1))
	else:
		dis_array = np.abs((a*index_array_dim1 - index_array_dim0 + b) / np.sqrt(a**2 + 1))
	# show2D(dis_array)

	# 符合条件的索引（mask）
	if gpu_device is not None:
		line_mask = tensor2numpy((dis_array <= line_width*mask.shape[0])).astype(np.float)
	else:
		line_mask = (dis_array <= line_width*mask.shape[0]).astype(np.float)
	# show2D(line_mask)

	# 赋值mask
	mask[line_mask==1] = line_value

	return mask

def draw_attention_map(mask, attention_map, line_width=0.007, gpu_device = None):
	"""

	:param mask:
	:param attention_map: shape of [ token_num, token_num], and sum(axis = 1) = 1.0
	:param line_width:
	:return:
	"""

	# 读取token数
	token_num = attention_map.shape[0]

	# 记录所有需要勾画的线段，格式：[ 注意力值，左坐标，右坐标]
	lines_list = []
	lines_list_per_token = [[] for _ in range(token_num)]
	mask_per_token_list = [np.zeros(mask.shape) for _ in range(token_num)]

	for query_token_index in range(token_num):
		attention_vector = list(attention_map[query_token_index])
		attention_vector_up2down = deepcopy(attention_vector)
		attention_vector_up2down.sort()
		for attention_value in attention_vector_up2down:
			key_token_index = attention_vector.index(attention_value)
			lines_list.append([attention_vector[key_token_index], key_token_index/(token_num-1), query_token_index/(token_num-1)])
			lines_list_per_token[query_token_index].append([attention_vector[key_token_index], key_token_index/(token_num-1), query_token_index/(token_num-1)])

	# 总图，所有token画在一起。逐行勾画(注意，要从注意力小勾画到大）
	lines_list_attention_value = [i[0] for i in lines_list]
	lines_list_attention_value_up2down =  deepcopy(lines_list_attention_value)
	lines_list_attention_value_up2down.sort()
	for _attention_value in lines_list_attention_value_up2down:
		line_index = lines_list_attention_value.index(_attention_value)
		mask = draw_line(mask, lines_list[line_index][1],
						 lines_list[line_index][2],
						 line_width, lines_list[line_index][0],
						 gpu_device = gpu_device)

	# 画分图
	for query_token_index in range(token_num):
		for line in lines_list_per_token[query_token_index]:
			mask_per_token_list[query_token_index] = draw_line(mask_per_token_list[query_token_index], line[1],line[2],line_width, line[0],gpu_device=gpu_device)

	return [mask,mask_per_token_list]

def draw_attention_map_multihead_multilayer(mask, attention_map_mhml, line_width=0.007, gpu_device = None):
	"""

	:param mask: shape of [batchsize, head, token, token] list， from layer_1 to layer_n
	:param attention_map_multihead:
	:param line_width:
\
	e.g.
	task = 6
	attention_map = np.stack([devideBySum(np.random.rand(task)) for _ in range(task)], 0)
	attention_map_mh = np.stack([np.stack([devideBySum(np.random.rand(task)) for _ in range(task)], 0) for _ in range(3)], 0)
	attention_map_mh_batch = np.stack([attention_map_mh]*4, 0)
	attention_map_mhml = [attention_map_mh_batch]*3 # 3 layers, with 3 head per layer

	"""

	mask_per_layer_list = [[] for _ in range(len(attention_map_mhml))]
	for layer_index, per_layer_list in enumerate(mask_per_layer_list):
		# layer_index, per_layer_list = [0, mask_per_layer_list[0]]

		attention_map_mh_batch = attention_map_mhml[layer_index]
		# 每个样本单独处理
		mask_per_case_list = [[] for _ in range(attention_map_mh_batch.shape[0])]
		for case_index in range(attention_map_mh_batch.shape[0]):
			# case_index = 0
			print('doing with layer ',layer_index+1,'/',len(mask_per_layer_list),' case ', case_index+1, '/', attention_map_mh_batch.shape[0])
			attention_map_mh_case = attention_map_mh_batch[case_index]
			mask_per_head_list = [draw_attention_map(deepcopy(mask), attention_map_mh_case[case_index], line_width=line_width,gpu_device=gpu_device)
								  for case_index in range(attention_map_mh_case.shape[0])]
			mask_per_case_list[case_index] = mask_per_head_list

		# 添加到每个layer的list
		mask_per_layer_list[layer_index] = mask_per_case_list

	return mask_per_layer_list

def make_attention_map_mh(head_num, task_num):
		"""

		:param head_num: head数量
		:param task_num:
		:return:
		"""
		return np.stack([np.stack([devideBySum(np.random.rand(task_num)) for _ in range(task_num)], 0) for _ in range(head_num)], 0)

















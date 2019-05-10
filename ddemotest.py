# extra_layers = [1,2,3,4,5,6,7,8]
#
# for k, v in enumerate(extra_layers[1::2],3):
#     print(str(k)+"*****"+str(v))
#
# print(extra_layers[1::3])

# sources = list()
# loc = list()
# conf = list()
#
# for (x, l, c) in zip(sources, loc, conf):
#     print("done")
#     pass
# from itertools import product
# product(range(2),repeat=3)
# print(a)
#
# f = 3
# for i, j in product(range(f), repeat=2):
#     print(str(i)+"****"+str(j))

# asqw = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# for ar in asqw[1]:
#     print(ar)

# import torch
# # a = torch.randn(3,3)
# # print(a)
# # best_prior_overlap, best_prior_idx = a.max(1,keepdim=True)
# # print(str(best_prior_overlap)+"*******"+str(best_prior_idx))
# box_a = torch.rand(10,4)
# box_b = torch.rand(20,4)
# A = box_a.size(0)
# B = box_b.size(0)
# x = box_a[:, 2:]
# xx = box_a[:, 2:].unsqueeze(1)
# xxx = box_a[:, 2:].unsqueeze(1).expand(A, B, 2)
# max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
# min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
# inter = torch.clamp((max_xy - min_xy), min=0)
# overlaps = inter[:, :, 0] * inter[:, :, 1]
# best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
# print(str(best_prior_overlap)+"*******"+str(best_prior_idx))
# # [1,num_priors] best ground truth for each prior
# best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
# print(str(best_truth_overlap)+"*******"+str(best_truth_idx))
# best_truth_idx.squeeze_(0)
# best_truth_overlap.squeeze_(0)
# best_prior_idx.squeeze_(1)
# best_prior_overlap.squeeze_(1)
# best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#
# for j in range(best_prior_idx.size(0)):
#     best_truth_idx[best_prior_idx[j]] = j
#
# aaa = torch.randn(10,4)
# yy = aaa[best_truth_idx]
# print(yy)
import torch
aa = torch.randn(10,8073,4)
a  = torch.tensor([-1,1,2,3,4,0,-2,-3,-5,1])
b = a>0
num_pos = b.sum(dim=0, keepdim=True)
c = b.unsqueeze(1)
d = c.unsqueeze(1).expand_as(aa)
# print()
print(num_pos)
#
#
# print('done')
# a = torch.nn.Softmax(dim=-1)
# input1 = torch.randn(10)
# output1 = a(input1)
# print(input1)
# print(output1)
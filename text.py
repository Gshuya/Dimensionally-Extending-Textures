import torch
import torch.nn as nn

#base = nn.Parameter(nn.init.constant_(torch.empty(1, 1, 1, 128), 1))
# broadcast (constant per pixel and per sample)
# print base
#print(base.shape)
#base = base.repeat(16, 128, 128, 1)
#print(base.shape[0])
#print(transformations.is_leaf)
#transformations = transformations.requires_grad_(True)
#print(transformations.is_leaf)
#print(transformations.device)
A = torch.tensor([1,2])
print(A.expand(4,1,2,2))

"""
1. To fix the inference FileExistsError
2. To fix the train.py RuntimeError
"""
import torch


tensor1 = torch.randn(2,3,4)
tensor2 = torch.randn(2,4,5)

mat = torch.matmul(tensor1, tensor2)


print(tensor1[:])

print(tensor2[:])

print(mat[:])

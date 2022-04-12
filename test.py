import torch


tensor1 = torch.randn(2,3,4)
tensor2 = torch.randn(3,4)

# # mat = torch.matmul(tensor1, tensor2)


# print(tensor1)
# print(tensor2)


# print(tensor1*tensor2)



arr1 = torch.tensor([1,2,3,4,5])

arr2 = torch.tensor([1,2,3,6,7])

print(arr1.eq(arr2))
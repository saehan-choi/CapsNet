import timm
import torch

avail_pretrained_models = timm.list_models(pretrained=True)

# print(avail_pretrained_models)

for i in avail_pretrained_models:
    print(i)
import torch
from model import *
import sys
import os
model_path = sys.argv[1]
out_path =  'quantized'+model_path
model_dict = torch.load(model_path)
for key, value in model_dict.items():
    model_dict[key] = value.type(torch.cuda.HalfTensor)
torch.save(model_dict, out_path)

from transformer import Transformer
#import math
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from typing import Optional, Any


model = Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)

src = torch.rand(10, 32, 512)  # (sequence length, batch size, d_model)
tgt = torch.rand(20, 32, 512)

# out = model(src, tgt)
# print(out.shape)
# print(out)

output = model(src, tgt)
logits = output[-1]  # last token
predicted = torch.argmax(logits, dim=-1)
print(predicted)
print(src.shape)
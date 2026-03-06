import torch
import torch.nn as nn
#from torch.nn import Transformer
import math
from transformer import Transformer

# Positional Encoding for time steps
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0)]

# Time Series Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_len, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.output_linear = nn.Linear(d_model, 1)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=False  # expects (seq_len, batch, features)
        )
        self.pred_len = pred_len

    def forward(self, src):
        # src: (batch, src_len, input_dim)
        src = src.permute(1, 0, 2)  # (src_len, batch, input_dim)
        src = self.input_linear(src)  # (src_len, batch, d_model)
        src = self.pos_encoder(src)

        # Create zero decoder input for prediction
        tgt = torch.zeros(self.pred_len, src.size(1), src.size(2)).to(src.device)
        tgt = self.pos_encoder(tgt)

        out = self.transformer(src, tgt)  # (pred_len, batch, d_model)
        return self.output_linear(out).permute(1, 0, 2)  # -> (batch, pred_len, 1)


# Example data
batch_size = 16
past_len = 30
future_len = 5

x = torch.randn(batch_size, past_len, 1)  # Simulated input

# Model
model = TimeSeriesTransformer(
    input_dim=1, d_model=64, nhead=4, num_layers=2, pred_len=future_len
)

y_pred = model(x)  # Output shape: (batch_size, 5, 1)
print(y_pred.shape)  # torch.Size([16, 5, 1])


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

y_true = torch.randn(batch_size, future_len, 1)  # Simulated ground truth

# Suppose y_true is the true next 5 values (shape: [batch, 5, 1])
loss = criterion(y_pred, y_true)
loss.backward()
optimizer.step()


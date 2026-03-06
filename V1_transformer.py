import torch
import torch.nn as nn
import transformer  # your local transformer module

X_tensor = torch.load(r'C:\Users\arezaei\OneDrive - Texas A&M University-Corpus Christi\NTP me\NTP\Coding and Data\Data V3\X_tensor.pt')
y_tensor = torch.load(r'C:\Users\arezaei\OneDrive - Texas A&M University-Corpus Christi\NTP me\NTP\Coding and Data\Data V3\y_tensor.pt')

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = transformer.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = transformer.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # required shape: (seq_len, batch, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x[-1]  # use the last token's output (shape: batch, d_model)
        out = self.regressor(x).squeeze(-1)  # (batch,)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # add positional encoding
        return self.dropout(x)

print(X_tensor.shape, y_tensor.shape)

model = TimeSeriesTransformer(input_size=X_tensor.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
batch_size = 64
print("Let's start ...")
for epoch in range(epochs):
    print("Epoch: ", epoch)
    model.train()
    perm = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        #print('i: ', i)
        idx = perm[i:i+batch_size]
        X_batch = X_tensor[idx]
        y_batch = y_tensor[idx]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

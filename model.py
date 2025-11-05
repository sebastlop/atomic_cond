import torch
import numpy as np


class FlexibleNN(torch.nn.Module):
    def __init__(self, input_dim, layers, activations, dropout_rate):
        super().__init__()
        modules = []
        dim_in = input_dim
        self.lin1 = torch.nn.Linear(input_dim,128)
        self.lin2 = torch.nn.Linear(128,128)
        self.lin3 = torch.nn.Linear(128,128)
        self.lin4 = torch.nn.Linear(128,128)
        self.lin5 = torch.nn.Linear(128,1)

        self.act = torch.nn.ELU()

        self.dp = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.dp(x)
        x = self.act(self.lin2(x))
        x = self.dp(x)
        x = self.act(self.lin3(x))
        x = self.dp(x)
        x = self.act(self.lin4(x))
        x = self.dp(x)
        return self.lin5(x)
    

def predict_conditional(model, one_hot_encoder, theory: str, Z_proj, Z_target, values):
    # assertions

    # --- input tensor creation
    enc_array = one_hot_encoder.transform([[theory]])
    input_tensor = torch.hstack([
                  torch.FloatTensor(np.ones_like(values).reshape(-1,1)*Z_proj),
                  torch.FloatTensor(np.ones_like(values).reshape(-1,1)*Z_target),
                  torch.FloatTensor(values.reshape(-1,1)),
                  torch.FloatTensor(np.repeat(enc_array.reshape(1,-1), values.shape[0], axis=0))
                  ]).requires_grad_(False)

    with torch.no_grad():
        predicted = model(input_tensor)
    
    return 10**predicted.numpy()


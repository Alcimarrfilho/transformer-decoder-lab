import torch
import torch.nn as nn
import math

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # Definido as matrizes lineares
        # Q (Query) vem do Decoder | K (Key) e V (Value) vêm do Encoder
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def cross_attention(self, x, encoder_out):
        # x é a entrada do decoder, encoder_out é a saída da aula anterior
        q = self.wq(x)
        k = self.wk(encoder_out)
        v = self.wv(encoder_out)

        # Cálculo do Scaled Dot-Product Attention
        # Nota: Sem máscara causal aqui, pois podemos ver todo o Encoder!
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x, encoder_out):
        return self.cross_attention(x, encoder_out)
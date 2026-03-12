import torch
import math

def create_causal_mask(seq_len):
    """
    Tarefa 1: Implementando a Máscara Causal (Look-Ahead Mask)
    Fundamento: Garante que a palavra na posição 'i' não atenda à posição 'i+1'.
    """
    # Cria uma matriz quadrada de 1s na parte superior (acima da diagonal)
    # triu = triangular upper
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    
    # Onde for 1, substituímos por -inf (infinito negativo)
    # Onde for 0, mantemos 0.0
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

    def forward(self, q, k, v, mask=None):
        # Cálculo do Scaled Dot-Product Attention (Aula 2)
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # Injeção da máscara causal se fornecida (Tarefa 1.2)
        if mask is not None:
            scaled_attention_logits += mask

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        return torch.matmul(attention_weights, v)
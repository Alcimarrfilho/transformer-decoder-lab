import torch
from decoder import DecoderLayer

def generate_next_token(contexto, encoder_out, decoder):
    # Mock de predição
    # Simula um vocabulário de 10.000 tokens
    vocab_size = 10000
    logits = torch.randn(1, vocab_size)
    probs = torch.softmax(logits, dim=-1)
    return probs

def run_lab():
    vocab_teste = ["<PAD>", "<START>", "<EOS>", "O", "Decoder", "funciona"]
    encoder_out = torch.randn(1, 5, 512)
    decoder = DecoderLayer()
    
    # token de início
    contexto = ["<START>"]
    
    print("--- Gerando Texto (Tarefa 3) ---")

    # O Loop while
    while len(contexto) < 15:
        probs = generate_next_token(contexto, encoder_out, decoder)
        
        # Seleciona a palavra com maior probabilidade (Argmax)
        idx = torch.argmax(probs).item()
        
        # se o idx for alto, sorteamos do nosso vocab_teste
        palavra = vocab_teste[idx % len(vocab_teste)]
        
        print(f"Token: {palavra}")
        
        if palavra == "<EOS>":
            break
            
        contexto.append(palavra)

    print("\nFRASE FINAL:", " ".join(contexto))

if __name__ == "__main__":
    run_lab()

def executar_geracao():
    decoder = DecoderLayer(d_model=512)
    vocab = ["<PAD>", "<START>", "<EOS>", "Transformers", "são", "poderosos"]
    encoder_out = torch.randn(1, 5, 512)
    frase = ["<START>"]
    
    print("--- Gerando ---")
    while len(frase) < 10:
        x = torch.randn(1, len(frase), 512)
        saida = decoder(x, encoder_out) # Chama seu forward
        
        # Simula escolha da próxima palavra
        idx = torch.randint(2, len(vocab), (1,)).item()
        palavra = vocab[idx]
        print(f"Token: {palavra}")

        if palavra == "<EOS>": break
        frase.append(palavra)

    print("\nResultado:", " ".join(frase))

if __name__ == "__main__":
    executar_geracao()
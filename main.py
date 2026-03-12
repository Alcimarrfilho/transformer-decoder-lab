import torch
from decoder import DecoderLayer


# Função separada pedida no laboratório
def generate_next_token(current_sequence, encoder_out, decoder, vocab):

    # Simula embeddings da sequência atual
    x = torch.randn(1, len(current_sequence), 512)

    # Passa pelo decoder
    output = decoder(x, encoder_out)

    # Simula projeção para vocabulário
    logits = torch.randn(len(vocab))

    # Converte em probabilidades
    probs = torch.softmax(logits, dim=0)

    # Escolhe o token com maior probabilidade
    idx = torch.argmax(probs).item()

    return vocab[idx]


def executar_geracao():

    decoder = DecoderLayer(d_model=512)

    vocab = ["<PAD>", "<START>", "<EOS>", "Transformers", "são", "poderosos"]

    # Saída simulada do encoder
    encoder_out = torch.randn(1, 5, 512)

    frase = ["<START>"]

    print("--- Gerando sequência ---")

    # LOOP AUTO-REGRESSIVO
    while len(frase) < 10:

        token = generate_next_token(frase, encoder_out, decoder, vocab)

        print("Token gerado:", token)

        if token == "<EOS>":
            break

        frase.append(token)

    print("\nResultado final:")
    print(" ".join(frase))


if __name__ == "__main__":
    executar_geracao()
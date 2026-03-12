 Lab P1-03: Transformer Decoder, este repositório contém a implementação dos componentes fundamentais do **Decoder** de um Transformer, focando em Máscara Causal, Cross-Attention e o loop de geração auto-regressiva, utilizando a biblioteca **PyTorch**.


### 1. Como Executar:

você precisará da biblioteca **PyTorch**. Caso não tenha, abra o terminal e digite:
pip install torch

### 2. Execução
No terminal (CMD), dentro da pasta do projeto, digite:
python main.py

COPONENTES IMPLEMENTADOS:

### 1. Máscara Causal (Look-Ahead Mask)
Implementada no arquivo multi_head_attention.py. Garante que o modelo não tenha acesso às palavras futuras durante o processamento.

### 2. Cross-Attention
Implementada no arquivo decoder.py. Permite que o Decoder consulte as informações processadas pelo Encoder. As **Queries (Q)** vêm do Decoder, enquanto **Keys (K)** e **Values (V)** vêm do Encoder.

### 3. Loop de Inferência Auto-Regressiva
Localizado no main.py. O modelo gera um token por vez até encontrar o sinal de parada <EOS>.



 Nota sobre o desenvolvimento:
Este projeto foi desenvolvido com o suporte da IA **Gemini**. A ferramenta foi utilizada exclusivamente como um par colaborativo para o esclarecimento de **dúvidas pontuais**, incluindo:
* Explicação da lógica matemática das máscaras.
* Refinamento da estrutura da classe `DecoderLayer`.
* Suporte em erros de ambiente e comandos de Git.

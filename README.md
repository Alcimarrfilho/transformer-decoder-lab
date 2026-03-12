Lab P1-03: Transformer Decoder - Máscara Causal e InferênciaEste repositório contém a implementação dos componentes fundamentais do Decoder de um Transformer, focando em Máscara Causal, Cross-Attention e o loop de geração auto-regressiva, utilizando a biblioteca PyTorch.Como Executar1. Baixar o ProjetoNo topo desta página, clique no botão verde "<> Code" e selecione "Download ZIP" ou clone o repositório via terminal:git clone https://github.com/SeuUsuario/transformer-decoder-lab.git2. Instalar DependênciasPara este laboratório, você precisará da biblioteca PyTorch. Abra o terminal e digite:DOSpip install torch
3. Execução por Sistema OperacionalSistema OperacionalComando para RodarWindowspython main.pyLinux / macOSpython3 main.pyComponentes Implementados1. Máscara Causal (Look-Ahead Mask)Implementada no arquivo multi_head_attention.py. A máscara garante que, durante o treinamento, o modelo não tenha acesso às palavras futuras da sequência.Lógica: Substituímos os valores acima da diagonal principal por $-\infty$. Ao aplicar o Softmax, esses valores tornam-se zero, impedindo o fluxo de informação do "futuro".2. Cross-AttentionImplementada no arquivo decoder.py. Este mecanismo permite que o Decoder consulte as informações processadas pelo Encoder.Diferença: Diferente da Self-Attention, aqui as Queries (Q) vêm do Decoder, enquanto as Keys (K) e Values (V) vêm da saída do Encoder.3. Loop de Inferência Auto-RegressivaLocalizado no main.py. O modelo gera um token por vez:Recebe o contexto gerado até o momento.Prediz a probabilidade do próximo token.Adiciona o token ao contexto e repete o processo até encontrar o token de parada <EOS>.Exemplo de Saída no TerminalAo executar o main.py, você verá o processo de geração palavra por palavra:Plaintext--- Iniciando Geração (Tarefa 3) ---
Token gerado: Transformers
Token gerado: são
Token gerado: poderosos
Token gerado: <EOS>

FRASE FINAL: <START> Transformers são poderosos
Nota sobre o desenvolvimento
Este projeto foi desenvolvido com o suporte da IA Gemini. A ferramenta foi utilizada exclusivamente como um par colaborativo para o esclarecimento de dúvidas pontuais, incluindo:

Explicação da lógica matemática por trás das máscaras triangulares superiores.

Refinamento da estrutura da classe DecoderLayer com torch.nn.Module.

Suporte na resolução de erros de ambiente (ModuleNotFoundError) e orientação nos comandos de versionamento via Git.
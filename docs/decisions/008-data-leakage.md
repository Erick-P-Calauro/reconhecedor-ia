# ADR 008 - Data Leakage

## Contexto

Esta decisão foi tomada após a execução do fluxo fine-tuned do reconhecimento facial. Essa decisão busca versar sobre o data leakage.

## Decisão

O sistema deve garantir, no momento da composição dos grupos de treino, validação e teste, que nenhuma foto vai estar presente em mais de um dos conjuntos, assim evitando o chamado data leakage.
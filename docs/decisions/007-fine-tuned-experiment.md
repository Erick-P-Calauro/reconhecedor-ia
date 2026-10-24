# ADR 007 - Fine Tuned Experiment

## Contexto

Esta decisão foi tomada após a execução do fluxo básico de reconhecimento facial. O objetivo é ajustar o fluxo de execução do sistema em experimentos que envolvam o re-treino dos modelos.

## Decisão

Em experimentos que envolvam fine tuning, os dados devem ser divididos em grupos de treino, validação e teste e executados em ordem. O primeiro serve para atualizar os parâmetros do modelo, o segundo para encontrar o threshold e o terceiro para avaliar o experimento.
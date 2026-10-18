# ADR 005 - Software Logging

## Contexto

Esta decisão foi tomada após testes com experimentos baseados no dataset lfw. O objetivo é melhorar a rastreabilidade de código, devido à possível demora inerente aos experimentos. Prentende-se aumentar a rastreabilidade por meio do logging dos passos executados pelo sistema.

## Decisão

Todas as camadas do sistema devem explicitar quando começam a ser executadas e quando terminam de ser executadas. O rastreio durante a execução é optativo, contudo é recomendável.
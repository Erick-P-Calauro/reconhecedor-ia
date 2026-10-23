# ADR 006 - Command Logging

## Contexto

Esta decisão foi tomada após a execução do fluxo básico de reconhecimento facial. O objetivo é melhorar a utilização do sistema por usuários não relacionados ao desenvolvimento do software.

## Decisão

Todos os comandos que podem ser executados pelo sistema devem ser aglutinados e explicados por meio de um comando adicional "help". Qualquer comando errado executado deve explicar a existência do help para ajudar o usuário a navegar pelo sistema.
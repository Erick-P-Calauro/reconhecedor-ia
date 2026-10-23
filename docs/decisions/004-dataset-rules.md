# ADR 004 - Dataset Rules

## Contexto

Esta decisão foi tomada após testes com experimentos baseados no dataset lfw. O objetivo é estabelecer a diferenciação entre datasets públicos e privados e suas características, de modo a facilitar a suas utilizações.

## Decisão

### Definições 
 - São considerados datasets públicos quaisquer datasets externos que podem ser utilizados com fins acadêmicos.
 - São considerados datasets privados quaisquer datasets internos ou que não podem ser utilizados com fins acadêmicos.

### Características
 - Cada dataset tem um dataloader próprio, mesmo que tal dataset seja materializado.
 - Cada dataset deve providenciar opção de execução direta, sem materialização.
 - Cada dataset deve providenciar opção de re-materialização via argumentos.
 - Datasets privados não devem ser disponibilizados via git, sendo seus arquivos resultantes (embeds, datasets e derivados) ignorados via .gitignore.

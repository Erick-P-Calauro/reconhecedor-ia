# Spec 003 - Face Recognition

Essa especificação trata da funcionalidade de reconhecimento facial. O sistema deve ser capaz de fazer o reconhecimento facial por meio de ferramentas de inteligência artificial.

## Requisitos

### Requisitos Funcionais

 - RF-01 : O sistema deve ser capaz de tratar imagens de datasets públicos para utilização no sistema de reconhecimento facial.
 - RF-02 : O sistema deve ser capaz de tratar arquivos de imagem para utilização no sistema de reconhecimento facial.
 - RF-03 : O sistema deve ser capaz de reconhecer imagens de pessoas previamente inseridas no domínio.
 - RF-04 : O sistema deve ser capaz de classificar pessoas não inseridas no domínio como pessoas desconhecidas.

### Requisitos Não Funcionais

 - RNF-01 : O sistema deve permitir a mudança de modelos utilizados por meio de uma interface comum sem exigir alterações significativas no pipeline.
 - RNF-02 : O sistema não deve disponibilizar arquivos de imagem ou dados biométricos privados de participantes.
 - RNF-03 : O sistema não deve disponibilizar publicamente qualquer artefato derivado de dados privados dos participantes, com exceção das métricas de avaliação do sistema.
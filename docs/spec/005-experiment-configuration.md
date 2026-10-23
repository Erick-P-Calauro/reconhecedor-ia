# Spec 005 - Experiment Configuration

Essa especificação trata da configuração dos experimentos disponíveis, no que concerne ao tamanho da amostra, quantidade de classes, fotos mínimas por classe e composição dos grupos de validação e teste (baseline), ou composição dos grupos de treino, validação e teste (fine-tuning).

## Requisitos

### Requisitos Funcionais 
 - RF-01 : Todo experimento deve ser capaz de receber parâmetros sobre tamanho da amostra, quantidade de classes e fotos mínimas por classe.
 - RF-02 : Experimentos baseline devem ser capazes de receber parâmetros sobre composição dos grupos de validação e teste. 
 - RF-03 : Experimentos com fine-tuning parcial ou total devem ser capazes de receber parâmetros sobre composição dos grupos de treino, validação e teste.
 - RF-04 : Os conjuntos de validação devem ser executados antes do teste.
 - RF-05 : Os conjuntos de validação devem ser utilizados para definir qual é a melhor threshold para utilização no conjunto de teste.

### Requisitos Não Funcionais
 - RNF-01 : Os parâmetros configuráveis devem ser explicitados na interface exposta ao usuário.
 - RNF-02 : O threshold escolhido à partir da validação deve ser mostrado ao usuário e registrado nos resultados.
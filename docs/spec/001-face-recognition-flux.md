# Spec 001 - Face Recognition Flux

Essa especificação trata do problema a ser resolvido nesse projeto, de maneira geral. O projeto tem como objetivo a implementação de um pipeline de reconhecimento facial que seja replicável para diferentes modelos e datasets. Cada replicação é chamada de experimento.

## Fluxo Esperado

O fluxo esperado de um experimento qualquer está disposto a seguir : 
 - Dataset -> Imagem -> Detecção da face -> Pré-Processamento -> Rede Neural -> Embedding facial -> Comparação -> Reconhecimento Facial -> Avaliação do resultado.
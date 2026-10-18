# ADR 001 - Software Architecture

## Contexto

Esta decisão foi tomada durante a concepção do projeto. O objetivo dessa decisão é definir o que é um experimento e como ele interage com o sistema.

## Decisão

O fluxo de reconhecimento facial do projeto deve ser implementado seguindo uma arquitetura em camadas. As camadas do projeto são : 
 - Dataloader : Camada responsável pelo carregamento das imagens dos datasets
 - FaceDetector : Camada responsável pela funcionalidade de detecção de faces
 - PreProcessor : Camada responsável por processar as imagens para que estejam em formatos adequados para a camada Model.
 - Model : Camada de rede neural e responsável pela geração dos embeddings faciais
 - FaceRecognition : Camada responsável por comparar os embeddings faciais e realizar o reconhecimento facial.
 - Evaluator : Camada responsável por gerar métricas e fazer avaliação do desempenho de um experimento.
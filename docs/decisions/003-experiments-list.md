# ADR 003 - Experiments List

## Contexto

Esta decisão foi tomada depois da implementação base do sistema e tem como objetivo definir uma lista de configuração de experimentos que serão implementados e executados pelo sistema.

## Decisão

Adotando a definição de experimento contida em "002-experiment-definition", serão implementados cinco experimentos : 
 - Público : (Detecção:Yunet; Reconhecimento: ResNet50 [via torchvision]; Dataset: Labeled Faces In The Wild [via Scikit-learn])
 - Público : (Detecção: Yunet; Reconhecimento : RestNet50 (FineTuned); Dataset: Labeled Faces In The Wild [via Scikit-learn])
 - Público : (Detecção:FaceNet; Reconhecimento: FaceNet ; Dataset: Labeled Faces In The Wild [via Scikit-learn])
 - Público : (Detecção:Yunet; Reconhecimento: FaceNet ; Dataset: Labeled Faces In The Wild [via Scikit-learn])
 - Private : (Detecção : Yunet; Reconhecimento : Facenet; Dataset : LFW + Private)
# ADR 002 - Experiment Definition

## Contexto

Esta decisão foi tomada durante a concepção do projeto. O objetivo dessa decisão é definir o que é um experimento e como ele interage com o sistema.

## Decisão

Definição de experimento : Um experimento é uma configuração de uma instância do sistema que contém modelo de detecção de face, modelo de reconhecimento de face, dataset utilizado e parâmetros de configuração.

Características de experimento : 
 - Cada experimento deve ser uma instância do sistema, implementando cada camada especificada na decisão "001-software-architecture".
 - Cada experimento é implementado na pasta src/experiments em uma subpasta dedicada.
 - Cada experimento contém em sua subpasta suas configurações, para que sua utilização seja replicada por qualquer usuário.
 - Cada experimento contém em sua subpasta seus resultados, para que sua replicação seja verificada por qualquer usuário.
 - Nenhuma implementação de experimento deve estar na mesma pasta da implementação base das camadas do sistema
# ADR 002 - Experiment Definition

## Contexto

Esta decisão foi tomada durante a concepção do projeto. O objetivo dessa decisão é definir o que é um experimento e como ele interage com o sistema.

## Decisão

Um experimento é uma configuração do sistema que contém modelo, dataset e parâmetros de configuração. Cada experimento deve ser uma instância do sistema, implementando cada camada especificada na decisão "001-software-architecture".
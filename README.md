# Reconhecedor Facial

Este repositório armazena a implementação de pipeline de reconhecimento facial replicável, onde cada replicação com modelo/dataset diferente é um experimento. Este sistema faz parte de um trabalho sobre reconhecimento facial por meio de técnicas de inteligência artificial.

Você pode consultar a pasta docs para entender quais funcionalidades o sistema pretende implementar e sobre quais restrições.

## Sumário 
 - [Arquitetura do sistema](#requisites)
 - [Como instalar as dependências](#instalacao)
 - [Como utilizar o sistema](#utilizacao)

<h2 id="arquitetura">Arquitetura do Sistema</h2>

O pipeline é implementado por meio de uma arquitetura em camadas, cada uma com interface própria em `src/reconhecedor/`:

| Camada           | Pacote                | Responsabilidade                                      |
|------------------|-----------------------|-------------------------------------------------------|
| Dataloader       | `dataloader/`         | Carregar imagens de datasets públicos ou por arquivos (`.png/.jpg/.jpeg`)     |
| FaceDetector     | `face_detector/`      | Detectar faces, descartar imagens sem face, recortar  |
| PreProcessor     | `preprocessor/`       | Padronizar a entrada da Model|
| Model            | `model/`              | Gerar o embedding facial (interface `Model`;) |
| FaceRecognition  | `face_recognition/`   | Comparar embeddings; reconhecer ou declarar `None`    |
| Evaluator        | `evaluator/`          | Métricas por experimento, salvas em arquivo           |

O arquivo `src/reconhecedor/pipeline.py` orquestra o fluxo genérico; as camadas chegam por
**injeção de dependência**, então trocar a Model (ex.: FaceNet/ArcFace) não exige alterações no pipeline — basta implementar `Model.embed`.

<h2 id="instalacao">Instalação</h2>

Requer Python 3.10+ (testado em 3.14, Windows).

```powershell
pip install -r requirements.txt
```

<h2 id="utilizacao">Como utilizar</h2>

### Comandos

```powershell

# Lista os experimentos de src/experiments/ (ADR 002)
python main.py experiments

# Roda o experimento baseline de ponta a ponta
python main.py run

# Consulta o histórico de métricas (na subpasta de cada experimento)
python main.py metrics
```

### Parâmetros

```powershell

# Define experimento, dataset, treshold e tamanho padrão das imagens via argumentos
python main.py run --experiment baseline --dataset data/meu-dataset `
                   --threshold 0.8 --image-size 128

# Consulta o histórico de métricas do experimento baseline
python main.py metrics --experiment baseline
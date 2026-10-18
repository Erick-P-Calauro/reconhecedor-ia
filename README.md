# Reconhecedor Facial

Experimentação para o problema de reconhecimento facial: um **pipeline de
reconhecimento facial replicável**, onde cada replicação com modelo/dataset
diferente é um **experimento** (Spec 001 + ADR 002).

## Arquitetura 

O pipeline é implementado como arquitetura em camadas, cada uma com interface
própria (ABC) em `src/reconhecedor/`:

| Camada           | Pacote                | Responsabilidade                                      |
|------------------|-----------------------|-------------------------------------------------------|
| Dataloader       | `dataloader/`         | Carregar imagens dos datasets (`.png/.jpg/.jpeg`)     |
| FaceDetector     | `face_detector/`      | Detectar faces, descartar imagens sem face, recortar  |
| PreProcessor     | `preprocessor/`       | Padronizar a entrada da Model (cinza, 128×128, [0,1]) |
| Model            | `model/`              | Gerar o embedding facial (interface `Model`; HOG do baseline no experimento) |
| FaceRecognition  | `face_recognition/`   | Comparar embeddings; reconhecer ou declarar `None`    |
| Evaluator        | `evaluator/`          | Métricas por experimento, salvas em arquivo           |

`src/reconhecedor/pipeline.py` orquestra o fluxo genérico; as camadas chegam por
**injeção de dependência**, então trocar a Model (ex.: FaceNet/ArcFace)
não exige alterações no pipeline — basta implementar `Model.embed`
(Spec 003, RNF-01).

## Instalação

Requer Python 3.10+ (testado em 3.14, Windows).

```powershell
pip install -r requirements.txt
```

## Uso

```powershell
# 1) Monta o dataset demo (imagens públicas) em data/demo —
#    a ferramenta é do experimento baseline (ADR 002):
#    src/experiments/baseline/demo.py
python main.py build-demo

# 2) Lista os experimentos de src/experiments/ (ADR 002)
python main.py experiments

# 3) Roda o experimento baseline de ponta a ponta
python main.py run

# 4) Consulta o histórico de métricas (na subpasta de cada experimento)
python main.py metrics
```

Opções úteis de `run` (sobrescrevem o `config.json` do experimento):

```powershell
python main.py run --experiment baseline --dataset data/meu-dataset `
                   --threshold 0.8 --image-size 128
python main.py metrics --experiment baseline
# reconhecedor-ia

Experimentação para o problema de reconhecimento facial: um **pipeline de
reconhecimento facial replicável**, onde cada replicação com modelo/dataset
diferente é um **experimento** (Spec 001 + ADR 002).

Fluxo (Spec 001):

```
Dataset -> Imagem -> Detecção da face -> Pré-Processamento -> Rede Neural
-> Embedding facial -> Comparação -> Reconhecimento Facial -> Avaliação
```

## Arquitetura (ADR 001)

O pipeline é implementado como arquitetura em camadas, cada uma com interface
própria (ABC) em `src/reconhecedor/`:

| Camada           | Pacote                | Responsabilidade                                      |
|------------------|-----------------------|-------------------------------------------------------|
| Dataloader       | `dataloader/`         | Carregar imagens dos datasets (`.png/.jpg/.jpeg`)     |
| FaceDetector     | `face_detector/`      | Detectar faces, descartar imagens sem face, recortar  |
| PreProcessor     | `preprocessor/`       | Padronizar a entrada da Model (cinza, 128×128, [0,1]) |
| Model            | `model/`              | Gerar o embedding facial (baseline HOG; DL plugável)  |
| FaceRecognition  | `face_recognition/`   | Comparar embeddings; reconhecer ou declarar `None`    |
| Evaluator        | `evaluator/`          | Métricas por experimento, salvas em arquivo           |

`src/reconhecedor/pipeline.py` orquestra o fluxo; as camadas chegam por
**injeção de dependência**, então trocar a Model (ex.: FaceNet/ArcFace)
não exige alterações no pipeline — basta implementar `Model.embed`
(Spec 003, RNF-01).

Componentes do experimento baseline: detector **YuNet** (OpenCV
`FaceDetectorYN`, modelo ONNX ~230 KB baixado automaticamente para `models/`),
Model **HOG clássico** (numpy, sem dependências pesadas de deep learning),
comparação por **cosseno** com limiar configurável.

## Instalação

Requer Python 3.10+ (testado em 3.14, Windows).

```powershell
pip install -r requirements.txt
```

## Uso

```powershell
# 1) Monta o dataset demo (imagens públicas) em data/demo
python main.py build-demo

# 2) Roda o experimento baseline de ponta a ponta
python main.py run

# 3) Consulta o histórico de métricas de todos os experimentos
python main.py metrics
```

Opções úteis de `run`:

```powershell
python main.py run --dataset data/meu-dataset --name experimento-2 `
                   --threshold 0.8 --image-size 128 --output results
```

Layout de dataset (datasets públicos como LFW seguem o mesmo formato):

```
data/<dataset>/
  gallery/<identidade>/*.png|jpg|jpeg   # pessoas cadastradas (reconhecidas)
  probe/<identidade>/*.png|jpg|jpeg     # a reconhecer; fora da galeria = desconhecida
```

Imagens sem face são detectadas e **descartadas** antes da avaliação
(Spec 002, RF-02).

## Métricas (Spec 004)

Cada experimento gera:

- `results/<nome-do-experimento>/<timestamp>/metrics.json` — relatório completo
  (configuração, contagens, acurácia, precisão, recall, FAR, FRR);
- `results/metrics_history.csv` — histórico agregado para consulta posterior
  (`python main.py metrics`).

Definições: acurácia = (TP+TN)/total; precisão = TP/(TP+FP);
recall = TP/(TP+FN); FAR = FP/(FP+TN) (desconhecidos aceitos);
FRR = FN/(FN+TP) (cadastrados recusados).

## Testes

```powershell
python -m pytest tests/ -q
```

Cobrem métricas (Spec 004), descarte de imagens sem face (Spec 002),
formatos de imagem, reconhecimento/desconhecido (Spec 003) e o pipeline
ponta a ponta (requer `python main.py build-demo`; sem ele o teste é pulado).

## Privacidade (Spec 003, RNF-02/RNF-03)

- `data/` (datasets/imagens) e `results/` **não são versionados**
  (`.gitignore`); imagens e dados biométricos privados de participantes não
  são disponibilizados publicamente.
- Apenas **métricas** podem ser publicadas como artefatos derivados.
- O dataset demo usa somente imagens públicas (scikit-image e amostras OpenCV).

## Estrutura do repositório

```
main.py                     # entrypoint (build-demo | run | metrics)
src/reconhecedor/           # pacote com as camadas da ADR 001
tests/                      # testes (pytest)
data/                       # datasets de leitura (gitignored; ver data/README.md)
results/                    # métricas por experimento (gitignored)
models/                     # modelo público do detector (auto-download)
docs/spec/                  # especificações 001-004
docs/decisions/             # ADRs 001-002
logs/                       # log de ações do agente
contexto.md                 # contexto consolidado (specs + decisions)
```

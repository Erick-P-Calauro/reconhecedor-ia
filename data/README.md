# data/ — datasets e imagens de leitura

Esta pasta concentra os **datasets** usados pelos experimentos (arquivos de
leitura do pipeline). Ela **não é versionada** (ver `.gitignore`): conforme a
Spec 003 (RNF-02/RNF-03), imagens e dados biométricos privados de
participantes não devem ser disponibilizados — apenas as métricas de
avaliação podem ser artefatos públicos.

## Layout esperado

```
data/
└── <dataset>/
    ├── gallery/          # pessoas previamente cadastradas no domínio (RF-03)
    │   └── <identidade>/*.png|jpg|jpeg
    └── probe/            # imagens a reconhecer
        └── <identidade>/*.png|jpg|jpeg
```

- Identidades presentes em `gallery/` são **reconhecidas**;
  identidades só em `probe/` são **desconhecidas** (classificadas como tal).
- Imagens sem face (em qualquer pasta) são **descartadas** pela camada
  FaceDetector antes da avaliação.

## Dataset demo

```powershell
python main.py build-demo     # baixa imagens públicas e monta data/demo
python main.py run            # roda o experimento baseline de ponta a ponta
```

O demo usa imagens públicas (scikit-image e amostras do OpenCV), não dados de
participantes. Datasets públicos reais (ex.: LFW) podem ser usados copiando ou
apontando `--dataset` para a pasta no layout acima.

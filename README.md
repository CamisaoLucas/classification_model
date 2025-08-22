# 🪨 Classificação de Rochas com AutoKeras

![Python](https://img.shields.io/badge/python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13+-orange)
![AutoKeras](https://img.shields.io/badge/autokeras-latest-red)

Classificação de imagens de rochas usando **AutoKeras**, com **pré-processamento offline** de crops e augmentações.


## 🗂 Estrutura do Dataset

imagens/
├─ train/
│ ├─ Bioclastic-Grainstone/
│ ├─ Mudstone/
│ ├─ Oolite/
│ └─ Wackestone/
└─ test/
├─ Bioclastic-Grainstone/
├─ Mudstone/
├─ Oolite/
└─ Wackestone/

## ⚙️ Pré-requisitos

- Python 3.11+
- TensorFlow
- AutoKeras
- OpenCV
- NumPy

> Instalação rápida: `pip install tensorflow autokeras opencv-python-headless numpy`

🛠 Pré-processamento

Execute o script augmentation.py para:

- Redimensionar imagens mantendo proporção.
-Criar 5 crops (top-left, top-right, bottom-left, bottom-right, center).
-Aplicar augmentações:
--Flip horizontal e vertical
--Rotação (90°, 180°, 270°)
--Brilho e contraste
--Zoom central

As imagens geradas não sobrescrevem as originais e recebem nomes únicos.

### 1️⃣ Crop e Augmentações

| Original | Crops | Flip/Rotação | Brilho/Zoom |
|----------|-------|--------------|-------------|
| ![original](docs/original.jpg) | ![crops](docs/crops.jpg) | ![flip](docs/flip_rotate.jpg) | ![brightness](docs/brightness_zoom.jpg) |


⚡ Treinamento:

Usando imagens já processadas:

import tensorflow as tf
import autokeras as ak

train_ds = tf.keras.utils.image_dataset_from_directory(
    "imagens/train",
    image_size=(256, 256),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "imagens/test",
    image_size=(256, 256),
    batch_size=32
)

# Normalização
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Criação do modelo
clf = ak.ImageClassifier(max_trials=3, overwrite=True)
clf.fit(train_ds, epochs=150, validation_data=test_ds)


💾 Exportar Modelo:

model = clf.export_model()
model.save("rock_classifier_model")

⚠️ Observações:

Diferentes dimensões nas imagens originais são automaticamente redimensionadas.
Todo o pré-processamento é offline.
Nomes únicos garantem que imagens originais não sejam sobrescritas.

📝 Autor:
Lucas Vinicius Camisão Alves
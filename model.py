import os
import tensorflow as tf
import autokeras as ak
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from autokeras.preprocessors.common import AddOneDimension, CastToString
from autokeras.preprocessors.encoders import OneHotEncoder
from keras_tuner import Objective

# Ativar logs detalhados do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Registrar classes personalizadas do AutoKeras
keras.saving.register_keras_serializable()(AddOneDimension)
keras.saving.register_keras_serializable()(CastToString)
for cls in [AddOneDimension, CastToString, OneHotEncoder]:
    keras.saving.register_keras_serializable()(cls)

# Parâmetros
img_size = (256, 256)  # tamanho dos crops
batch_size = 16

# ImageDataGenerator com augmentations para treino
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ImageDataGenerator para teste (apenas normalização)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Carregar imagens das pastas
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Função para converter generator em tf.data.Dataset
def generator_to_dataset(generator):
    def gen():
        for x, y in generator:
            yield x, y
    output_signature = (
        tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

train_ds = generator_to_dataset(train_generator)
test_ds = generator_to_dataset(test_generator)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Criar e treinar modelo com AutoKeras
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=30,
    loss="sparse_categorical_crossentropy",
    objective=Objective("val_loss", direction="min")
)

clf.fit(train_ds, epochs=150, validation_data=test_ds, callbacks=[early_stop, checkpoint])

# Exportar e salvar modelo final
model = clf.export_model()
model.save("meu_modelo_autokeras_final.h5")

# Avaliação
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
X_test = np.concatenate([x.numpy() for x, y in test_ds], axis=0)
preds = clf.predict(X_test)
preds = np.argmax(preds, axis=1)

# Matriz de confusão
cm = confusion_matrix(y_true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.savefig("matriz_confusao.png")

# Métricas detalhadas
print(f"Acurácia no teste: {accuracy_score(y_true, preds):.4f}\n")
print("Relatório de classificação detalhado:\n")
print(classification_report(y_true, preds, target_names=list(train_generator.class_indices.keys())))

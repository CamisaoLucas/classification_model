import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# Configurações
# ===============================
img_size = (224, 224)
batch_size = 32

train_dir = "dataset/train"
test_dir = "dataset/test"
best_model_path = "best_model.keras"

# ===============================
# Funções de pré-processamento
# ===============================
def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    return img, label

def one_hot_encode(x, y, num_classes):
    return x, tf.one_hot(y, depth=num_classes)

def create_dataset(base_dir):
    class_names = sorted(os.listdir(base_dir))
    class_indices = {name: i for i, name in enumerate(class_names)}

    all_images = []
    all_labels = []

    for class_name in class_names:
        paths = glob.glob(os.path.join(base_dir, class_name, "*.jpg"))
        all_images.extend(paths)
        all_labels.extend([class_indices[class_name]] * len(paths))

    ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    ds = ds.shuffle(buffer_size=len(all_images), reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: one_hot_encode(x, y, len(class_names)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, class_names

# ===============================
# Criar datasets
# ===============================
train_ds, class_names = create_dataset(train_dir)
test_ds, _ = create_dataset(test_dir)
num_classes = len(class_names)

# ===============================
# Calcular class_weights
# ===============================
class_counts = {class_name: len(glob.glob(os.path.join(train_dir, class_name, "*.jpg")))
                for class_name in class_names}
print("Distribuição de classes no treino:", class_counts)

all_labels = []
for class_name, count in class_counts.items():
    all_labels.extend([class_name] * count)

label_indices = [class_names.index(lbl) for lbl in all_labels]

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=label_indices
)

class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class Weights:", class_weights)

# ===============================
# Modelo com ResNet50
# ===============================
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=img_size + (3,)
)
base_model.trainable = False  # congela pesos base

inputs = tf.keras.Input(shape=img_size + (3,))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===============================
# Callbacks
# ===============================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ===============================
# Treinar
# ===============================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stopping, checkpoint]
)

# ===============================
# Avaliação final
# ===============================
best_model = tf.keras.models.load_model(best_model_path)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
print(f"\nAcurácia no conjunto de teste: {acc:.4f}\n")

print("Relatório de classificação detalhado:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===============================
# Matriz de confusão
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predição")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.savefig("confusion_matrix.png")
plt.close()

# ===============================
# Gráficos de treinamento
# ===============================
def plot_training(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Treino")
    plt.plot(epochs_range, val_acc, label="Validação")
    plt.legend(loc="lower right")
    plt.title("Acurácia por Época")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Treino")
    plt.plot(epochs_range, val_loss, label="Validação")
    plt.legend(loc="upper right")
    plt.title("Loss por Época")

    plt.savefig("training_curves.png")
    plt.close()

plot_training(history)

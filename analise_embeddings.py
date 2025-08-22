"""
Pós-processamento no conjunto de teste:
1) extrair embeddings (vetores de características) do penúltimo estágio do modelo.
2) reduzir para 2D para visualização.
3) destacar as amostras classificadas incorretamente.
4) calcular centroides por classe (média dos embeddings).
5) reclassificar cada amostra pela menor distância euclidiana ao centroide.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import cdist
import tensorflow as tf

# ===============================
# Configurações
# ===============================
img_size = (224, 224)
batch_size = 32

train_dir = "dataset/train"
test_dir = "dataset/test"
best_model_path = "best_model.keras"

# ===============================
# Funções auxiliares
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
        paths = tf.io.gfile.glob(os.path.join(base_dir, class_name, "*.jpg"))
        all_images.extend(paths)
        all_labels.extend([class_indices[class_name]] * len(paths))

    ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: one_hot_encode(x, y, len(class_names)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, class_names, np.array(all_labels)

# ===============================
# Carregar modelo + dados
# ===============================
test_ds, class_names, y_true = create_dataset(test_dir)
num_classes = len(class_names)

model = tf.keras.models.load_model(best_model_path)

# ===============================
# Extrair embeddings
# ===============================
# Pega a saída da penúltima camada (antes do Dense softmax)
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output  # camada de pooling/dropout
)

embeddings = []
preds = []

for images, labels in test_ds:
    emb = embedding_model.predict(images, verbose=0)
    pred = model.predict(images, verbose=0)

    embeddings.append(emb)
    preds.append(np.argmax(pred, axis=1))

embeddings = np.vstack(embeddings)
preds = np.concatenate(preds)
y_true = np.array(y_true)

print(f"Embeddings extraídos: {embeddings.shape}")

# ===============================
# Destacar erros em gráfico 2D
# ===============================
# Reduz dimensões
pca = PCA(n_components=50).fit_transform(embeddings)
emb2d = TSNE(n_components=2, random_state=42).fit_transform(pca)

errors = preds != y_true

plt.figure(figsize=(10, 8))
plt.scatter(emb2d[:, 0], emb2d[:, 1], c=y_true, cmap="tab10", alpha=0.5, label="Correta")
plt.scatter(emb2d[errors, 0], emb2d[errors, 1], c="red", marker="x", label="Erros")
plt.legend()
plt.title("Embeddings das imagens (erros destacados)")
plt.savefig("embeddings_erros.png")
plt.close()

# ===============================
# Calcular centroides
# ===============================
centroids = {}
for c in range(num_classes):
    centroids[c] = embeddings[y_true == c].mean(axis=0)

centroids_matrix = np.vstack([centroids[c] for c in range(num_classes)])

# ===============================
# Reclassificação por distância euclidiana
# ===============================
distances = cdist(embeddings, centroids_matrix, metric="euclidean")
new_preds = distances.argmin(axis=1)

# ===============================
# Relatório da reclassificação
# ===============================
acc = accuracy_score(y_true, new_preds)
print(f"\nAcurácia após reclassificação por centroides: {acc:.4f}\n")
print("Relatório de classificação:\n")
print(classification_report(y_true, new_preds, target_names=class_names))

cm = confusion_matrix(y_true, new_preds, labels=range(num_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de confusão (Reclassificação por centroides)")
plt.savefig("confusion_matrix_centroides.png")
plt.close()

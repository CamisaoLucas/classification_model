import os
import shutil
import random

# Caminho da pasta principal do dataset
dataset_path = "dataset"

# Pastas train e test
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Garantir que as pastas de classe existam no test
for class_name in os.listdir(train_dir):
    class_test_path = os.path.join(test_dir, class_name)
    if not os.path.exists(class_test_path):
        os.makedirs(class_test_path)

# Para cada classe em train
for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)
    class_test_path = os.path.join(test_dir, class_name)
    
    # Lista todos os arquivos na pasta da classe
    images = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]
    
    # Seleciona aleatoriamente 30% das imagens
    num_to_move = max(1, int(len(images) * 0.3))  # pelo menos 1 imagem
    images_to_move = random.sample(images, num_to_move)
    
    # Move as imagens selecionadas para a pasta de test correspondente
    for image in images_to_move:
        src_path = os.path.join(class_train_path, image)
        dst_path = os.path.join(class_test_path, image)
        shutil.move(src_path, dst_path)
    
    print(f"{num_to_move} imagens movidas da classe '{class_name}' para a pasta test.")

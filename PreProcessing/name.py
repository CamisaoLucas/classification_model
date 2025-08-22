import os

# Caminho da pasta principal do dataset
dataset_path = "dataset"  # ajuste para o caminho correto

# Mapeamento de pastas/classes para prefixos
prefix_map = {
    "Bioclastic-Grainstone": "bg",
    "Mudstone": "ms",
    "Oolite": "ol",
    "Wackestone": "ws"
}

# Função para renomear imagens em uma pasta
def rename_images_in_folder(folder_path, subset_name):
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            prefix = prefix_map.get(class_folder)
            if not prefix:
                print(f"Aviso: Nenhum prefixo definido para a pasta '{class_folder}'. Pulando...")
                continue
            prefix = f"{prefix}_{subset_name.lower()}"
            for idx, filename in enumerate(os.listdir(class_path), start=1):
                old_file = os.path.join(class_path, filename)
                # Mantém a extensão original
                ext = os.path.splitext(filename)[1]
                new_file = os.path.join(class_path, f"{prefix}_{idx}{ext}")
                os.rename(old_file, new_file)
            print(f"Pasta '{class_folder}' renomeada com prefixo '{prefix}'.")

# Renomear imagens nas pastas train e test
for subset in ["train", "test"]:
    subset_path = os.path.join(dataset_path, subset)
    if os.path.exists(subset_path):
        rename_images_in_folder(subset_path, subset)
    else:
        print(f"Pasta '{subset}' não encontrada no dataset.")

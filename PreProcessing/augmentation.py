import cv2
import os
import glob
import numpy as np

input_base = "imagens"          # Pasta das imagens originais
output_base = "augmentadas"     # Pasta onde as augmentações serão salvas
os.makedirs(output_base, exist_ok=True)

target_size = 256  # Lado do crop final

def resize_keep_aspect(img, min_side):
    h, w = img.shape[:2]
    scale = max(min_side / h, min_side / w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def five_crops(img, size):
    h, w = img.shape[:2]
    crops = []
    crops.append(img[0:size, 0:size])         # Top-left
    crops.append(img[0:size, w-size:w])       # Top-right
    crops.append(img[h-size:h, 0:size])       # Bottom-left
    crops.append(img[h-size:h, w-size:w])     # Bottom-right
    center_y, center_x = h // 2, w // 2
    crops.append(img[center_y - size//2:center_y + size//2,
                     center_x - size//2:center_x + size//2])  # Center
    return crops

def augmentations(img):
    aug_list = []

    # Flips
    aug_list.append(cv2.flip(img, 1))   # horizontal
    aug_list.append(cv2.flip(img, 0))   # vertical

    # Rotations
    rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot180 = cv2.rotate(img, cv2.ROTATE_180)
    rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    aug_list.extend([rot90, rot180, rot270])

    # Brilho/contraste
    for alpha, beta in [(1.2, 30), (0.8, -30)]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        aug_list.append(bright)

    # Zoom central
    h, w = img.shape[:2]
    zoom = img[h//4:3*h//4, w//4:3*w//4]
    zoom = cv2.resize(zoom, (w, h))
    aug_list.append(zoom)

    return aug_list

# Loop pelas pastas de imagens originais
for pasta in os.listdir(input_base):
    input_folder = os.path.join(input_base, pasta)
    if not os.path.isdir(input_folder):
        continue
    
    output_folder = os.path.join(output_base, pasta)
    os.makedirs(output_folder, exist_ok=True)

    for img_path in glob.glob(os.path.join(input_folder, "*")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        nome = os.path.splitext(os.path.basename(img_path))[0]

        # Redimensiona mantendo proporção para que lado menor >= target_size
        img_resized = resize_keep_aspect(img, target_size)

        # Gera 5 crops
        crops = five_crops(img_resized, target_size)

        total_generated = 0

        # Para cada crop, salvar crop e gerar augmentations
        for i, crop in enumerate(crops):
            crop_name = f"{nome}_crop{i+1}"
            # Salva o crop original
            cv2.imwrite(os.path.join(output_folder, f"{crop_name}.jpg"), crop)
            total_generated += 1

            # Augmentations do crop
            extra_augs = augmentations(crop)
            for j, aug in enumerate(extra_augs):
                cv2.imwrite(os.path.join(output_folder, f"{crop_name}_aug{j+1}.jpg"), aug)
                total_generated += 1

        print(f"✅ {total_generated} imagens geradas para {img_path}")

print("\n🎉 Finalizado! Todas as imagens augmentadas estão na pasta 'augmentadas/'.")

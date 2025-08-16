import cv2
import os
import glob

# Pasta base onde estão as imagens originais
input_base = "imagens"

for pasta in os.listdir(input_base):
    input_folder = os.path.join(input_base, pasta)
    if not os.path.isdir(input_folder):
        continue

    print(f"\n📂 Pasta: {pasta}")
    for img_path in glob.glob(os.path.join(input_folder, "*")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]  # altura, largura
        print(f"🖼️ {os.path.basename(img_path)} -> {w} x {h}")

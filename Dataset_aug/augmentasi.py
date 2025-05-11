import os
import random
import cv2
import pandas as pd
import albumentations as A
from collections import defaultdict
from albumentations import Compose, Rotate, RandomCrop, RandomScale
from tqdm import tqdm

# Definisi transformasi augmentasi dengan penyesuaian padding
transform = Compose([
    A.PadIfNeeded(min_height=200, min_width=200, border_mode=0, value=0),  # Menambah padding jika gambar kecil
    Rotate(limit=30, p=0.7),
    RandomCrop(width=200, height=200, p=0.5),
    RandomScale(scale_limit=0.2, p=0.5),
])

def augment_train_balanced(
    attention_csv_path,
    train_folder,
    output_folder,
    total_augmented_images=80
):
    os.makedirs(output_folder, exist_ok=True)

    # 1. Load CSV tanpa header
    df = pd.read_csv(attention_csv_path, header=None, names=["filename", "attention_level"])
    df['attention_level'] = df['attention_level'].astype(int)

    # 2. Filter hanya file yang ada di folder train
    train_filenames = set(os.listdir(train_folder))
    df_train = df[df['filename'].isin(train_filenames)]

    # 3. Hitung distribusi attention level
    attention_counts = df_train['attention_level'].value_counts().to_dict()
    print("Distribusi awal di train:", attention_counts)

    # 4. Temukan attention level yang jumlahnya paling sedikit
    levels_sorted = sorted(attention_counts.items(), key=lambda x: x[1])  # [(0,10), (3,15),...]

    # 5. Buat mapping level ke list gambar
    attention_to_files = defaultdict(list)
    for _, row in df_train.iterrows():
        attention_to_files[row['attention_level']].append(row['filename'])

    # 6. Lakukan augmentasi untuk mencapai total target
    aug_count = 0
    level_index = 0
    pbar = tqdm(total=total_augmented_images)

    while aug_count < total_augmented_images:
        # Pilih attention level paling sedikit
        level = levels_sorted[level_index % len(levels_sorted)][0]
        candidates = attention_to_files[level]

        if not candidates:
            level_index += 1
            continue

        # Ambil random gambar dari level ini
        img_name = random.choice(candidates)
        img_path = os.path.join(train_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Gagal baca: {img_path}")
            level_index += 1
            continue

        augmented = transform(image=image)
        aug_img = augmented["image"]

        # Simpan hasil augmentasi
        aug_name = f"{os.path.splitext(img_name)[0]}_aug{aug_count+1}.png"
        output_path = os.path.join(output_folder, aug_name)
        cv2.imwrite(output_path, aug_img)

        aug_count += 1
        level_index += 1
        pbar.update(1)

    pbar.close()
    print(f"✅ Selesai augmentasi. {total_augmented_images} gambar ditambahkan ke {output_folder}")

# CONTOH PENGGUNAAN:
augment_train_balanced(
    attention_csv_path='attention.csv',
    train_folder='./train',
    output_folder='./train_aug',
    total_augmented_images=80
)

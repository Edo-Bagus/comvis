import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def extract_color_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    pixels_rgb = img.reshape(-1, 3)
    pixels_hsv = img_hsv.reshape(-1, 3)

    mean_r, mean_g, mean_b = np.mean(pixels_rgb, axis=0)
    std_r, std_g, std_b = np.std(pixels_rgb, axis=0)

    mean_h, mean_s, mean_v = np.mean(pixels_hsv, axis=0)
    std_h, std_s, std_v = np.std(pixels_hsv, axis=0)

    luminance = 0.2126 * pixels_rgb[:, 0] + 0.7152 * pixels_rgb[:, 1] + 0.0722 * pixels_rgb[:, 2]
    mean_luminance = np.mean(luminance)
    mean_saturation = mean_s

    return {
        'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
        'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
        'mean_h': mean_h, 'mean_s': mean_s, 'mean_v': mean_v,
        'std_h': std_h, 'std_s': std_s, 'std_v': std_v,
        'mean_luminance': mean_luminance,
        'mean_saturation': mean_saturation
    }

def global_contrast_std_luminance(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    luminance_std = np.std(gray)
    return luminance_std

def wcag_contrast_ratio(lum1, lum2):
    L1, L2 = max(lum1, lum2), min(lum1, lum2)
    return (L1 + 0.05) / (L2 + 0.05)

def contrast_textlike_vs_background(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contrast_values = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter kontur kecil (mirip karakter)
        if 5 < w < 100 and 5 < h < 100:
            roi = gray[y:y+h, x:x+w]
            text_luminance = np.mean(roi)

            bg_patch = gray[max(0, y-10):y, x:x+w]
            if bg_patch.size == 0:
                bg_patch = gray[min(y+h+1, gray.shape[0]-1):y+h+11, x:x+w]

            bg_luminance = np.mean(bg_patch) if bg_patch.size > 0 else 255
            contrast = wcag_contrast_ratio(text_luminance / 255.0, bg_luminance / 255.0)
            contrast_values.append(contrast)

    return np.mean(contrast_values) if contrast_values else 0.0

def bright_dark_ratio(image_path, threshold=127):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    bright_pixels = np.sum(gray > threshold)
    return bright_pixels / total_pixels

def dominant_colors(image_path, n_colors=3):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = img_rgb.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)

    counts = np.bincount(kmeans.labels_)
    dominant_idx = np.argmax(counts)

    dominant_rgb = kmeans.cluster_centers_[dominant_idx].astype(int)
    dominant_hsv = cv2.cvtColor(np.uint8([[dominant_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    return dominant_rgb.tolist(), dominant_hsv.tolist()

def color_temperature_proportion(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Tentukan batasan untuk warna hangat (R > G > B) dan dingin (B > G > R)
    warm_pixels = np.sum((img_rgb[:, :, 0] > img_rgb[:, :, 1]) & (img_rgb[:, :, 0] > img_rgb[:, :, 2]))
    cold_pixels = np.sum((img_rgb[:, :, 2] > img_rgb[:, :, 0]) & (img_rgb[:, :, 2] > img_rgb[:, :, 1]))

    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    
    warm_percentage = (warm_pixels / total_pixels) * 100
    cold_percentage = (cold_pixels / total_pixels) * 100

    return warm_percentage, cold_percentage

def unique_colors_count(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Reshape the image and remove duplicates to count unique colors
    unique_colors = np.unique(img_rgb.reshape(-1, img_rgb.shape[2]), axis=0)

    return len(unique_colors)

def quantize_colors(image_path, n_clusters=16):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

    # Reshape image into a 2D array of pixels
    pixels = img_rgb.reshape(-1, 3)

    # Apply KMeans to quantize the image colors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)
    
    # Get the quantized colors (centroids)
    quantized_colors = np.unique(kmeans.labels_, return_counts=True)

    return len(quantized_colors[0])

def process_all_image_features(folder_path, output_csv):
    image_features = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            name_only = os.path.splitext(filename)[0]
            try:
                image_number = name_only
            except ValueError:
                continue  # skip jika nama bukan angka

            # Ekstraksi fitur warna
            color_features = extract_color_features(image_path)
            if not color_features:
                continue

            # Ekstraksi fitur kontras & warna lainnya
            luminance_std = global_contrast_std_luminance(image_path)
            contrast_text = contrast_textlike_vs_background(image_path)
            bright_ratio = bright_dark_ratio(image_path)
            dominant_rgb, dominant_hsv = dominant_colors(image_path)
            warm_percentage, cold_percentage = color_temperature_proportion(image_path)
            unique_colors = unique_colors_count(image_path)
            quantized_colors = quantize_colors(image_path)

            # Pastikan array dominan RGB & HSV valid (3 elemen)
            rgb_dict = {}
            hsv_dict = {}
            if isinstance(dominant_rgb, (list, tuple, np.ndarray)) and len(dominant_rgb) == 3:
                rgb_dict = {f'dominant_rgb_{i}': val for i, val in enumerate(dominant_rgb)}
            if isinstance(dominant_hsv, (list, tuple, np.ndarray)) and len(dominant_hsv) == 3:
                hsv_dict = {f'dominant_hsv_{i}': val for i, val in enumerate(dominant_hsv)}

            # Gabungkan semua fitur
            combined_features = {
                'image_name': image_number,
                **color_features,
                **rgb_dict,
                **hsv_dict,
                'warm_percentage': warm_percentage,
                'cold_percentage': cold_percentage,
                'unique_colors': unique_colors,
                'quantized_colors': quantized_colors,
                'luminance_std': luminance_std,
                'contrast_text_vs_background': contrast_text,
                'bright_dark_ratio': bright_ratio
            }

            image_features.append(combined_features)

    # Buat DataFrame dan urutkan berdasarkan nama gambar (angka)
    df = pd.DataFrame(image_features)
    df = df.sort_values(by='image_name').reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    print(f"✅ Semua fitur berhasil digabung dan disimpan: {output_csv}")


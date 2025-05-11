import os
import cv2
import numpy as np
import easyocr
import pandas as pd
import textstat

def extract_visual_composition_features(image_path):
    """
    Mengekstraksi fitur komposisi visual seperti simetri, aspect ratio, visual balance,
    dan kepatuhan terhadap rule of thirds.

    Args:
        image_path (str): Path ke file gambar.

    Returns:
        dict: Berisi fitur-fitur komposisi visual.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca gambar di path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # --- Simetri Horizontal (LR) dan Vertikal (TB) ---
    left_half = gray[:, :width // 2]
    right_half = gray[:, width - width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)

    diff_lr = np.abs(left_half.astype("float") - right_half_flipped.astype("float"))
    symmetry_lr = 1 - (np.mean(diff_lr) / 255)

    top_half = gray[:height // 2, :]
    bottom_half = gray[height - height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)

    diff_tb = np.abs(top_half.astype("float") - bottom_half_flipped.astype("float"))
    symmetry_tb = 1 - (np.mean(diff_tb) / 255)

    # --- Aspect Ratio ---
    aspect_ratio = round(width / height, 3)

    # --- Visual Balance (gunakan float64 untuk hindari overflow) ---
    left_sum = np.sum(left_half.astype(np.float64))
    right_sum = np.sum(right_half.astype(np.float64))
    lr_balance = 1 - abs(left_sum - right_sum) / (left_sum + right_sum)

    top_sum = np.sum(top_half.astype(np.float64))
    bottom_sum = np.sum(bottom_half.astype(np.float64))
    tb_balance = 1 - abs(top_sum - bottom_sum) / (top_sum + bottom_sum)

    # --- Rule of Thirds Compliance (pakai edge sebagai indikator visual interest) ---
    edges = cv2.Canny(gray, 100, 200)
    thirds_x = [width // 3, 2 * width // 3]
    thirds_y = [height // 3, 2 * height // 3]

    window_size = min(width, height) // 10  # area sekitar titik kuat
    roi_strengths = []

    for y in thirds_y:
        for x in thirds_x:
            x1 = max(0, x - window_size // 2)
            y1 = max(0, y - window_size // 2)
            x2 = min(width, x + window_size // 2)
            y2 = min(height, y + window_size // 2)
            roi = edges[y1:y2, x1:x2]
            roi_strengths.append(np.mean(roi) / 255)  # normalize

    thirds_score = round(np.mean(roi_strengths), 3)

    return {
        "height": height,
        "width": width,
        "symmetry_left_right": round(symmetry_lr, 3),
        "symmetry_top_bottom": round(symmetry_tb, 3),
        "aspect_ratio": aspect_ratio,
        "balance_left_right": round(lr_balance, 3),
        "balance_top_bottom": round(tb_balance, 3),
        "rule_of_thirds_score": thirds_score
    }

def extract_edge_features(image_path, low_threshold=100, high_threshold=200, num_orientation_bins=8):
    """
    Mengekstraksi berbagai fitur edge untuk menilai kompleksitas visual dari sebuah gambar poster.

    Args:
        image_path (str): Path ke file gambar.
        low_threshold (int): Threshold rendah untuk Canny.
        high_threshold (int): Threshold tinggi untuk Canny.
        num_orientation_bins (int): Jumlah bin histogram untuk distribusi arah tepi.

    Returns:
        dict: {
            "edge_count": int,
            "edge_density": float,
            "mean_edge_orientation": float,
            "std_edge_orientation": float,
            "edge_orientation_histogram": list,
            "contour_count": int,
            "avg_contour_length": float
        }
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca gambar di path: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Fitur dasar
    edge_count = int(np.count_nonzero(edges))
    h, w = edges.shape
    edge_density = edge_count / (h * w)

    # Gradient arah (menggunakan Sobel)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_orientation = np.arctan2(sobely, sobelx) * 180 / np.pi  # konversi ke derajat
    gradient_orientation = (gradient_orientation + 180) % 180  # arah 0–180 derajat

    edge_mask = edges > 0
    edge_orientations = gradient_orientation[edge_mask]
    if edge_orientations.size > 0:
        mean_orientation = float(np.mean(edge_orientations))
        std_orientation = float(np.std(edge_orientations))
        hist, _ = np.histogram(edge_orientations, bins=num_orientation_bins, range=(0, 180))
        hist_normalized = hist / hist.sum()  # normalisasi ke proporsi
        edge_orientation_histogram = hist_normalized.tolist()
    else:
        mean_orientation = 0.0
        std_orientation = 0.0
        edge_orientation_histogram = [0.0] * num_orientation_bins

    # Kontur
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_lengths = [cv2.arcLength(c, True) for c in contours]
    contour_count = len(contours)
    avg_contour_length = float(np.mean(contour_lengths)) if contour_lengths else 0.0

    return {
        "edge_count": edge_count,
        "edge_density": edge_density,
        "mean_edge_orientation": mean_orientation,
        "std_edge_orientation": std_orientation,
        "edge_orientation_histogram": edge_orientation_histogram,
        "contour_count": contour_count,
        "avg_contour_length": avg_contour_length
    }

def analyze_text_features(image_path, confidence_threshold=0.6, min_confidence=0.5):
    """
    Menganalisis berbagai fitur teks dari sebuah gambar poster menggunakan EasyOCR.

    Args:
        image_path (str): Path ke file gambar.
        confidence_threshold (float): Threshold confidence untuk validasi teks.
        min_confidence (float): Confidence minimal untuk ukuran teks dan variasi font.

    Returns:
        dict: Hasil analisis semua fitur teks dalam satu dictionary.
    """
    reader = easyocr.Reader(['en', 'id'], gpu=False)
    results = reader.readtext(image_path)

    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gagal membaca gambar dari {image_path}")
    img_height, img_width = image.shape[:2]
    total_image_area = img_height * img_width

    # --- Fitur 1: Word, Line, Char count ---
    valid_texts = [res[1] for res in results if res[2] >= confidence_threshold]
    word_count = sum(len(t.split()) for t in valid_texts)
    line_count = len(valid_texts)
    char_count = sum(len(t) for t in valid_texts)

    # --- Fitur 2: Area rasio teks ---
    total_text_area = 0
    y_centers = []
    widths = []
    heights = []

    for (bbox, text, conf) in results:
        if not text.strip():
            continue

        # Ukuran kotak bounding box
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height
        total_text_area += area

        # Posisi vertikal tengah (y-center)
        y_center = np.mean(y_coords)
        y_centers.append(y_center)

        if conf >= min_confidence:
            widths.append(width)
            heights.append(height)

    text_area_ratio = total_text_area / total_image_area if total_image_area > 0 else 0

    # --- Fitur 3: Posisi dominan teks ---
    if len(y_centers) == 0:
        text_position = "no_text"
    else:
        y_centers_norm = np.array(y_centers) / img_height
        top = np.sum((y_centers_norm >= 0.0) & (y_centers_norm < 0.33))
        center = np.sum((y_centers_norm >= 0.33) & (y_centers_norm < 0.66))
        bottom = np.sum((y_centers_norm >= 0.66) & (y_centers_norm <= 1.0))

        if max(top, center, bottom) == top:
            text_position = "top"
        elif max(top, center, bottom) == center:
            text_position = "center"
        else:
            text_position = "bottom"

    # --- Fitur 4: Ukuran rata-rata teks ---
    if widths and heights:
        avg_text_width = round(np.mean(widths), 2)
        avg_text_height = round(np.mean(heights), 2)
    else:
        avg_text_width = avg_text_height = 0

    # --- Fitur 5: Estimasi variasi font ---
    if len(widths) >= 2 and len(heights) >= 2:
        std_width = round(np.std(widths), 2)
        std_height = round(np.std(heights), 2)
        font_variety = "Likely varied fonts/styles" if std_width > 20 or std_height > 10 else "Likely uniform font"
    else:
        std_width = std_height = 0
        font_variety = "Insufficient text"

    # --- Fitur 6: FRE
    # Ambil hanya teks dengan confidence cukup
    valid_texts = [text for _, text, conf in results if conf >= min_confidence]
    full_text = " ".join(valid_texts)

    if not full_text.strip():
        return {
            "flesch_score": 0,
            "readability": "No readable text detected"
        }

    # Hitung Flesch Reading Ease Score
    score = textstat.flesch_reading_ease(full_text)

    # Interpretasi skor (berdasarkan standar FRES)
    if score >= 90:
        level = "Very Easy (5th grade)"
    elif score >= 80:
        level = "Easy (6th grade)"
    elif score >= 70:
        level = "Fairly Easy (7th grade)"
    elif score >= 60:
        level = "Standard (8th-9th grade)"
    elif score >= 50:
        level = "Fairly Difficult (10th-12th grade)"
    elif score >= 30:
        level = "Difficult (College)"
    else:
        level = "Very Confusing (College graduate)"

    return {
        "word_count": word_count,
        "line_count": line_count,
        "char_count": char_count,
        "text_area_ratio": text_area_ratio,
        "dominant_text_position": text_position,
        "avg_text_width": avg_text_width,
        "avg_text_height": avg_text_height,
        "font_width_std": std_width,
        "font_height_std": std_height,
        "font_variety_estimation": font_variety,
        "flesch_score": round(score, 2),
        "readability": level
    }

def process_shape_features(folder_path, output_csv):
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0]
            try:
                image_number = int(name)
            except ValueError:
                continue

            text = analyze_text_features(path)
            edge = extract_edge_features(path)
            visual = extract_visual_composition_features(path)

            data.append({'image_name': filename} | text | edge | visual)

    df = pd.DataFrame(data)
    df = df.sort_values(by='image_name').reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV berhasil disimpan: {output_csv}")
    
def resize_with_padding(image, target_size=(600, 600)):
    old_size = image.shape[:2]  # (height, width)
    ratio = min(target_size[0]/old_size[0], target_size[1]/old_size[1])
    new_size = tuple([int(x*ratio) for x in old_size[::-1]])  # (width, height)

    image_resized = cv2.resize(image, new_size)

    delta_w = target_size[1] - new_size[0]
    delta_h = target_size[0] - new_size[1]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)

    color = [0, 0, 0]  # hitam
    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

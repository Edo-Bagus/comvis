from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random

def display_images(dataset_path, n=5, random_state=None):
    """
    Displays n random images from the dataset folder.

    Args:
        dataset_path (str): Path to the dataset folder.
        n (int): Number of images to display.
        random_state (int, optional): Seed for reproducibility.
    """
    dataset_path = Path(dataset_path)
    image_paths = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png'))

    if not image_paths:
        print("No image files found in the dataset path.")
        return

    if random_state is not None:
        random.seed(random_state)

    selected_paths = random.sample(image_paths, min(n, len(image_paths)))

    # Display images in a grid
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, img_path in enumerate(selected_paths):
        try:
            image = Image.open(img_path)
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.title(img_path.name, fontsize=9)
            plt.axis('off')
        except Exception as e:
            print(f"Error displaying {img_path}: {e}")

    plt.tight_layout()
    plt.show()

def get_image_shape(image_path):
    """
    Returns the shape (width, height, channels) of an image given its path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (width, height, channels)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            channels = len(img.getbands())  # e.g., RGB = 3, RGBA = 4
            return (width, height, channels)
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_all_images_shape(dataset_path):
    """
    Returns the shape (width, height, channels) of all images in the dataset folder.

    Args:
        dataset_path (str): Path to the dataset folder.

    Returns:
        list: List of tuples, where each tuple is (width, height, channels) of an image.
    """
    dataset_path = Path(dataset_path)
    image_paths = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png'))

    if not image_paths:
        print("No image files found in the dataset path.")
        return []

    image_shapes = []
    
    for img_path in image_paths:
        shape = get_image_shape(img_path)  # Reusing the get_image_shape function
        if shape:
            image_shapes.append((img_path.name, shape))  # Store the image name with its shape
    
    return image_shapes

def plot_image_shapes(df):
    """
    Plots a scatter plot of image sizes where x-axis is the width and y-axis is the height,
    and each point is colored based on its 'Attention' label.

    Args:
        df (pd.DataFrame): DataFrame containing 'Images' (image file paths) and 'Attention' (label).
    """
    widths = []
    heights = []
    labels = []

    for _, row in df.iterrows():
        image_path = row['Image']
        label = row['Attention']
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                labels.append(label)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(widths, heights, c=labels, cmap='viridis', alpha=0.7)

    plt.title("Image Shapes: Width vs Height", fontsize=14)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.grid(True)

    # Add color legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Attention Label')

    plt.show()

def plot_rgb_histogram(image_path):
    """
    Displays the RGB histogram of a given image.

    Args:
        image_path (str): Path to the image file.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure image is in RGB mode
            # Split into R, G, B channels
            r, g, b = img.split()

            # Get pixel values for each channel
            r_values = list(r.getdata())
            g_values = list(g.getdata())
            b_values = list(b.getdata())

            # Plot histograms
            plt.figure(figsize=(10, 5))
            plt.hist(r_values, bins=256, color='red', alpha=0.5, label='Red')
            plt.hist(g_values, bins=256, color='green', alpha=0.5, label='Green')
            plt.hist(b_values, bins=256, color='blue', alpha=0.5, label='Blue')

            plt.title(f"RGB Histogram for {Path(image_path).name}", fontsize=14)
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error: {e}")

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_average_color_per_label(df):
    """
    Computes and plots the average RGB color for each Attention label.

    Args:
        df (pd.DataFrame): DataFrame containing 'Images' (image paths) and 'Attention' (label).
    """
    label_colors = {}

    for label in df['Attention'].unique():
        subset = df[df['Attention'] == label]
        r_total, g_total, b_total = 0, 0, 0
        pixel_count = 0

        for _, row in subset.iterrows():
            image_path = row['Image']
            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")  # Make sure it's RGB
                    img_array = np.array(img)
                    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

                    r_total += r.sum()
                    g_total += g.sum()
                    b_total += b.sum()
                    pixel_count += r.size
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if pixel_count > 0:
            avg_color = (
                int(r_total / pixel_count),
                int(g_total / pixel_count),
                int(b_total / pixel_count),
            )
            label_colors[label] = avg_color

    # Plot average colors
    labels = list(label_colors.keys())
    avg_colors = [label_colors[label] for label in labels]

    fig, ax = plt.subplots(figsize=(2 * len(labels), 4))
    for i, (label, color) in enumerate(zip(labels, avg_colors)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255))
        ax.text(i + 0.5, -0.1, str(label), ha='center', va='top', fontsize=10)
    
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("Average RGB Color per Attention Label")
    plt.show()



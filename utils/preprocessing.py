import cv2
import numpy as np

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

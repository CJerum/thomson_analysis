from scipy.signal import correlate2d
from PIL import Image, ImageEnhance
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import cv2
import pickle
import os 

def find_circles_in_image(gray=None):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Transform to detect ellipses
    ellipses = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=40, param1=20, param2=20, minRadius=10, maxRadius=500)

    # Draw the detected ellipses on the original image
    if ellipses is not None:
        ellipses = np.round(ellipses[0, :]).astype(int)
        for (x, y, r) in ellipses:
            cv2.ellipse(gray, (x, y), (r, r), 0, 0, 360, (0, 255, 0), 2)
    else:
        print("No ellipse in frame: ", i)

    # Display the image with detected ellipses
    cv2.imshow("Elliptical Objects", image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def load_image_parts(m_frame_offsets={}, m_frame=0, m_save_directory="", m_enhance_val=1):

    # Load the target image
    target_image_path = f'{m_save_directory}/part_{m_frame}.jpg'
    target_image = Image.open(target_image_path).convert('L')
    enhancer = ImageEnhance.Contrast(target_image)
    target_image = enhancer.enhance(m_enhance_val)
    target_array = np.array(target_image)

    return target_array

def remove_white_spaces(files=[], im_directory=""):
    for file in files:
        os.rename(im_directory+file, im_directory+file.replace(" ", ""))

def split_image_into_horizontal_parts(image_path, num_parts=12):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    part_height = height // num_parts
    parts = [image[i * part_height:(i + 1) * part_height, :, :] for i in range(num_parts)]
    return parts

def save_parts(directory="", image_parts=[]):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, part in enumerate(image_parts):
        save_path = directory + f'part_{i}.jpg'
        cv2.imwrite(save_path, part)

def compute_shift(ref_array, target_array):
    fft_ref = fft2(ref_array)
    fft_target = fft2(target_array)
    cross_corr = fftshift(ifft2(fft_ref * np.conj(fft_target)))
    max_idx = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    shift_vals = np.array(cross_corr.shape) // 2 - np.array(max_idx)
    return shift_vals[1], shift_vals[0]  #give x shift, y shift

# Function to apply the shift to the target image
def shift_image(image_array, shift_x, shift_y):

    # Create an array filled with zeros
    shifted_image = np.zeros_like(image_array)

    # Calculate the source and destination coordinates
    src_y1 = max(shift_y, 0)
    src_y2 = min(image_array.shape[0] + shift_y, image_array.shape[0])
    dst_y1 = max(-shift_y, 0)
    dst_y2 = min(image_array.shape[0] - shift_y, image_array.shape[0])

    src_x1 = max(shift_x, 0)
    src_x2 = min(image_array.shape[1] + shift_x, image_array.shape[1])
    dst_x1 = max(-shift_x, 0)
    dst_x2 = min(image_array.shape[1] - shift_x, image_array.shape[1])

    if src_y2 > src_y1 and src_x2 > src_x1:
        shifted_image[dst_y1:dst_y2, dst_x1:dst_x2] = image_array[src_y1:src_y2, src_x1:src_x2]

    return shifted_image
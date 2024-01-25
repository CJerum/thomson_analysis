from scipy.signal import correlate2d
from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import cv2
import pickle
def split_image_into_horizontal_parts(image_path, num_parts=12):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    part_height = height // num_parts
    parts = [image[i * part_height:(i + 1) * part_height, :, :] for i in range(num_parts)]
    return parts

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

save_directory = './calibration/'
image_path_to_analyze = './calibration/airforce-test-strip-backlit-greenlaser.tif'

image_parts = split_image_into_horizontal_parts(image_path_to_analyze)
for i, part in enumerate(image_parts):
    save_path = save_directory + f'part_{i}.jpg'
    cv2.imwrite(save_path, part)


image_dir = save_directory


# Load the reference image
ref_image_path = f'{image_dir}/part_0.jpg'
ref_image = Image.open(ref_image_path).convert('L')
ref_array = np.array(ref_image)

aligned_images = [ref_array]

frames_offsets = {}
# # Add these offsets to a python dictionary and save a a pickle
shift_dict = {'x':0, 'y':0}
frames_offsets[0] = shift_dict

# Loop over the remaining images to align them
for i in range(1, 12):
    target_image_path = f'{image_dir}/part_{i}.jpg'
    target_image = Image.open(target_image_path).convert('L')
    target_array = np.array(target_image)

    # Compute the shift needed to align the target image to the reference
    shift_x, shift_y = compute_shift(ref_array, target_array)
    print("shift for image", i, "x,y:",shift_x, shift_y)
    # Apply the shift to align the target image
    aligned_image = shift_image(target_array, shift_x, shift_y)
    aligned_images.append(aligned_image)

    # # Add these offsets to a python dictionary and save a a pickle
    shift_dict = {'x':shift_x, 'y':shift_y}
    frames_offsets[i] = shift_dict
pickle.dump(frames_offsets, open(f'{save_directory}/offsets.pkl', 'wb'))

# save images as gif
gif_path = f'{image_dir}/aligned_images.gif'
aligned_images_pil = [Image.fromarray(img) for img in aligned_images]
aligned_images_pil[0].save(gif_path,
                           save_all=True,
                           append_images=aligned_images_pil[1:],
                           duration=100,  # Duration for each frame in milliseconds
                           loop=0)

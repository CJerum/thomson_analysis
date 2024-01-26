from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def extract_lineout(image_array, pixel_index, n_avg=10, is_horizontal=True):
    if is_horizontal:
        # Extract a horizontal lineout by averaging over rows
        lineout = image_array[pixel_index - n_avg : pixel_index + n_avg, :].mean(axis=0)
    else:
        # Extract a vertical lineout by averaging over columns
        lineout = image_array[:, pixel_index - n_avg : pixel_index + n_avg].mean(axis=1)

    max_value = np.max(lineout)
    half_max_value = max_value / 2
    crossing_points = np.where(np.diff(np.sign(lineout - half_max_value)))[0]

    if len(crossing_points) > 2:
        max_index = np.argmax(lineout)
        closest_points = crossing_points[np.argsort(np.abs(crossing_points - max_index))[:2]]
    else:
        closest_points = crossing_points

    return closest_points

def plot_image_with_all_half_max_points(image_array, half_max_points_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image_array, cmap='gray', aspect="auto")
    ax.set_title("Original Image with All Half-Maximum Points")

    for x, y in half_max_points_list:
        ax.scatter(x, y, color='blue', s=50)
    plt.show()

# Path to the image part_0.jpg
image_path = './1_19_2024_laser_spark/Sequence45(UBSi121HB2062)/part_0.jpg'

# Load the image and convert to grayscale
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Parameters for lineouts
num_lineouts = 35
image_height = image_array.shape[0]
image_width = image_array.shape[1]
interval_h = image_height // num_lineouts
interval_w = image_width // num_lineouts

all_half_max_points = []

# Horizontal lineouts
for i in range(10):
    pixel_row = 555 + i * 5
    half_max_points_x = extract_lineout(image_array, pixel_row, is_horizontal=True)
    for x in half_max_points_x:
        all_half_max_points.append((x, pixel_row))

# Vertical lineouts
for i in range(num_lineouts):
    pixel_column = 570 + i * 7
    half_max_points_y = extract_lineout(image_array, pixel_column, is_horizontal=False)
    for y in half_max_points_y:
        all_half_max_points.append((pixel_column, y))

plot_image_with_all_half_max_points(image_array, all_half_max_points)

# Convert points to a suitable format for OpenCV
points = np.array(all_half_max_points, dtype=np.float32)

# Fit an ellipse using least squares method
if len(points) >= 5:  # fitEllipse requires at least 5 points
    ellipse = cv2.fitEllipse(points)
    
    # Plotting the ellipse
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image_array, cmap='gray', aspect="auto")
    ax.set_title("Ellipse Fitted to Points")

    # Extracting ellipse parameters
    (x_center, y_center), (MA, ma), angle = ellipse

    # Ellipse needs to be rotated and repositioned
    ellipse_patch = matplotlib.patches.Ellipse((x_center, y_center), MA, ma, angle=angle, edgecolor='r', facecolor='none')
    ax.add_patch(ellipse_patch)

    plt.show()
else:
    print("Not enough points to fit an ellipse.")


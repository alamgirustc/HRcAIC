import numpy as np
import os
import cv2
from tqdm import tqdm

# Directories
bbox_folder = r'D:\GeoHol\mscoco\feature\up_down_100_box'
output_folder = r'D:\GeoHol\mscoco\feature\gten_feats'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to normalize the features
def normalize_features(features, img_width, img_height):
    # Normalize x, y, Centroid_x, and Centroid_y by img_width and img_height
    features[:, [0, 7]] = np.clip(features[:, [0, 7]] / img_width, 0, 1)
    features[:, [1, 8]] = np.clip(features[:, [1, 8]] / img_height, 0, 1)

    # Normalize width (w) and height (h)
    features[:, 2] = np.clip(features[:, 2] / img_width, 0, 1)
    features[:, 3] = np.clip(features[:, 3] / img_height, 0, 1)

    # Normalize aspect ratio (already ratio, no need for min-max or z-score normalization)
    features[:, 4] = np.clip(features[:, 4] / features[:, 4].max(), 0, 1) if features[:, 4].max() != 0 else features[:, 4]

    # Normalize area and perimeter by dividing by the maximum possible area and perimeter
    max_area = img_width * img_height
    max_perimeter = 2 * (img_width + img_height)
    features[:, 5] = np.clip(features[:, 5] / max_area, 0, 1)
    features[:, 6] = np.clip(features[:, 6] / max_perimeter, 0, 1)

    # Normalize compactness
    features[:, 9] = np.clip(features[:, 9] / features[:, 9].max(), 0, 1) if features[:, 9].max() != 0 else features[:, 9]

    return features

# Function to extract additional geometric features from bounding boxes
def extract_geometric_features(bbox):
    x, y, w, h = bbox
    aspect_ratio = w / h if h != 0 else 0
    area = w * h
    perimeter = 2 * (w + h)
    centroid_x = x + w / 2
    centroid_y = y + h / 2
    compactness = area / (perimeter ** 2) if perimeter != 0 else 0
    return np.array([x, y, w, h, aspect_ratio, area, perimeter, centroid_x, centroid_y, compactness])

# Function to process all files
def process_files(bbox_folder, output_folder):
    files = [f for f in os.listdir(bbox_folder) if f.endswith('.npy')]

    for file_name in tqdm(files, desc="Processing files", unit="file"):
        try:
            # Load bounding box data
            bbox_path = os.path.join(bbox_folder, file_name)
            bboxes = np.load(bbox_path)

            # Process the original bounding boxes to extract geometric features
            img_width, img_height = 640, 480
            geo_features = np.zeros((bboxes.shape[0], 10))
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                if bbox[2] > 0 and bbox[3] > 0:  # Ensure width and height are valid
                    geo_features[i] = extract_geometric_features(bbox)

            # Normalize the features
            geo_features = normalize_features(geo_features, img_width, img_height)

            # Save final features
            if geo_features.size > 0:
                save_path = os.path.join(output_folder, file_name.replace('.npy', '_gten.npz'))
                np.savez_compressed(save_path, geometric_features=geo_features)
            else:
                print(f"No features generated for {file_name}. Skipping saving.")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Execute the processing
process_files(bbox_folder, output_folder)

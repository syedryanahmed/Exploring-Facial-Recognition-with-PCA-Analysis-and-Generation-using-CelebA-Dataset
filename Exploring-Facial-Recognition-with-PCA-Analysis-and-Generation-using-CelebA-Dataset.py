import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load subset of facial images from the dataset
data_dir = "/Users/syedemadahmed/Downloads/archive/img_align_celeba/img_align_celeba/"
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")][:250]

# Make consistent image size
target_size = (64, 64)

images = []
for img_file in image_files:
    img = Image.open(img_file)
    img = img.resize(target_size)
    img = img.convert('L')  # Convert to grayscale
    img_array = np.array(img).flatten()  # Convert to 1D vector
    images.append(img_array)

images = np.array(images)

# Center the data
mean_face = images.mean(axis=0)
centered_data = images - mean_face

# Perform PCA
pca = PCA(n_components=50)
pca.fit(centered_data)

# Project data onto principal components
proj_data = pca.transform(centered_data)

# Reconstruct faces
reconstructed_faces = pca.inverse_transform(proj_data) + mean_face

# Visualize original and reconstructed faces
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    original_face = images[i].reshape(target_size)
    reconstructed_face = reconstructed_faces[i].reshape(target_size)
    axes[0, i].imshow(original_face, cmap='gray')
    axes[0, i].set_title(f'Original Face {i+1}')
    axes[1, i].imshow(reconstructed_face, cmap='gray')
    axes[1, i].set_title(f'Reconstructed Face {i+1}')
plt.show()

# Evaluate reconstruction accuracy
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(images.reshape(-1), reconstructed_faces.reshape(-1))
print(f"Mean Squared Error (MSE): {mse}")
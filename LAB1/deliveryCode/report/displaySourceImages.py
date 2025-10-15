# take the 6 images in the folder ../assets and display them in a 2x3 grid
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# path to the folder containing the images
path = os.path.join(os.path.dirname(__file__), '..', 'assets')
# list of image filenames
image_filenames = [
    "100.pgm",
    "couchersoleil.pgm",
    "einstein.pgm",
    "mandrill256.pgm",
    "monroe.pgm",
    "peppers.png"
]

# force all the images to be 256x256
images = []
for filename in image_filenames:
    img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    images.append(img_resized)
    
# create a 2x3 grid of images
fig, axs = plt.subplots(2, 3, figsize=(10, 7))
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(image_filenames[i])
    ax.axis('off')
plt.tight_layout()
plt.show()
# save the figure
fig.savefig(os.path.join(os.path.dirname(__file__), 'sourceImages.png'), dpi=300)
plt.close(fig)

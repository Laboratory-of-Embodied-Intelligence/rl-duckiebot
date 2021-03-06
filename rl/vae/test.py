"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds

from vae.controller import VAEController

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=20)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = VAEController()
vae.load(args.vae_path)

images = [im for im in os.listdir(args.folder) if im.endswith('.jpg') or im.endswith('.png')]
images = np.array(images)
n_samples = len(images)


for i in range(args.n_samples):
    # Load test image

    image_idx = np.random.randint(n_samples)
    image_path = args.folder + images[image_idx]
    image = cv2.imread(image_path)
    image = image[0:][156:]
    im = cv2.resize(image,(160, 80))
    encoded = vae.encode_from_raw_image(im)
    print(im.dtype)
    reconstructed_image = vae.decode(encoded)[0]
    print(reconstructed_image.shape)
    # Plot reconstruction
    cv2.imshow("Original", image)
    cv2.imshow("Reconstruction", reconstructed_image)
    cv2.waitKey(0)

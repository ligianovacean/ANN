import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from pathlib import Path
from glob import glob

NR_IMAGES = 1368
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
NR_CLASSES = 10


def preprocess_image(src):
    dst = src.copy()

    # Resize image
    dims = (IMAGE_WIDTH, IMAGE_HEIGHT)
    dst = cv2.resize(dst, dims, interpolation = cv2.INTER_NEAREST) 

    return dst


def compute_mse(img1, img2):
    return np.mean((img1 - img2)**2)


def compute_psnr(img1, img2):
    return 10 * np.log10((255 * 255) / compute_mse(img1, img2))


def compute_covariance(img1, img2):
    return np.cov(img1.reshape(img1.shape[0]*img1.shape[1]*3))


def compute_dot_product(img1, img2):
    return np.dot(img1.reshape(img1.shape[0]*img1.shape[1]*3), img2.reshape(img2.shape[0]*img2.shape[1]*3))


def compute_correlation(img1, img2):
    return np.corrcoef(img1.reshape(img1.shape[0]*img1.shape[1]*3), img2.reshape(img2.shape[0]*img2.shape[1]*3))[0, 1]


if __name__ == "__main__":
    all_images = np.zeros((10, 130, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for k in range(NR_CLASSES):
        image_paths = glob(f'./monkeys_dataset/n{k}/*.jpg')
        images = np.zeros((len(image_paths), IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        

        for i in range(len(image_paths)):
            img = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
            images[i] = preprocess_image(img)

        all_images[k] = images[:130]

    correlation_matrix = np.zeros((NR_CLASSES, NR_CLASSES))
    for i in range(NR_CLASSES):
        for j in range(NR_CLASSES):
            correlation = 0.0
            count = 0
            for l in range(130):
                for k in range(130):
                    img1 = all_images[i, l]
                    img2 = all_images[j, k]

                    correlation += compute_correlation(img1, img2)
                    count += 1

            correlation_matrix[i, j] = round(correlation / count, 4)

            print(f"({i}, {j}): {correlation/count}\n")

    print(correlation_matrix)

    plt.figure(figsize = (10,7))
    df_cm = pd.DataFrame(np.abs(correlation_matrix), index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
    sn.heatmap(df_cm, annot=True)

    plt.savefig(f"correlation_matrix.png")
    plt.close()



                
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.io
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="This script creates database for training from the UTKFace dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to the UTKFace image directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_dir = Path(args.input)
    output_path = args.output
    img_size = args.img_size

    out_ages = []
    out_imgs = []

    age_list = open(Path(image_dir, "list/train_age.txt"))
    pic_list = open(Path(image_dir, "list/train_name.txt"))

    for image_name, age in tqdm(zip(pic_list, age_list)):
        image_name = image_name.strip()
        age = age.strip()
        image_path = Path(image_dir, "train", image_name).resolve()
        out_ages.append(min(int(age), 100))
        img = cv2.imread(str(image_path))
        img = img[30:188,10:-10]
        out_imgs.append(cv2.resize(img, (img_size, img_size)))

    output = {"image": np.array(out_imgs), "age": np.array(out_ages),
              "db": "megaage", "img_size": img_size, "min_score": -1}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()

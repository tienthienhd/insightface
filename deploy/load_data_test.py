import glob
import face_model
import argparse
import cv2
import sys
import numpy as np


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='pretrain_model/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

test_images = glob.glob('../../../vn_celeb_face_recognition/test/*.png')
n_per_file = 5000

images = []

for i, img_path in enumerate(test_images):
    img = cv2.imread(img_path)
    img_input = model.get_input(img)
    f1 = model.get_feature(img_input)
    images.append(f1)
    if i % n_per_file == 0:
        file_name = '{}s_to_{}'.format(n_per_file, i)
        np.save(file_name, np.array(images))
        images.clear()

if len(images) > 0:
    file_name = '{}s_to_{}'.format(n_per_file, len(test_images))
    np.save(file_name, np.array(images))
    images.clear()
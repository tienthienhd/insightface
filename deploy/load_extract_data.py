import pandas as pd 
import face_model
import argparse
import cv2
import sys
import numpy as np



train = pd.read_csv('/content/data/train.csv')
# print(train.head())



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


images = []
labels = []

for i,row in train.iterrows():
    image_path = '/content/data/train/' + row['image']
    label = row['label']
    if i % 100 == 0:
        print('Extracted {} image'.format(i))

    try:
        img = cv2.imread(image_path)

        img_input = model.get_input(img)
        f1 = model.get_feature(img_input)
        images.append(f1)
        labels.append(label)
    except Exception:
        print(image_path, label)

np.save('/content/pretrain_model/features', np.array(images))
np.save('/content/pretrain_model/labels', np.array(labels))

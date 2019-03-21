from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import pandas as pd 
import glob
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])

def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model

class FaceModel:
    def __init__(self, args):
        self.args = args 
        ctx = mx.cpu()
        _vec = args.image_size.split(',')
        assert len(_vec)==2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        self.ga_model = None
        if len(args.model)>0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')
        if len(args.ga_model)>0:
            self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

        self.threshold = args.threshold
        self.det_minsize = 50
        self.det_threshold = [0.6,0.7,0.8]
        #self.det_factor = 0.9
        self.image_size = image_size
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        if args.det==0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
        self.detector = detector

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return np.transpose(cv2.cvtColor(cv2.resize(face_img, (112, 112)), cv2.COLOR_BGR2RGB), (2,0,1))
        bbox, points = ret
        if bbox.shape[0]==0:
            return np.transpose(cv2.cvtColor(cv2.resize(face_img, (112, 112)), cv2.COLOR_BGR2RGB), (2,0,1))
        bbox = bbox[0,0:4]
        points = points[0,:].reshape((2,5)).T
        nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        return aligned

    def get_batch_input(self, imgs):
        batch = []
        for img in imgs:
            img = self.get_input(img)
            batch.append(img)
        return np.array(batch)

    def get_features(self, imgs):
        imgs = self.get_batch_input(imgs)
        # input_blob = np.expand_dims(imgs, axis=0)
        # print('input blob:',input_blob.shape)
        data = mx.nd.array(imgs)
        print('batch_data:', data.shape)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        print('batch_embeding:',embedding.shape)
        embedding = sklearn.preprocessing.normalize(embedding)
        return embedding

    def get_features_train(self, folder_image, train_csv, batch_size=64):
        train = pd.read_csv(train_csv)
        images = []
        features = None
        labels = []
        for i,row in train.iterrows():
            image_path = folder_image + row['image']
            label = row['label']

            img = cv2.imread(image_path)
            # print(img.shape)
            images.append(img)
            labels.append(label)

            if (i + 1) % batch_size == 0:
                feature_batch = self.get_features(images)
                if i < batch_size:
                    features = feature_batch
                else:
                    features = np.concatenate((features, feature_batch), axis=0)
                images.clear()
        feature_batch = self.get_features(images)
        if len(images) > 0:
            if len(train) > batch_size:
                features = np.concatenate((features, feature_batch), axis=0)
            else:
                features = self.get_features(images)
        images.clear()

        labels = np.array(labels)
        return features, labels

    def get_features_test(self, folder_image, batch_size=64):
        test_paths = glob.glob(folder_image + '*.png')
        images = []
        file_names = []
        features = None
        for i, img_path in enumerate(test_paths):
            img_name = os.path.basename(img_path)
            file_names.append(img_name)
            img = cv2.imread(img_path)
            images.append(img)

            if (i + 1) % batch_size == 0:
                feature_batch = self.get_features(images)
                if i < batch_size:
                    features = feature_batch
                else:
                    features = np.concatenate((features, feature_batch), axis=0)
                images.clear()
        feature_batch = self.get_features(images)
        if len(images) > 0:
            if len(test_paths) > batch_size:
                features = np.concatenate((features, feature_batch), axis=0)
            else:
                features = self.get_features(images)
        images.clear()
        return features, file_names







parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/content/pretrain_model/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--data_path', default='/content/data/')
parser.add_argument('--embedding_folder', default='/content/embedding/')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

a = FaceModel(args)

features, labels = a.get_features_train(args.data_path + 'train/', args.data_path + 'train.csv', batch_size=args.batch_size)
print(features.shape, labels.shape)
np.save(args.embedding_folder + 'features_train', features)
np.save(args.embedding_folder + 'labels_train', labels)
del features
del labels


features, file_names = a.get_features_test(args.data_path + 'test/', batch_size=args.batch_size)
print(features.shape, len(file_names))
np.save(args.embedding_folder + 'features_test', features)
with open(args.embedding_folder + 'file_names_test.txt', 'w') as f:
    for file_name in file_names:
        f.write('{}\n'.format(file_name))

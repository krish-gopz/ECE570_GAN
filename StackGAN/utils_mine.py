# -*- coding: utf-8 -*-

# Utilities used during development

import pickle
import cv2
import numpy as np

def image_from_pickle(filename, index):
    # View index-th image from pickle data pointed to by filename
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)
    print(len(data))
    cv2.imshow('image',data[index])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1


def load_image_data(filename):
    # View index-th image from pickle data pointed to by filename
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)
    data = np.array(data)
    return data


def emb_from_pickle(filename):
    with open(filename, 'rb') as fid:
        embeddings = pickle.load(fid, encoding='latin1')
    # print(type(embeddings))
    embeddings = np.array(embeddings)
    # print(embeddings.shape)
    # print(embeddings[0])
    return embeddings
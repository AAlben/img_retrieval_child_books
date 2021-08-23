import cv2
import pdb
import time
import faiss
import random
import base64
import pickle
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from loguru import logger
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import timm
import torch
from torch import nn, optim
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

import grpc
from concurrent import futures
import image_classifier_pb2
import image_classifier_pb2_grpc

from utils.crow import Crow
from utils.spoc import SPoC
from utils.r_mac import RMAC


def get_title_transforms(normalize_mean, normalize_std, resize=256):
    transform = A.Compose([
        A.SmallestMaxSize(max_size=resize),
        A.CenterCrop(height=resize - 32, width=resize - 32),
        A.Normalize(mean=normalize_mean, std=normalize_std,),
        ToTensorV2()
    ])
    return transform


def get_retrieval_transforms(normalize_mean, normalize_std, resize=384):
    transform = A.Compose([
        A.SmallestMaxSize(max_size=400),
        A.CropAndPad(percent=(-0.14, 0, -0.3, 0)),
        A.Resize(resize - 32, resize - 32),
        A.Normalize(mean=normalize_mean, std=normalize_std,),
        ToTensorV2()
    ])
    return transform


def init_labels(LABELS_D_PATH):
    labels_d, id_d = {}, {}
    with open(LABELS_D_PATH, 'rb') as f:
        labels_d = pickle.load(f)

    id_d = dict(zip(labels_d.values(), labels_d.keys()))
    return labels_d, id_d


def get_pooling_fn():
    GAP = lambda features: features.mean(dim=3).mean(dim=2)
    GMP = lambda features: (features.max(dim=3)[0]).max(dim=2)[0]

    default_hyper_params = {
        "spatial_a": 2.0,
        "spatial_b": 2.0,
    }
    crow = Crow(default_hyper_params)

    def gem(features):
        fea = features
        p = 3.0
        fea = fea ** p
        h, w = fea.shape[2:]
        fea = fea.sum(dim=(2, 3)) * 1.0 / w / h
        fea = fea ** (1.0 / p)
        return fea

    default_hyper_params = {
        "level_n": 3,
    }
    rmac = RMAC(default_hyper_params)

    default_hyper_params = {
        "level_n": 3,
    }
    rmac = RMAC(default_hyper_params)

    spoc = SPoC()

    return GAP, GMP, crow, gem, rmac, spoc


def reduction(features, pca, fited=False):
    features = np.nan_to_num(features)
    X = normalize(features, norm='l2')
    if not fited:
        pca.fit(X)
    X = pca.transform(X)
    X = normalize(X, norm='l2')
    return X, pca


class Greeter(image_classifier_pb2_grpc.GreeterServicer):

    def __init__(self, transforms_, models, files_d, SAVE_PHOTO_PATH, logger, feat_act, ACTIVATION_KEYS, RETRIEVAL_PATH):
        super(image_classifier_pb2_grpc.GreeterServicer, self).__init__()
        self.title_transform, self.retrieval_transform = transforms_
        self.title_model, self.retrieval_model = models
        self.files_d = files_d
        self.logger = logger
        self.SAVE_PHOTO_PATH = SAVE_PHOTO_PATH
        self.feat_act = feat_act
        self.ACTIVATION_KEY_1, self.ACTIVATION_KEY_2 = ACTIVATION_KEYS
        self.resize_transform_1 = A.SmallestMaxSize(max_size=400)
        self.resize_transform_2 = nn.Upsample(size=(22, 22), mode='nearest')
        self.RETRIEVAL_PATH = RETRIEVAL_PATH

        GAP, GMP, crow, gem, rmac, spoc = get_pooling_fn()
        self.pooling_fns = [GMP, gem]
        self.pooling_keys = [None, None]

        self.load_pickle()

    def Classify(self, request, context):
        result = {'code': -1,
                  'message': None,
                  'book_name': None,
                  'page_num': None,
                  'is_last_page': False,
                  'next_page_num': -1}

        logger.info('receive classify photo')

        try:
            buffer = request.content
            decoded = base64.b64decode(buffer)
            img_np = np.frombuffer(decoded, dtype=np.uint8)
            img_np = cv2.imdecode(img_np, flags=1)

            resize_img = self.resize_transform_1(image=img_np)['image']
            receive_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')
            cv2.imwrite(str(SAVE_PHOTO_PATH / 'title' / f'{receive_time}.jpg'), resize_img)

            with torch.no_grad():
                image = self.title_transform(image=img_np)['image']
                image = image.to(device)[None]
                output = self.title_model(image)
                predict = torch.argmax(output, 1)
                index = predict.cpu().numpy().item()

            parts = self.files_d[index].split('/')
            result['code'] = 1
            result['book_name'] = parts[0]
            result['page_num'] = 0
            logger.info(f'img_shape = {img_np.shape}; book_name = {parts[0]}; page_num = {result["page_num"]}; file = {receive_time}.jpg')
        except Exception as e:
            logger.info(f'exception_type = {type(e)}; error_msg = {traceback.format_exc()}')
            result['message'] = str(type(e))
            result['page_num'] = -1

        return image_classifier_pb2.ExecuteReply(book_name=result['book_name'],
                                                 page_num=result['page_num'],
                                                 code=result['code'],
                                                 message=result['message'],
                                                 executed_time=receive_time,
                                                 user_id=request.user_id,
                                                 is_last_page=result['is_last_page'])

    def Search(self, request, context):
        result = {'code': -1,
                  'message': None,
                  'book_name': request.book_name,
                  'page_num': None,
                  'is_last_page': None,
                  'next_page_num': -1}

        logger.info('receive search photo')

        try:
            where = np.where(self.book_names == request.book_name)
            page_nums = self.page_nums[where]
            buffer = request.content
            decoded = base64.b64decode(buffer)
            img_np = np.frombuffer(decoded, dtype=np.uint8)
            img_np = cv2.imdecode(img_np, flags=1)

            resize_img = self.resize_transform_1(image=img_np)['image']
            receive_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')
            cv2.imwrite(str(SAVE_PHOTO_PATH / 'retrieval' / f'{receive_time}.jpg'), resize_img)

            with torch.no_grad():
                image = self.retrieval_transform(image=img_np)['image']
                image = image.to(device)[None]
                output = self.retrieval_model(image)

                features_7_7 = self.feat_act[self.ACTIVATION_KEY_1]
                features_7_7 = self.resize_transform_2(features_7_7)
                features = torch.cat((features_7_7, self.feat_act[ACTIVATION_KEY_2]), 1)

                where_addition = np.where(self.addition_book_names == request.book_name)
                where_pages, next_page, last_page = self.gallery_limit(page_nums, request.previous_page_num)
                result['next_page_num'] = int(next_page.split('.')[0])

                predict, dist = self.loop_search(where, where_pages, where_addition, features, page_nums[where_pages])

            is_last_page = predict == last_page
            if dist >= 1.5:
                result['page_num'] = None
                result['code'] = -1
            else:
                result['page_num'] = int(predict.split('.')[0])
                result['code'] = 1
            result['book_name'] = request.book_name
            result['is_last_page'] = is_last_page
            logger.info(f'img_shape = {img_np.shape}; book_name = {request.book_name}; previous_page_num = {request.previous_page_num}; page_num = {predict}; result_page_num = {result["page_num"]}; dist = {dist:5.4}; file = {receive_time}.jpg')
        except Exception as e:
            logger.info(f'exception_type = {type(e)}; error_msg = {traceback.format_exc()}')
            result['message'] = str(type(e))
            result['page_num'] = -1

        return image_classifier_pb2.ExecuteReply(book_name=result['book_name'],
                                                 page_num=result['page_num'],
                                                 code=result['code'],
                                                 message=result['message'],
                                                 executed_time=receive_time,
                                                 user_id=request.user_id,
                                                 is_last_page=result['is_last_page'])

    def gallery_limit(self, page_nums, previous_page_num):
        unique_page = set(page_nums)
        unique_pages = [int(item.rstrip('.jpg')) for item in unique_page]

        not_in_flag = False
        if previous_page_num not in unique_pages:
            unique_pages.append(previous_page_num)
            not_in_flag = True

        unique_pages = sorted(unique_pages, key=lambda x: x)
        next_page = previous_page_num if previous_page_num == unique_pages[-1] else unique_pages[unique_pages.index(previous_page_num) + 1]
        next_page = f'{next_page}.jpg'
        page_index = -1 if previous_page_num == 0 else unique_pages.index(previous_page_num)
        if not_in_flag:
            unique_pages.remove(previous_page_num)
        last_page = f'{unique_pages[-1]}.jpg'

        if page_index <= 2:
            gallery_pages = unique_pages[:5]
        elif page_index >= len(unique_pages) - 3:
            gallery_pages = unique_pages[-5:]
        else:
            gallery_pages = unique_pages[page_index - 2:page_index + 3]

        l = []
        for galley_page in gallery_pages:
            page = f'{galley_page}.jpg'
            l.append(np.where(page_nums == page))
        where_pages = np.append(l[0], l[1:])

        return where_pages, next_page, last_page

    def loop_search(self, where, where_pages, where_addition, query_features, page_nums):
        dist_d = {0: {}, 1: {}, 2: {}}
        addition_d = {}
        for transform_gallery_Xs, addition_X, pca, pooling_fn, pooling_key in zip(self.gallery_Xs, self.addition_Xs, self.pcas, self.pooling_fns, self.pooling_keys):
            if where_addition[0].size > 0:
                addition_d = self.search_fn(query_features, pca, pooling_fn, pooling_key, addition_X[where_addition], self.addition_page_nums[where_addition], addition_d)
            for i, transform_gallery_X in enumerate(transform_gallery_Xs):
                dist_d[i] = self.search_fn(query_features, pca, pooling_fn, pooling_key, transform_gallery_X[where][where_pages], page_nums, dist_d[i])

        if where_addition[0].size > 0:
            sorted_0 = sorted(addition_d.items(), key=lambda x: np.mean(x[1]['dist']))
            dist_0 = [(item[0], np.min(item[1]['dist']), np.mean(item[1]['dist']), np.std(item[1]['dist']), len(item[1]['dist'])) for item in sorted_0]
            predict_0 = sorted_0[0][0]
            mean_dist = dist_0[0][1]
            if mean_dist < 0.5:
                predict = predict_0
                dist = mean_dist
                return predict, dist

        sorted_1 = sorted(dist_d[0].items(), key=lambda x: np.mean(x[1]['dist']))
        sorted_2 = sorted(dist_d[1].items(), key=lambda x: np.mean(x[1]['dist']))
        sorted_3 = sorted(dist_d[2].items(), key=lambda x: np.mean(x[1]['dist']))

        dist_1 = [(item[0], np.min(item[1]['dist']), np.mean(item[1]['dist']), np.std(item[1]['dist']), len(item[1]['dist'])) for item in sorted_1]
        dist_2 = [(item[0], np.min(item[1]['dist']), np.mean(item[1]['dist']), np.std(item[1]['dist']), len(item[1]['dist'])) for item in sorted_2]
        dist_3 = [(item[0], np.min(item[1]['dist']), np.mean(item[1]['dist']), np.std(item[1]['dist']), len(item[1]['dist'])) for item in sorted_3]

        predict, dist = self.logic_1(dist_1, dist_2, dist_3)
        return predict, dist

    def search_fn(self, query_features, pca, pooling_fn, pooling_key, gallery_X, page_nums, top_1_d):
        if pooling_key:
            global_features = pooling_fn(query_features)[pooling_key]
        else:
            global_features = pooling_fn(query_features)

        global_features = global_features.cpu().numpy()
        global_features = np.nan_to_num(global_features)
        X = normalize(global_features, norm='l2')
        X = pca.transform(X)
        X = normalize(X, norm='l2')
        query_X = X

        index = faiss.IndexFlatL2(gallery_X.shape[1])
        index.add(gallery_X)
        D, I = index.search(query_X.reshape(1, -1), 5)

        for i in range(5):
            top_1 = page_nums[I[0][i]]
            d = top_1_d.get(top_1, {'num': 0, 'dist': []})
            d['num'] = d['num'] + 1
            d['dist'].append(D[0][i])
            top_1_d[top_1] = d

        return top_1_d

    def logic_1(self, dist_1, dist_2, dist_3):
        d = {}
        for items in [dist_1[:2], dist_2[:2], dist_3[:2]]:
            for item in items:
                d_std = d.get(item[0], {'std': [], 'mean': []})
                d_std['std'].append(item[3])
                d_std['mean'].append(item[2])
                d[item[0]] = d_std
        dd = {}
        for k, v in d.items():
            if len(v['std']) == 1:
                continue
            dd[k] = v
        sorted_ = sorted(dd.items(), key=lambda x: np.mean(x[1]['std']))
        var_d = {k: {'std': np.mean(v['std']), 'mean': np.mean(v['mean'])} for k, v in d.items()}

        top_1_1, top_1_2, top_1_3 = dist_1[0][0], dist_2[0][0], dist_3[0][0]
        top_2_1, top_2_2, top_2_3 = dist_1[1][0], dist_2[1][0], dist_3[1][0]
        if len(set([top_1_1, top_1_2, top_1_3])) == 1:
            return top_1_1, var_d[top_1_1]['mean']

        top_1s, top_2s = [], []
        for item in [top_1_1, top_1_2, top_1_3]:
            if item in var_d.keys() and var_d[item]['mean'] < 1.4:
                top_1s.append(item)
        for item in [top_2_1, top_2_2, top_2_3]:
            if item in var_d.keys() and var_d[item]['mean'] < 1.4:
                top_2s.append(item)

        if len(set(top_1s)) == 1:
            return top_1s[0], var_d[top_1s[0]]['mean']

        if len(top_1s) >= 1 and set(top_1s) == set(top_2s):
            count_d = {}
            for item in set(top_1s):
                i = count_d.get(item, 0)
                i += 1
                count_d[item] = i
            sorted_1 = sorted(count_d.items(), key=lambda x: x[1], reverse=True)
            if var_d[sorted_1[0][0]]['std'] - var_d[sorted_1[1][0]]['std'] <= 0.05:
                return sorted_1[0][0], var_d[sorted_1[0][0]]['mean']

        return sorted_[0][0], var_d[sorted_[0][0]]['mean']

    def load_pickle(self, i=2):
        with open(RETRIEVAL_PATH / f'pcas_{i}.pickle', 'rb') as f:
            self.pcas = pickle.load(f)
        with open(RETRIEVAL_PATH / f'gallery_Xs_{i}.pickle', 'rb') as f:
            self.gallery_Xs = pickle.load(f)
        with open(RETRIEVAL_PATH / f'book_names_{i}.pickle', 'rb') as f:
            self.book_names = pickle.load(f)
        with open(RETRIEVAL_PATH / f'page_nums_{i}.pickle', 'rb') as f:
            self.page_nums = pickle.load(f)
        with open(RETRIEVAL_PATH / f'addition_Xs_{i}.pickle', 'rb') as f:
            self.addition_Xs = pickle.load(f)
        with open(RETRIEVAL_PATH / f'addition_book_names_{i}.pickle', 'rb') as f:
            self.addition_book_names = pickle.load(f)
        with open(RETRIEVAL_PATH / f'addition_page_nums_{i}.pickle', 'rb') as f:
            self.addition_page_nums = pickle.load(f)


def get_activation(name):
    def hook(model, input, output):
        feat_act[name] = output.detach()
    return hook


def serve(transforms_, models, files_d, SAVE_PHOTO_PATH, looger, feat_act, ACTIVATION_KEYS, RETRIEVAL_PATH):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    image_classifier_pb2_grpc.add_GreeterServicer_to_server(Greeter(transforms_, models, files_d, SAVE_PHOTO_PATH, looger, feat_act, ACTIVATION_KEYS, RETRIEVAL_PATH), server)
    server.add_insecure_port('[::]:5000')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    PRJ_PATH = Path('/root/code/img_retrieval_child_books')
    SAVE_PHOTO_PATH = PRJ_PATH / 'photos'
    MODEL_SAVE_PATH = PRJ_PATH / 'model_state'
    RPC_PATH = PRJ_PATH / 'rpc'
    PICKLE_PATH = RPC_PATH / 'pickles'
    LOG_PATH = PRJ_PATH / 'log' / 'grpc_service.log'
    logger.add(LOG_PATH, rotation="1 day")

    LABELS_D_TITLE_PATH = PICKLE_PATH / 'labels_d_v1.pickle'
    LABELS_D_RETRIEVAL_PATH = PICKLE_PATH / 'labels_d_all.pickle'

    TITLE_MODEL_STATE_PATH = MODEL_SAVE_PATH / 'efficientnet_v2_b0_0805_label_smoothing_94.pth'
    i = 2
    RETRIEVAL_MODEL_STATE_PATH = MODEL_SAVE_PATH / f'efficientnet_v2_s_0817_3_retrieval_centerloss_{i}.pth'

    normalize_mean, normalize_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    title_transform = get_title_transforms(normalize_mean, normalize_std)
    retrieval_transform = get_retrieval_transforms(normalize_mean, normalize_std)

    labels_d_title, files_d_title = init_labels(LABELS_D_TITLE_PATH)
    NUM_TITLES = len(labels_d_title.keys())

    labels_d_retrieval, files_d_retrieval = init_labels(LABELS_D_RETRIEVAL_PATH)
    NUM_PAGES = len(labels_d_retrieval.keys())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    title_model = timm.create_model('tf_efficientnetv2_b0')
    classifier_in_features = title_model.classifier.in_features
    title_model.classifier = torch.nn.Linear(in_features=classifier_in_features, out_features=NUM_TITLES, bias=True)
    title_model.load_state_dict(torch.load(str(TITLE_MODEL_STATE_PATH)))
    title_model.to(device)
    title_model.eval()

    retrieval_model = timm.create_model('tf_efficientnetv2_s')
    classifier_in_features = retrieval_model.classifier.in_features
    retrieval_model.classifier = torch.nn.Linear(in_features=classifier_in_features, out_features=NUM_PAGES, bias=True)
    retrieval_model.load_state_dict(torch.load(str(RETRIEVAL_MODEL_STATE_PATH)))
    retrieval_model.to(device)
    retrieval_model.eval()

    feat_act = {}
    ACTIVATION_KEY_1, ACTIVATION_KEY_2 = 'act2', 'b_4_8_act2'
    retrieval_model.act2.register_forward_hook(get_activation(ACTIVATION_KEY_1))
    retrieval_model.blocks[4][8].act2.register_forward_hook(get_activation(ACTIVATION_KEY_2))

    serve([title_transform, retrieval_transform], [title_model, retrieval_model], files_d_title, SAVE_PHOTO_PATH, logger, feat_act, [ACTIVATION_KEY_1, ACTIVATION_KEY_2], PICKLE_PATH)

from models import BERTJudger
from models.Judger import Judger
from tqdm import tqdm
import math
from transformers import AutoConfig,  AutoModel, AutoTokenizer
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from BERT import get_mean_representation_from_sentence
from distance import  cosine_distance,canberra,directed_hausdorff,euclidean,jsd_kmean,jsd_mixture
from router import calculate_features
class Trainer(BERTJudger):
    def __init__(self,threshold = None, corpus1=None, corpus2=None, bert_path=".cache/bert-multilingual"):
        super(Trainer,self).__init__(threshold,corpus1,corpus2,bert_path)
        self.bert_path = bert_path
        self.bert_zoo = None

    def generate_features(self, target_file, k=-1):
        words, scores = self.read_scores(target_file,is_int=False)
        X ,y = [],[]
        for i,word in enumerate(words):
            raw_features = self.get_features(word)
            if raw_features is None:
                continue
            if k==-1:
                features = calculate_features(raw_features[0], raw_features[1], k=-1)
                distances = self.get_distances(raw_features[0], raw_features[1])
                X.append(np.concatenate([features, distances]))
                y.append(scores[i])
            else:
                for i in range(5):
                    features = calculate_features(raw_features[0],raw_features[1],k=-1)
                    distances = self.get_distances(raw_features[0],raw_features[1])
                    X.append(np.concatenate([features,distances]))
                    y.append(scores[i])
        return X,y

    def generate_features_from_multiple_files(self,datas):
        Xs,ys= [], []
        for data in datas:
            self.set_corpora(data.corpus1, data.corpus2)
            X, y = self.generate_features(data.graded)
            print("{} with {} features with {} dimension".format(data.name, len(X), len(X[0])))
            Xs.append(X)
            ys.append(y)
        return  Xs,ys

    def get_distances(self,x,y):
        distance1 = cosine_distance(np.mean(x,0),np.mean(y,0)) * -1
        distance2 = canberra(np.mean(x, 0), np.mean(y, 0))
        distance3 = euclidean(np.mean(x, 0), np.mean(y, 0))

        distance4,_,_ = directed_hausdorff(x,y)
        distance5 = jsd_kmean(x,y)
        distance6 = jsd_mixture(x, y)
        return np.array([distance1,distance2,distance3,distance4,distance5,distance6])









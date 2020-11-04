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
from sys import platform

def do_visuallize(x,y,k= 100,word=None):
    x,y = np.array(x),np.array(y)
    if k!=-1:
        selected_x = np.random.choice(len(x), size=k, replace=True)
        selected_y = np.random.choice(len(y), size=k, replace=True)
        x,y = x[selected_x], y[selected_y]
    # a = np.matmul(x,y.T)
    a=[]
    for x0 in x:
        a0=[]
        for y0 in y:
            a0.append(cosine_distance(x0,y0))
        a.append(np.array(a0))
    a= np.array(a)
    a = a[:, np.argsort(a.sum(axis=0))[::-1]]
    a = a[np.argsort(a.sum(axis=1))[::-1],:]
    print(a)
    plt.imshow(a,vmin=0, vmax=1735680) #,vmin=0, vmax=1
    plt.savefig("attention/{}.png".format(word))
class BERTJudger(Judger):
    def __init__(self,threshold = None, corpus1=None, corpus2=None, bert_path=".cache/bert-multilingual"):
        super(BERTJudger,self).__init__(threshold,corpus1,corpus2)
        self.bert_path = bert_path
        self.bert_zoo = None

    def get_bert_model(self):
        if self.bert_zoo is None:
            config = AutoConfig.from_pretrained(self.bert_path)
            tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            model =  AutoModel.from_pretrained(self.bert_path,config=config)
            if torch.cuda.is_available():
                model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            model.eval()
            self.bert_zoo = model,tokenizer,config
        return self.bert_zoo


    def get_features(self,word,save_feature = True):
        features = []
        if self.bert_path == ".cache/bert-multilingual":
            pkl_filename = "pkl/{}.pik".format(word)
        else:
            path = "pkl/" + self.bert_path.split("/")[-1]
            if not os.path.exists(path):
                os.mkdir(path)
            pkl_filename = "{}/{}.pik".format(path, word)
        if os.path.exists(pkl_filename):
            features = np.load(open(pkl_filename, "rb"), allow_pickle=True)
            # print("{} exists".format(pkl_filename))
        else:
            print(os.listdir("pkl"))
            print("{} not exists".format(pkl_filename))
            model, tokenizer, config = self.get_bert_model()
            for sentences in self.get_cached_sentences(word):
                print(word)
                print(len(sentences))
                if len(sentences) == 0:
                    print("no sentence was given for {}".format(word))
                    features.append(None)
                    return None
                features.append(get_mean_representation_from_sentence(word, sentences, model, tokenizer,
                                                                      output_representation=True))
            if save_feature:
                if not os.path.exists("pkl"):
                    os.mkdir("pkl")
                np.save(open(pkl_filename, "wb"), features)
        if features[0] is None or features[1] is None:
            return None
        return features

    def draw(self,word,features):
        img_filname = "img/{}.jpg".format(word)
        if os.path.exists(img_filname):
            return
        from matplotlib import pyplot as plt
        from sklearn.manifold import TSNE
        model = TSNE(n_components=2)
        size1 = len(features[0])
        size2 = len(features[1])
        if size1 >2000:
            features[0] = features[0][:2000]
            size1 = 2000
        if size2 > 2000:
            features[1] = features[1][:2000]
            size2 = 2000
        print(size1,size2)
        print(features)
        embedding1 = model.fit_transform(np.concatenate(features,0))
        embedding2 = embedding1[size1:]
        embedding1 = embedding1[:size1]
        plt.plot(embedding1[:,0],embedding1[:,1],'bx')
        plt.plot(embedding2[:,0],embedding2[:,1],'r+')
        plt.xlabel(word)
        # plt.show()
        plt.savefig(img_filname,bbox_inches='tight')

        plt.close()

    def _get_score(self,word,save_feature = True, do_draw = False):

        features = self.get_features(word,save_feature)

        if  platform != "linux" and platform != "linux2" and do_draw:# not os.path.exists(img_filname) and
            self.draw(word,features)
        try:
            # distance = cosine_distance(np.mean(features[0],0),np.mean(features[1],0)) * -1
            # distance = canberra(np.mean(features[0], 0), np.mean(features[1], 0))
            # distance = euclidean(np.mean(features[0], 0), np.mean(features[1], 0))

            # distance,_,_ = directed_hausdorff(features[0],features[1])
            # distance = jsd_kmean(features[0],features[1])
            distance = jsd_mixture(features[0], features[1])
        except Exception  as e:
            print(e)
            return  1000
        # do_visuallize(features[0],features[1],k=-1,word=word)
        # print("{0:.3f}".format(distance))
        # print("{0:} & {1:} & {2:} & {3:.3f} \\\\".format(word,size1,size2,distance))
        # print("{0:} & {1:.3f} \\\\".format(word, distance))
        return  distance











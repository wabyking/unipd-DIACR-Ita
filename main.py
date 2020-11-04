import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import random
from sklearn.metrics import accuracy_score
import os
import numpy as np
from models import Judger,BERTJudger,CountJudger,Trainer
from datahelper import  datasets,it
import subprocess
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,Ridge,Perceptron
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
MODEL =  BERTJudger # CountJudger # #Judger
import  pickle
# MODEL =
def submit(data=it):
    model = MODEL()
    model.set_corpora(data.corpus1, data.corpus2)
    output_file = model.predict(data.target,is_bool=True)
    # os.system("python  accuracy.py {} {}".format(data.truth, output_file))
    proc = subprocess.Popen(["python", "accuracy.py", data.truth, output_file], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print(out)

def submit_semeval( path ="answer"):
    for data in datasets[:1]:
        print("processing {}".format(data.name))
        model = MODEL()#bert_path=".cache/bert-multilingual-it-xxl"
        model.set_corpora(data.corpus1, data.corpus2)
        output_file = model.predict(data.target, output_file=data.name, path = path)

def test_model():
    model = MODEL()
    for data in datasets[:1]:
        print("processing {} with {} & {} ".format(data.name,data.corpus1,data.corpus1))
        model.set_corpora(data.corpus1, data.corpus2)
        score = model.get_acc_by_files_with_testing(data.truth,data.graded)
        print(score)


def train_and_predict():
    model = Trainer()
    filename = "Xy.pkl"
    if not os.path.exists(filename):
        Xs,ys =model.generate_features_from_multiple_files(datasets[2:])
        pickle.dump([Xs,ys], open(filename, "wb"))
    else:
        Xs,ys = pickle.load(open(filename, "rb"))
    Xs,ys = np.array(Xs),np.array(ys)

    kf = KFold(n_splits=4)
    print(Xs.shape)
    for train_index,test_index  in kf.split(range(4)):
        train_X, test_X =  np.concatenate( Xs[train_index],0),  np.concatenate(Xs[test_index],0)
        train_y, test_y = np.concatenate(ys[train_index],0), np.concatenate(ys[test_index],0)
        min_max_scaler = preprocessing.MinMaxScaler()
        # train_X,test_X = min_max_scaler.fit_transform(train_X),min_max_scaler.fit_transform(test_X)
        print(train_X.shape)
        print(datasets[2:][test_index[0]].name)

        feature_index =   [55,59] + [ i for i in range(55)]  #,57,58,59,60
        model = MLPRegressor(max_iter = 50,hidden_layer_sizes=100,activation = "logistic")
        model.n_layers_ = 2
        print(train_X[:2,feature_index])
        reg = model.fit(train_X[:,feature_index], train_y)

        predicted = reg.predict(test_X[:,feature_index])
        print(test_X[:,feature_index].shape)
        # print(reg.coef_)
        from scipy.stats import pearsonr
        print(pearsonr(predicted,test_y))


if __name__ == "__main__":
    test_model()
    # submit_semeval()
    # train_and_predict()
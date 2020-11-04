import torch
import random
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import os
import numpy as np
from tqdm import tqdm
import math
class Judger(object):
    def __init__(self, threshold = None, corpus1=None, corpus2=None):
        self.threshold = threshold

    def set_corpora(self,corpus1,corpus2):
        self.corpus1 = corpus1
        self.corpus2 = corpus2

    def _get_score(self,word):
        return random.random()

    def clean(self,s):
        if "_" in s:
            return  s.split("_")[0]
        else:
            return s

    def get_binary_score(self,score ):
        if self.threshold is None:
            self.threshold = 0.5
        return int(score>self.threshold)

    def get_score(self,word,is_bool=True):
        score = self._get_score(self.clean(word))
        if  is_bool:
            return self.get_binary_score(score)
        else:
            return score

    def get_scores(self,words, is_bool=True ):
        scores = [self.get_score(word,is_bool) for word in words]
        return scores

    def get_cached_sentences(self,word):
        lines_pair= []
        for file_path in [self.corpus1,self.corpus2]:
            lines = []
            path,filename = os.path.split(file_path)
            path = os.path.join("cache", path.replace("/","_"))
            if not os.path.exists(path):
                os.mkdir(path)
            cached_filename = os.path.join(path,word+".txt")
            if not os.path.exists(cached_filename):
                from sys import platform
                if platform == "linux" or platform == "linux2":
                    cmd = "grep -i ' {} '  {} > {}".format(word,file_path,cached_filename)
                    print(cmd)
                    os.system(cmd)
                else:# linux
                    with open(file_path, encoding="utf-8") as f, open(cached_filename, "w", encoding="utf-8") as newf:
                        for index, line in enumerate(f):
                            if " {} ".format(word) in line:
                                newf.write(line)
                                lines.append(line)

            with open(cached_filename, encoding="utf-8") as f:
                for index, line in enumerate(f):
                    lines.append(line)
            lines_pair.append(lines)
        return lines_pair

    def get_acc(self,targets, words = None):
        if words is None:
            words = []
        predicted =  self.get_scores(words)
        return accuracy_score(predicted,np.array(targets))

    def predict(self,words,output_file =None, path="./"):
        if  type(words) == list:
            scores = self.get_scores(words,is_bool=False)

            if output_file is None:
                with open("submission.txt","w",encoding="utf-8") as f:
                    for word,score in zip(words,scores):
                        f.write("{}\t{}\n".format(word,self.get_binary_score(score)))
            else:
                if not os.path.exists(path):
                    os.mkdir(path )
                    os.mkdir(path+"/task1")
                    os.mkdir(path + "/task2")
                file1 =  os.path.join(path+"/task1",output_file.lower()+".txt")
                file2 =  os.path.join(path+"/task2",output_file.lower()+".txt")
                print("writing submissions to {} and {}".format( file1,file2))
                with open(file1,"w",encoding="utf-8") as f:
                    for word,score in zip(words,scores):
                        f.write("{}\t{}\n".format(word,self.get_binary_score(score)))
                with open(file2,"w",encoding="utf-8") as f:
                    for word,score in zip(words,scores):
                        f.write("{}\t{}\n".format(word,score))
        else:
            words = [word.strip() for word in open(words,encoding="utf-8").readlines()]
            self.predict(words,output_file=output_file, path=path)
        return output_file

    def get_acc_by_files(self,target_file):
        words,scores = [] , []
        with open(target_file, "r",encoding="utf-8") as f:
            for line in f:
                line = line.split()
                words.append(line[0].strip())
                scores.append(int(line[1]))
        return self.get_acc(scores,words)

    def read_scores(self,filename,is_int =True):
        words, scores = [], []
        # print(filename)
        with open(filename, "r",encoding="utf-8") as f:
            for line in f:
                tokens = line.split()
                # print(line)
                word = tokens[0].strip()
                if "_" in word:
                    word = word.split("_")[0]
                words.append(word)
                if is_int:
                    scores.append(int(tokens[1]))
                else:
                    scores.append(float(tokens[1]))
        return words,scores

    def get_acc_by_files_with_testing(self,target_file,graded_file=None):
        words,scores = self.read_scores(target_file)

        predicted = self.get_scores(words, is_bool=False)
        print("-" * 50)
        print(target_file)
        print(predicted)
        print(scores)

        if graded_file is not None:
            graded_words,graded_scores = self.read_scores(graded_file, is_int = False)
            assert  len(words) == len(graded_words), "size does not match"
            print("pearsonr")
            print(pearsonr(predicted, graded_scores))


        # for i in range(10):
        # threshold = np.median(np.array([score  for score in predicted if score!=0 and score!=1 ]))
        threshold = np.percentile(np.array([score for score in predicted if score != 0 and score != 1]),60)
        print("threshold")
        normalized_predicted = [int(score> threshold) if not math.isnan(score) else 0 for score in predicted]
        print(threshold)
        print(normalized_predicted)
        return accuracy_score(normalized_predicted, scores) # benyou




    def filter(self,words,prefix="clean"):

        words = set(words)
        for path in [self.corpus1, self.corpus2]:
            if os.path.exists(path+prefix):
                continue
            with open(path, encoding="utf-8") as f, open(path+prefix, "r", encoding="utf-8") as out:
                for line in f:
                    line = line.lower().strip()
                    tokens = set(line.split())
                    if len(words.intersection(tokens)):
                        out.write(line+"\n")
        self.corpus1 += prefix
        self.corpus2 += prefix


if __name__ == "__main__":
    model = Judger("flat/corpus_0.txt","flat/corpus_1.txt")
    score = model.get_acc_by_files("trial/truth.txt")
    print(score)
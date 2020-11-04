from models.Judger import Judger
from tqdm import tqdm
import math
from transformers import AutoConfig,  AutoModel, AutoTokenizer
import os
class CountJudger(Judger):
    def __init__(self,threshold = None, corpus1=None, corpus2=None):
        super(CountJudger,self).__init__(threshold,corpus1,corpus2)

    def _get_score(self,word):
        features = []
        for sentences in self.get_cached_sentences(word):
            features.append(len(sentences))
        # print(features)
        return math.fabs(features[0]-features[1])/(max(features[0],features[1])+0.0001)







from models.Judger import Judger
from tqdm import tqdm
import math
from transformers import AutoConfig,  AutoModel, AutoTokenizer
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import unicodedata
max_seq = 128
batch_size = 64
word = "attack"

import re

def remove_accents(string):

    string = re.sub(u"[àáâãäå]", 'a', string)
    string = re.sub(u"[èéêë]", 'e', string)
    string = re.sub(u"[ìíîï]", 'i', string)
    string = re.sub(u"[òóôõö]", 'o', string)
    string = re.sub(u"[ùúûü]", 'u', string)
    string = re.sub(u"[ýÿ]", 'y', string)

    return string

bert_path=".cache/bert-multilingual"
config = AutoConfig.from_pretrained(bert_path)
tokenizer = AutoTokenizer.from_pretrained(bert_path)
model =  AutoModel.from_pretrained(bert_path,config=config)
# sentences = open("cache/data_English_corpus1/{}.txt".format(word)).readlines()

def tokenized_word(word):
    return "".join([token if not token.startswith("##") else token[2:] for token in tokenizer.tokenize(word)])

def get_word_positions(word, token_ids):
    positions = []
    for token in token_ids:
        # print(token)
        token_positions = dict()
        count = 0
        pre_subword = ""
        start = 0
        for i, subword in enumerate(token):
            if subword.startswith("##"):
                pre_subword += subword[2:]
            else:
                token_positions.setdefault(pre_subword, [])
                token_positions[pre_subword].append([pos for pos in range(start, i)])
                start = i
                pre_subword = subword
                count += 1
        token_positions.setdefault(pre_subword, [])
        token_positions[pre_subword].append([pos for pos in range(start, i)])
        # word = unicodedata.normalize('NFD', word).encode('ascii', 'ignore')
        word = tokenized_word(word)
        if word in token_positions:
            positions.append(token_positions[word] if word in token_positions else None)
    return positions

def get_mean_representation_from_sentence( word, sentences ,model,tokenizer, output_representation = False):

    token_ids = [tokenizer.tokenize("[CLS] " + sent + " [SEP]") for sent in sentences]
    token_ids = [(token_id + ["[PAD]"]*(max_seq-len(token_id)))[:max_seq] for token_id in token_ids]
    # print([len(i) for i in token_ids])
    segments_ids = [tokenizer.convert_tokens_to_ids(token_id)  for token_id in token_ids]

    full_batch_nums = len(segments_ids)//batch_size
    batched_segments_tokens = [(segments_ids[batch_size*i:batch_size*(i+1)],token_ids[batch_size*i:batch_size*(i+1)]) for i in range(full_batch_nums)] \
                              + [(segments_ids[full_batch_nums*batch_size:],token_ids[full_batch_nums*batch_size:])]
    # for i in range(full_batch_nums-1):
    #     print(batch_size*i,batch_size*(i+1))
    #     print(len(segments_ids[batch_size*i:batch_size*(i+1)]))
    # print(len(segments_ids[full_batch_nums*batch_size:]))
    # print("*"*10)

    representations = []
    for step, (segments_ids,token_ids) in enumerate(batched_segments_tokens):
        # print("len of segments {} and token id {} in {}-th step".format(len(segments_ids),len(token_ids),step))
        with torch.no_grad():
            if len(segments_ids)==0:
                # print("len of segments is zero: in {}-th step".format(step))
                continue
            embedding,cls_embedding = model(torch.tensor(segments_ids,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        embedding = embedding.cpu().detach().numpy()
        positions = get_word_positions(word,token_ids)
        # print(positions)
        for i,sentence_positions in enumerate(positions):
            for position in sentence_positions:
                new_representations =  [embedding[i][pos] for pos in position]
                if len(new_representations) > 0:
                    representations.extend([np.mean(new_representations, 0) ])
                else:
                    print("word  {} :something went wrong and we cannot get any represenation with a batch (length {}) in {}-th step".format(word,len(token_ids),step))
            # representations.extend([np.mean([embedding[i][pos] for pos in position],0) for position in sentence_positions])

    print("{} word has {} batch with {} sentences representation in total: {}".format(word, len(batched_segments_tokens), len(sentences), len(representations)))

    if output_representation:
        return representations
    return np.mean(representations,0)




import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import  random
num_capsules, num_route_nodes, in_channels, out_channels = 64,3 ,20,20

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class RoutingLayer(nn.Module):
    def __init__(self, routing_nums = 3):
        super(RoutingLayer,self).__init__()
        self.routing_nums = routing_nums

    def forward(self, x):
        taken = torch.zeros(x.size()[0])
        for i in range(self.routing_nums):
            mean_x = x[taken==0].mean(0,keepdim=False)
            probs = F.softmax(1/torch.matmul(x,mean_x)+taken)
            taken[probs.argmax(-1)] += -100000
        return x[taken!=0]

class RandomSampler(nn.Module):
    def __init__(self, k = 3):
        super(RandomSampler,self).__init__()
        self.k = k

    def forward(self, x):
        perm = torch.randperm(x.size(0))
        idx = perm[:self.k]
        samples = x[idx]
        return samples


class SuperSampler(nn.Module):
    def __init__(self, k = 3):
        super(SuperSampler,self).__init__()
        self.k = k

    def forward(self, x):
        x = torch.multinomial(x, self.k, replacement=True)
        return x

def balance_sampler(x,y,k=10):
    # x = [item for item in x]
    # y = [item for item in y]
    x, y = np.array(x), np.array(y)
    if k!=-1:
        selected_x = np.random.choice(len(x), size=k, replace=True)
        selected_y = np.random.choice(len(y), size=k, replace=True)

        x = x[selected_x]
        y = y[selected_y]
    return  x,y



class Matcher(nn.Module):
    def __init__(self, k = 3):
        super(Matcher,self).__init__()
        self.sampler = SuperSampler(k)

    def forward(self,x,y, balance_needed = True):
        print(x,y)
        if balance_needed:
            x, y = self.sampler(x),self.sampler(y)
        print(x,y)

        A = torch.matmul(x,y.transpose(0,1))
        return A


def histogram(a,step=10):
    bins = 100//step   #fully divided
    features = []
    for bin in range(bins+1):
        features.append(np.percentile(a, bin*10))
    return features

def calculate_features(x,y,k=10):
    x,y = balance_sampler(x,y,k)
    A = np.matmul(x,y.T)   # k * K
    row_mean = A.mean(0)
    row_max = A.max(0)

    col_mean = A.mean(1)
    col_max = A.max(1)

    features = []
    for item in [row_mean,row_max,col_mean,col_max]:
        features.append(histogram(item))
    if random.random()>0.5:
        features[0],features[1],features[2],features[3] = features[2],features[3],features[0],features[1]

    A.resize(k*k)
    features.append(histogram(A))
    final = np.concatenate(features,-1)
    return final

if __name__ == "__main__":

    # x = torch.rand([5,10])
    # y = torch.rand([5, 10])
    # r = RoutingLayer()
    # print(r(x))
    # RS = RandomSampler()
    # print(RS(x))
    # M= Matcher()
    # print(M(x,y))
    x = np.random.randn(15, 20)
    y = np.random.randn(13, 20)
    print(calculate_features(x,y))



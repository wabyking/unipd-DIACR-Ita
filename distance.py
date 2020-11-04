from scipy.spatial.distance import canberra
from scipy.spatial.distance import directed_hausdorff,euclidean,jensenshannon
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

def cosine_distance(a,b):
    a,b = np.array(a),np.array(b)
    return np.sum(a*b) / np.linalg.norm(a)/np.linalg.norm(b)


def convert_to_distribution(labels, K=4):
    p = np.zeros(K)
    for label in labels:
        p[label] =  p[label]+1
    # print(labels)
    # print(p)
    return p

def jsd_kmean(set1,set2,model = "kmeans",max_num=1000):
    if len(set1)>max_num:
        index = np.random.choice(len(set1), size=max_num, replace=False)
        set1 = np.array(set1)[index]
    if len(set2)>max_num:
        index = np.random.choice(len(set2), size=max_num, replace=False)
        set2 = np.array(set2)[index]
    size1,size2 = len(set1),len(set2)
    model =  KMeans(n_clusters=4)#, random_state=0
    clusters = model.fit(np.concatenate([set1,set2],0)).labels_
    # print(clusters)
    clusters1 = clusters[:size1]
    clusters2 = clusters[size1:]
    jsd = jensenshannon(convert_to_distribution(clusters1),convert_to_distribution(clusters2))
    return jsd **2

def jsd_mixture(set1,set2,max_num = 1000):
    if len(set1) > max_num:
        index = np.random.choice(len(set1), size=max_num, replace=False)
        set1 = np.array(set1)[index]
    if len(set2) > max_num:
        index = np.random.choice(len(set2), size=max_num, replace=False)
        set2 = np.array(set2)[index]

    size1,size2 = len(set1),len(set2)
    model = GaussianMixture(n_components=4)
    clusters = model.fit_predict(np.concatenate([set1,set2],0))
    # print(clusters)
    clusters1 = clusters[:size1]
    clusters2 = clusters[size1:]
    jsd = jensenshannon(convert_to_distribution(clusters1),convert_to_distribution(clusters2))
    return jsd **2

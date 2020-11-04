import collections
import os
fields = ["name","corpus1", "corpus2","target","truth","graded"]
dataset = collections.namedtuple("dataset",fields)
it_new = dataset("Italian_new", "data/Italian_new/corpus1/T0_plain.txt", "data/Italian_new/corpus2/T1_plain.txt", "data/Italian_new/targets.txt", "data/Italian_new/truth.txt",None)
it = dataset("Italian", "data/Italian/corpus1/corpus_0.txt", "data/Italian/corpus2/corpus_1.txt", "data/Italian/targets.txt", "data/Italian/truth.txt", None)
en = dataset("English", "data/English/corpus1/ccoha1.txt", "data/English/corpus2/ccoha2.txt", "data/English/targets.txt", "data/English/truth/binary.txt","data/English/truth/graded.txt")
sw = dataset("Swedish", "data/Swedish/corpus1/kubhist2a.txt", "data/Swedish/corpus2/kubhist2b.txt", "data/Swedish/targets.txt", "data/Swedish/truth/binary.txt", "data/Swedish/truth/graded.txt")
ge = dataset("German", "data/German/corpus1/dta.txt", "data/German/corpus2/bznd.txt", "data/German/targets.txt", "data/German/truth/binary.txt","data/German/truth/graded.txt")
la = dataset("Latin", "data/Latin/corpus1/LatinISE1.txt", "data/Latin/corpus2/LatinISE2.txt", "data/Latin/targets.txt", "data/Latin/truth/binary.txt", "data/Latin/truth/graded.txt")

datasets = [it_new,it,en,sw,ge,la]
if __name__ == "__main__":
    for data in datasets:
        for field in range(1,len(fields)):
            assert os.path.isfile(data[field]), "{} is not a file path".format(data[field])


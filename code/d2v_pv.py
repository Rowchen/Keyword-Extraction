import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
import re
import jieba.analyse
import gensim
from gensim.models import word2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument
from gensim.models import doc2vec


alldoc = pd.read_csv('../data/all_docs.txt',sep='\001',header=None)
alldoc.columns = ['id','title','doc']

train2 = pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
train2.columns = ['id','label']
alldoc = pd.merge(alldoc,train2,on=['id'],how='left')

train_idx=list(alldoc[~alldoc['label'].isnull()].index)
test_idx=list(alldoc[alldoc['label'].isnull()].index)

train=alldoc.iloc[train_idx].copy().reset_index(drop=True)

tf=np.load('../lsdata2/tf.npy')
tf2=tf[train_idx]


w2v_train_doc=list(pd.read_csv('../data/w2v_train_doc.txt',header=None)[0].values)

print (len(w2v_train_doc))
w2v_train_doc2=[]
for i in range(len(w2v_train_doc)//2):
    w2v_train_doc2.append(w2v_train_doc[i]+w2v_train_doc[i+len(w2v_train_doc)//2])
print (len(w2v_train_doc2))

print ('now train word2vec')


import time
a=time.time()
class sentences_generator():
    def __init__(self, doc):
        self.doc = doc
    def __iter__(self):
        for i,line in enumerate(self.doc):
            sentence = TaggededDocument(line.split(), tags=[i])
            yield sentence
sents=sentences_generator(w2v_train_doc2)
model = doc2vec.Doc2Vec(sents,dm=1,vector_size=300,window=13,min_count=1,hs=1,workers=12,epochs=30)
b=time.time()
model.save("../data/pv.txt")
print (b-a)

import operator
def get_doc2vec_score(data,tf):
    DS=[]
    tag_num=0
    hit_num=0
    for row in tqdm(range(data.shape[0])):
        d2v_score={}
        doc_vec=model[train_idx[row]]
        doc_vec/=np.sqrt(sum(doc_vec**2))
        for w in tf[row]:
            try:
                ls=model.wv.word_vec(w)
            except:
                continue
            d2v_score[w]=np.sum(ls*doc_vec)/np.sqrt(sum(ls**2))
        DS.append(d2v_score)
        
        tag=data.loc[row,'label'].split(',')
        w_tf=dict(sorted(d2v_score.items(),key=operator.itemgetter(1),reverse=True)[:2])
        for w in w_tf:
            if w in tag:
                hit_num+=1
        tag_num+=len(tag)
    print(hit_num/tag_num)
    print(hit_num)
    
    return DS

xxx=get_doc2vec_score(train,tf2)

import operator
def get_doc2vec_score2(data,tf):
    DS=[]
    for row in tqdm(range(data.shape[0])):
        d2v_score={}
        doc_vec=model[row]
        doc_vec/=np.sqrt(sum(doc_vec**2))
        for w in tf[row]:
            try:
                ls=model.wv.word_vec(w)
            except:
                continue
            d2v_score[w]=np.sum(ls*doc_vec)/np.sqrt(sum(ls**2))
        DS.append(d2v_score)
    return DS
D2V=get_doc2vec_score2(alldoc,tf)
np.save('../lsdata2/D2V.npy',D2V)



















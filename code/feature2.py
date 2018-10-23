import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import re
import operator

alldoc = pd.read_csv('../data/all_docs.txt',sep='\001',header=None)
alldoc.columns = ['id','title','doc']
train = pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
train.columns = ['id','label']
alldoc = pd.merge(alldoc,train,on=['id'],how='left')
train_idx=list(alldoc[~alldoc['label'].isnull()].index)
test_idx=list(alldoc[alldoc['label'].isnull()].index)
train=alldoc.iloc[train_idx].copy().reset_index(drop=True)
import gc
gc.collect()


deldata=alldoc

import jieba
import jieba.posseg as pseg
import jieba.analyse
jieba.load_userdict('../data/newword.txt')
jieba.enable_parallel(12)

'''hanlp'''
'''
注意,运行此代码之前,需要做两件事
1,先把第一步生成的用户词典copy到pyhanlp的目录下,具体路径为:
python3.6/site-packages/pyhanlp/static/data/dictionary/custom,请根据自己的目录进行修改
2,修改python3.6/site-packages/pyhanlp/static/目录下的hanlp.properties,修改
CustomDictionaryPath=data/dictionary/custom/newword.txt;data/dictionary/custom/CustomDictionary.txt;.....
'''
from pyhanlp import *
NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
NLPTokenizer.ANALYZER.enableCustomDictionaryForcing(False)
NLPTokenizer.ANALYZER.enableAllNamedEntityRecognize(True)
NLPTokenizer.ANALYZER.enableTranslatedNameRecognize(True)
NLPTokenizer.ANALYZER.enableNameRecognize(True)
NLPTokenizer.ANALYZER.enableOrganizationRecognize(True)
NLPTokenizer.ANALYZER.enablePlaceRecognize(True)
NLPTokenizer.ANALYZER.enableJapaneseNameRecognize(True)
NLPTokenizer.ANALYZER.enablePartOfSpeechTagging(True)


import gensim
from gensim.models import word2vec
TaggededDocument = gensim.models.doc2vec.TaggedDocument
from gensim.models import doc2vec
model=gensim.models.Doc2Vec.load("../data/d2v_dbow.txt")

i1=pd.read_csv('../data/idf.txt',header=None,sep=' ')
idf1={}
for i in range(i1.shape[0]):
    idf1[i1.loc[i,0]]=i1.loc[i,1]
stopword=list(pd.read_csv('../data/stopwords.txt',header=None,sep='\t')[0].values)

keyword2=pd.read_csv('../data/top3.csv')

keyword1=pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
keyword1.columns=['id','top3']
keyword=pd.concat([keyword1,keyword2],axis=0).reset_index(drop=True)

key_times={}
keywordlist=keyword['top3'].values
for row in tqdm(range(keywordlist.shape[0])):
    ls=keywordlist[row].split(',')
    for w in ls:
        key_times[w]=key_times.get(w,0)+1
        
tfidf1=np.load('../lsdata2/tfidf1.npy')


import numpy as np
import operator
def get_tfidf_TR_withKT(data,key_times,topk=30):
    TFIDF=[]
    TR=[]
    pos_ls=list(pd.read_csv('../data/jiebapos.txt',header=None)[0].values)
    doc_data=data['doc'].values
    title_data=data['title'].values
    
    for row in tqdm(range(data.shape[0])):   
        ls=str(doc_data[row])
        ls2=str(title_data[row])
        '''首先计算textrank,这里不带标题算,取前100个'''
        tr={}
        pred=jieba.analyse.textrank(ls, topK=100, withWeight=True, withFlag=True,allowPOS=pos_ls)
        for pr in pred:
            w=pr[0].word
            tr[w]=pr[1]  
        '''再计算tfidf,取前100个'''
        w_cnt={}
        for words in pseg.cut(ls,HMM=True):
            w=words.word
            if len(w)<2 or w in stopword:
                continue
            w_cnt[w]=w_cnt.get(w,0)+1
        for k,v in w_cnt.items():
            w_cnt[k]*=(idf1.get(k,12))
        tfidf=dict(sorted(w_cnt.items(),key=operator.itemgetter(1),reverse=True)[:500])

        
        '''用加权法求出前topk关键词,以此基础构建其他特征'''
        a=np.array(list(tfidf.values()))
        b=np.array(list(tr.values()))
        amean=a.mean()
        astd=a.std()+1e-7
        bsum=b.sum()
        
        doc_vec=model[row]
        doc_vec/=np.sqrt(sum(doc_vec**2))
        
        criti={}
        for w in tr:
            if w in tfidf:
                try:
                    lsw=model.wv.word_vec(w)
                except:
                    lsw=0.1
                    print (w)
                d2v_score=np.sum(lsw*doc_vec)/np.sqrt(sum(lsw**2))
                criti[w]=(tfidf[w]-amean)*0.28/astd+(tr[w])/bsum*3.17+np.log(key_times.get(w,0)+1)*0.43+d2v_score*5.66
                
        choose_top30=dict(sorted(criti.items(),key=operator.itemgetter(1),reverse=True)[:topk])
        tr2={}
        tfidf2={}
        for w in choose_top30:
            tfidf2[w]=tfidf[w]
            tr2[w]=tr[w]
        '''最后加上标题,标题如果没出现过就置0,因为后面会单独算标题的,至于textrank,就赋最小值'''
        for words in pseg.cut(ls2,HMM=True):
            w=words.word
            if len(w)<2 or w in stopword:
                continue
            if w not in choose_top30:
                tfidf2[w]=0
                try:
                    tr2[w]=tr2.get(w,pred[-1][1]*1.1)
                except:
                    tr2[w]=0

        TFIDF.append(tfidf2)
        TR.append(tr2)
    return TFIDF,TR

tfidf1,TR1=get_tfidf_TR_withKT(deldata,key_times)
np.save('../lsdata2/tfidf1.npy',tfidf1)
np.save('../lsdata2/TR1.npy',TR1)

tag_num=0
hit_num=0
for row in range(1000):
    tag=train.loc[row,'label'].split(',')
    for w in tag:
        tag_num+=1
        if w in tfidf1[row]:
            hit_num+=1
print (hit_num)

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
#D2V=get_doc2vec_score2(alldoc,tfidf1)
#np.save('../lsdata2/D2V2.npy',D2V)

def predict_proba(oword, iword):
    if oword==iword:
        return 0
    iword_vec = model[iword]#获取输入词的词向量
    oword = model.wv.vocab[oword]
    oword_l = model.trainables.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code*dot) 
    return lprob

import operator
def get_word2vec_score2(data,tf):
    WS=[]
    for row in tqdm(range(data.shape[0])):
        w2v_score={}
        for w in tf[row]:
            w2v_score[w]=sum([predict_proba(u, w)*tf[row][u] for u in tf[row]])/(sum([tf[row][u] for u in tf[row]])+1e-10)
        WS.append(w2v_score)
    return WS
W2V=get_word2vec_score2(alldoc,tfidf1)
np.save('../lsdata2/W2V2.npy',W2V)

print ('ifxingshi')
def ifxingshi(data,tfidf):
    f = open("../data/xingshi.txt")
    line = f.readline()   
    xingshi=[]
    while line:
        print(line[0], end = '')
        xingshi.append(line[0])
        line = f.readline()
    f.close()
    xingshi=set(xingshi)
    
    
    ifxing=[]
    for row in tqdm(range(data.shape[0])):   
        ls={}
        candi=tfidf[row]
        for w in candi:
            if w[0] in xingshi and len(w)<4:
                ls[w]=1
            else:
                ls[w]=0
        ifxing.append(ls)
    return ifxing
ifxing=ifxingshi(deldata,tfidf1)
np.save('../lsdata2/ifxing.npy',ifxing)
del ifxing
gc.collect()



print ('get_textrank')
def get_textrank(data,tfidf):
    TR=[]
    pos_ls=list(pd.read_csv('../data/jiebapos.txt',header=None)[0].values)
    doc_list=data['doc'].values
    title_list=data['title'].values
    for row in tqdm(range(data.shape[0])):
        tr={}
        candi=tfidf[row]
        ls=str(data.loc[row,'title'])+'&'+\
            str(data.loc[row,'title'])+'&'+str(data.loc[row,'title'])+'&'+\
            str(data.loc[row,'title'])+'&'+str(data.loc[row,'title'])+'&'+\
            str(data.loc[row,'title'])+'&'+str(data.loc[row,'title'])+'&'+\
            str(data.loc[row,'title'])+'&'+str(data.loc[row,'title'])+'&'+\
            str(data.loc[row,'title'])+'&'+str(data.loc[row,'doc'])
        
        pred=jieba.analyse.textrank(ls, topK=100, withWeight=True, withFlag=True,allowPOS=pos_ls)
        for pr in pred:
            w=pr[0].word
            if w in candi:
                tr[w]=pr[1]
        if len(tr)<len(candi):
            for w in candi:
                try:
                    tr[w]=tr.get(w,pred[-1][1]*1.1)
                except:
                    tr[w]=0
        TR.append(tr)
    return TR
TR2=get_textrank(deldata,tfidf1)
np.save('../lsdata2/TR2.npy',TR2)
import gc
del TR2
gc.collect()

print ('get_title_tfidf')
def get_title_tfidf(data,tfidf):
    TFIDF=[]
    for row in tqdm(range(data.shape[0])):  
        candi=tfidf[row]
        ls=str(data.loc[row,'title'])
        w_cnt={}
        for words in pseg.cut(ls,HMM=True):
            w=words.word
            if len(w)<2 or w in stopword:
                continue
            w_cnt[w]=w_cnt.get(w,0)+1

        for k,v in w_cnt.items():
            w_cnt[k]*=(idf1.get(k,12))
            
        ans={}
        for w in candi:
            ans[w]=w_cnt.get(w,0)
        TFIDF.append(ans)
    return TFIDF
tfidf2=get_title_tfidf(deldata,tfidf1)
np.save('../lsdata2/tfidf2.npy',tfidf2)

print ('get_tf_position')
def get_tf_position(data,tfidf):
    TF=[]
    FIRST=[]
    LAST=[]
    for row in tqdm(range(data.shape[0])):  
        candi=tfidf[row]
        ls=str(data.loc[row,'title'])+'&'+str(data.loc[row,'doc'])
        w_cnt={}
        word_seg=[]
        for words in pseg.cut(ls,HMM=True):
            w=words.word
            if len(w)<2 or w in stopword:
                continue
            word_seg.append(w)
            w_cnt[w]=w_cnt.get(w,0)+1
        
        cnt=0
        first={}
        last={}
        for w in word_seg:
            if w in candi:
                if w not in first:
                    first[w]=cnt/(len(word_seg)+1e-7)
                last[w]=cnt/(len(word_seg)+1e-7)-first[w]
            cnt+=1
          
        ans={}
        for w in candi:
            first[w]=first.get(w,0.5)
            last[w]=last.get(w,0.0)
            ans[w]=w_cnt.get(w,0)
            
        TF.append(ans)
        FIRST.append(first)
        LAST.append(last)
    return TF,FIRST,LAST

tf,FR,LR=get_tf_position(deldata,tfidf1)
np.save('../lsdata2/tf.npy',tf)
np.save('../lsdata2/FR.npy',FR)
np.save('../lsdata2/LR.npy',LR)
del tf
del FR
del LR
gc.collect()

print ('get_pos')
def get_pos(data,tfidf,use='jieba'):
    pos_ls=list(pd.read_csv('../data/jiebapos.txt',header=None)[0].values)
    jieba_pos_idx={}
    for i,pos in enumerate(pos_ls):
        jieba_pos_idx[pos]=i

    pos_ls=list(pd.read_csv('../data/hanlppos.txt',header=None)[0].values)
    hanlp_pos_idx={}
    for i,pos in enumerate(pos_ls):
        hanlp_pos_idx[pos]=i    
    
    POS=[]
    for row in tqdm(range(data.shape[0])):
        pos={}
        candi=tfidf[row]
        ls1=str(data.loc[row,'doc'])
        ls2=str(data.loc[row,'title'])
        if use=='jieba':
            for x in [ls1,ls2]:
                for w in pseg.cut(x):
                    if w.word in candi:
                        if str(w.flag) in jieba_pos_idx:
                            pos[w.word]=str(w.flag)
                        
            for w in candi:
                pos[w]=jieba_pos_idx[pos.get(w,'un')]
                
        if use=='hanlp':
            for x in [ls1,ls2]:
                for w in NLPTokenizer.segment(x):
                    if w.word in candi:
                        if str(w.nature) in hanlp_pos_idx:
                            pos[w.word]=str(w.nature)
            for w in candi:
                pos[w]=hanlp_pos_idx[pos.get(w,'un')]
                
        POS.append(pos)
    return POS

POS_jieba=get_pos(deldata,tfidf1,'jieba')
POS_hanlp=get_pos(deldata,tfidf1,'hanlp')
np.save('../lsdata2/POS_jieba.npy',POS_jieba)
np.save('../lsdata2/POS_hanlp.npy',POS_hanlp)
del POS_hanlp
del POS_jieba
gc.collect()


print ('teding')
def teding(data,tfidf,title=True):
    te=[]
    for row in tqdm(range(data.shape[0])):
        tr={}
        candi=tfidf[row]
        if title:
            ls=str(data.loc[row,'title'])
        else:
            ls=str(data.loc[row,'doc'])

        tmp=set(re.findall(r"《(.*?)》",ls))
        for w in candi:
            if w in tmp:
                tr[w]=1
            else:
                tr[w]=0
        te.append(tr)
    return te
TE1=teding(deldata,tfidf1)
TE2=teding(deldata,tfidf1,title=False)
np.save('../lsdata2/TE1.npy',TE1)
np.save('../lsdata2/TE2.npy',TE2)




































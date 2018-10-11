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

i1=pd.read_csv('../data/idf.txt',header=None,sep=' ')
idf1={}
for i in range(i1.shape[0]):
    idf1[i1.loc[i,0]]=i1.loc[i,1]
stopword=list(pd.read_csv('../data/stopwords.txt',header=None,sep='\t')[0].values)

print ('tfidf and tr')
def get_tfidf_TR(data,topk=30):
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
        tfidf=dict(sorted(w_cnt.items(),key=operator.itemgetter(1),reverse=True)[:100])
        '''用加权法求出前topk关键词,以此基础构建其他特征'''
        criti={}
        for w in tfidf:
            if w in tr:
                criti[w]=tfidf[w]*0.8+tr[w]*0.7
                
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
tfidf1,TR1=get_tfidf_TR(deldata)
np.save('../lsdata/tfidf1.npy',tfidf1)
np.save('../lsdata/TR1.npy',TR1)



print ('ifxingshi')
def ifxingshi(data,tfidf):
    '''姓氏是我总结出来的词库,一共一百多个姓氏'''
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
np.save('../lsdata/ifxing.npy',ifxing)


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
np.save('../lsdata/TR2.npy',TR2)


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
np.save('../lsdata/tfidf2.npy',tfidf2)

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
np.save('../lsdata/tf.npy',tf)
np.save('../lsdata/FR.npy',FR)
np.save('../lsdata/LR.npy',LR)

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
np.save('../lsdata/POS_jieba.npy',POS_jieba)
np.save('../lsdata/POS_hanlp.npy',POS_hanlp)


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
np.save('../lsdata/TE1.npy',TE1)
np.save('../lsdata/TE2.npy',TE2)
















































































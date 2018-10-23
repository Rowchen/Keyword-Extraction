import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import jieba
import jieba.analyse
alldoc = pd.read_csv('../data/all_docs.txt',sep='\001',header=None)
alldoc.columns = ['id','title','doc']
train = pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)
train.columns = ['id','label']
train_id_list = list(train['id'].unique())
train = pd.merge(alldoc[alldoc['id'].isin(train_id_list)],train,on=['id'],how='inner')
test = alldoc[~alldoc['id'].isin(train_id_list)]

def get_unigram(data,num):
    char_unigram={}
    for row in tqdm(range(num)):
        ls=str(data.loc[row,'doc'])+'&'+str(data.loc[row,'title'])
        for i in range(len(ls)):
            char_unigram[ls[i]]=char_unigram.get(ls[i],0)+1
    
    return char_unigram

char_unigram=get_unigram(alldoc,alldoc.shape[0])

'''stopword是从前文档中200频繁词提取出来的,这里就不重复这个过程了'''
'idf是从jieba里复制出来的,路径是jieba/analys/idf.txt,将它拷贝过来,存在data下'

stopword=list(pd.read_csv('../data/stopwords.txt',header=None,sep='\t')[0].values)
i1=pd.read_csv('../data/idf.txt',header=None,sep=' ')
idf1={}
for i in range(i1.shape[0]):
    idf1[i1.loc[i,0]]=i1.loc[i,1]
    
def get_new_word(data,char_unigram,num):
    char_sum=sum(char_unigram.values())
    new_gram={}
    for row in tqdm(range(num)):
        n_gram={}
        free_left={}
        free_right={}
        cnt=0
        ls='*'+str(data.loc[row,'doc'])+'&'+str(data.loc[row,'title'])+'%'
        for j in range(2,8):
            for i in range(1,len(ls)-j):
                tmp=ls[i:i+j]
                if re.search(u'，|。|！|：|？|《|》|、|；|“|”|;|\.|\"|）|（',tmp,flags=re.U) is not None \
                or tmp[0]=='的' or tmp[0]=='是' or tmp[-1]=='的' or tmp[-1]=='是':#or tmp in stopword:
                    continue

                n_gram[tmp]=n_gram.get(tmp,0)+1
                
                left=ls[i-1:i]
                if left=='，'or left=='。'or left=='？'or left=='《': 
                    free_left[tmp]=(free_left.get(tmp,set()))
                    free_left[tmp].add(cnt)
                    cnt+=1
                else:
                    free_left[tmp]=(free_left.get(tmp,set()))
                    free_left[tmp].add(left)
                    
                right=ls[i+j:i+j+1]
                if right=='，'or right=='。'or right=='？'or right=='》': 
                    free_right[tmp]=(free_right.get(tmp,set()))
                    free_right[tmp].add(cnt)
                    cnt+=1
                else:
                    free_right[tmp]=(free_right.get(tmp,set()))
                    free_right[tmp].add(right)
        
        for w,v in n_gram.items():
            thres1=5
            thres2=3
            if len(w)>=4:
                thres1-=1
                thres2-=1
            
            if v>=thres1:
                minfree=min(len(free_left[w]),len(free_right[w]))
                if minfree>=thres2:
                    if w not in stopword and w not in idf1:
                        new_gram[w]=new_gram.get(w,0)+n_gram[w]
                        
    print (len(new_gram))
    return (new_gram)

new_gram=get_new_word(alldoc,char_unigram,alldoc.shape[0])


np.save('../data/new_gram2.npy',new_gram)
np.save('../data/char_unigram2.npy',char_unigram)
import numpy as np
new_gram=np.load('../data/new_gram2.npy')[()]
char_unigram=np.load('../data/char_unigram2.npy')[()]
keywo = list(pd.read_csv('../data/train_docs_keywords.txt',sep='\t',header=None)[1].values)
keyword=[]
for i in keywo:
    keyword.extend(i.split(','))
keyword=set(keyword)


ls=[]
newword=[]
char_sum=sum(char_unigram.values())
for w,v in new_gram.items():
    unimuty=1
    for i in range(len(w)):
        unimuty*=(char_unigram[w[i]]/char_sum)
    unimuty=(new_gram[w]/char_sum/unimuty)
    if len(w)==2:
        if unimuty>0.75:
            newword.append(w)
    if len(w)==3:
        if unimuty>250:
            newword.append(w)
    if len(w)==4:
        if unimuty>12000:
            newword.append(w)
    if len(w)==5:
        if unimuty>7000000: 

            newword.append(w)
    if len(w)==6:    
        newword.append(w)
    if len(w)==7:    
        newword.append(w)
        
newword=list(set(newword))
print (len(newword))     


import re
new_word_list=set()
data=alldoc
for row in tqdm(range(data.shape[0])):
    try:
        new_word_list|=set(re.findall(r"《(.*?)》",str(data.loc[row,'doc'])+'&'+str(data.loc[row,'title'])))
        new_word_list|=set(re.findall(r'[A-z]{2,} [A-z]{2,}|[A-z]{2,}[0-9]+[A-z]*|[A-z]{2,}',str(data.loc[row,'doc'])+'&'+str(data.loc[row,'title'])))
    except:
        pass

new_word_list2=[]
for i in new_word_list:
    if len(i)<30 and len(i)>0:
        new_word_list2.append(i)
new_word=pd.DataFrame({'word':list(keyword)+list(new_word_list2)+list(newword)})
new_word.to_csv('../data/newword.txt',header=None,index=None)
print (new_word.shape)



import jieba
import jieba.posseg as pseg
jieba.load_userdict('../data/newword.txt')
jieba.enable_parallel(12)


tag_num=0
hit_num=0
right_num=0
error_list=set()
for row in tqdm(range(train.shape[0])):
    ls=str(train.loc[row,'doc'])+'&'+str(train.loc[row,'title'])
    tag=train.loc[row,'label'].split(',')
    seg_list=set()
    
    for w,pos in pseg.cut(ls,HMM=True):
        if w in tag:
            right_num+=1
            seg_list.add(w)
    tag_num+=len(tag)
    hit_num+=len(seg_list)
    error_list|=set(tag)-seg_list
print (hit_num/tag_num)
print(right_num)


def fenci(data): 
    embed={}
    embed['doc_embed']=[]
    embed['title_embed']=[]
    cnt=0
    for row in tqdm(range(0,data.shape[0]//120+1)): 
        for col in ['title','doc']:
            ls=""
            sp_w=[]
            for sent in data.loc[row*120:row*120+119,col].values:
                ls+=(str(sent)+'\n')
            ls=ls[:-1]
            for words in pseg.cut(ls,HMM=True):
                sp_w.append(words.word)
            sp=" ".join(sp_w)
            c=sp.split('\n')
            for em in c:
                embed[col+'_embed'].append(em)
    return embed

doc_title_embed=fenci(alldoc)
w2v_train_doc=doc_title_embed['doc_embed']+doc_title_embed['title_embed']

w2v_train_doc2=[]
for i in range(len(w2v_train_doc)//2):
    w2v_train_doc2.append(w2v_train_doc[i]+w2v_train_doc[i+len(w2v_train_doc)//2])
print (len(w2v_train_doc2))
w2v_train_doc2=pd.DataFrame(w2v_train_doc)
w2v_train_doc2.to_csv('../data/w2v_train_doc.txt',header=None,index=None)









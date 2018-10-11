## 运行
./runme.sh</br>


## 环境
ubuntu16.04</br>
python 3.6</br>
jieba</br>
pyhanlp</br>


## 各文件说明
fenci.py 主要用于提取特殊短语，生成用户词典，提高分词准确率</br>
feature1.py 提取了多种信息，包括tfidf,lda,textrank,word2vec,doc2vec，生成候选关键词及相关信息</br>
combine1.py 把这些信息融合起来，训练并预测</br>




# coding: utf-8

# In[2]:


import jieba
from gensim.models import word2vec
from gensim import models

emotionless={}
##語庫資料鍵入
with open('emotionless1.txt', 'r',encoding = 'utf8') as f:
    a=f.read().split("\t")
    for i in range(0,len(a),2):
        label=a[i]
        inword=a[i+1].split(" ")
        emotionless[label]=inword
        
        
itemize=["NA","NB","NC","ND","NE","NG","NH","NI","NJ","NK","NL","NN","PA","PB","PC","PD","PE","PF","PG","PH","PK"]
label=[['NA'],['NB'],['NC'],['ND'],['NE'],['NG'],['NH'],['NI'],['NJ'],['NK'],['NL'],['NN'],['PA'],['PB'],['PC'],['PD'],['PE'],['PF'],['PG'],['PH'],['PK']]

with open("stopwords.txt","r",encoding = 'utf8') as f:
    stopword_set=f.read().split("\n")

##將情緒標籤建立字典
##key:情緒標籤 value:之後鍵入的情緒相似度數值(目前為0)
dictwor={}
for i in range(len(itemize)):
    dictwor[itemize[i]]=0

    
##開啟需要預測的詞檔案
with open("input.txt",'r',encoding = 'utf8') as f:
    tokens = jieba.lcut(f.read(), HMM=True) #.lcut將切好的多個字變成字串

    q_list = []
    for token in tokens:
         if token not in stopword_set:
            q_list.append(token)

    ##建立欲預測詞的字典，用來容納算出來的100個詞
    ##KEY:預測詞  VALUE：在字庫中相似的前100個詞
    predword={} 
    for i in range(len(q_list)):
        predword[q_list[i]]="0"
      
##使用word2vec
##匯入已經使用維基百科條目訓練好的model
model = models.Word2Vec.load('word2vec.model')

##建立放置相似度分數的字典
##key:預測詞 value:預測詞運算分數字典
dictwor2={}

##第一步：先搜尋是否被切出來的詞與情緒語庫的文字有所相同，若有，直接貼上標籤。
for keyword in predword.keys():      
    for inword in predword[keyword]:    
        count={}
        for i in range(-1,20):
            a=str(itemize[i+1])
            if keyword in emotionless[a]: 
                dictwor2[keyword]=[a,1000]
                break
            else:continue
    
##第二步：若沒有相符的文字，則利用word2vec與情緒語庫進行相似度比對，出來的分數加總若最高則該預測詞標籤為該分數最高的標籤
for keyword in predword.keys():      
    if keyword not in dictwor2.keys():    
            count={}
            for i in range(-1,20):
                a=str(itemize[i+1])                        
                wordcount=0
                emsimsum=0
                for j in range(len(emotionless[itemize[i+1]])):
                    try:
                        emsimsum+=model.similarity(keyword,emotionless[itemize[i+1]][j])
                        inwordcount+=1
                    except Exception as e:
                        repr(e)
                        continue
                count[a]=emsimsum/(wordcount+1)
                dictwor2[keyword]=count
    print(keyword+"end")

##排序並且抽取出分數最高的情緒標籤
##key:預測詞 values:最高分數的標籤
worddict={}

for j in dictwor2:
    if type(dictwor2[j]) == list:
        worddict[j]=dictwor2[j][0]
    else:
        worddict[j]=sorted(dictwor2[j].items(),key=lambda item:item[1], reverse = 1 )[0][0]

##key:label value:計數
emolabelcount={}
elist=list(worddict.values())

for i in range(-1,20):
    a=str(itemize[i+1])  
    emolabelcount[a]=elist.count(a)

sentenceemotion=sorted(emolabelcount.items(),key=lambda item:item[1], reverse = 1 )[0][0]
print(sentenceemotion)
with open (".\sentencewmotionoutput\sentenceemot.txt","w") as f1:
    f1.write(sentenceemotion)


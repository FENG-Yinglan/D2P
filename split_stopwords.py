
# coding: utf-8


from nltk.corpus import stopwords

def removeStopWords(originSegs):
    stopWords = set(stopwords.words('english'))
    punctuationList = [',','\t',';','/','*','-','_','[',']','&','(',')','+','$','{', '%','}',':','#','"', '=','<','>']
    #'.','!','?'分句
    resultStr = ""
    for seg in originSegs:
        if seg not in stopWords:
            resultStr += seg+" "
    for punctuation in punctuationList:
        resultStr = resultStr.replace(punctuation,'')
    return resultStr 



import warnings 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora



# 词干化_LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer 

file = open('sentences_LS.txt', 'w')
for i in ds2:
    documents = i
    texts = [[word for word in document.split()]for document in documents]

    for item in texts:
        if len(item) > 0:
            for word in item:
                ls = LancasterStemmer()
                word = ls.stem(word)
                file.write(word)
                file.write(' ')
            file.write('\n')    
            



# 词干化_PorterStemmer
from nltk.stem import PorterStemmer

file = open('sentences_PS.txt', 'w')
for i in ds2:
    documents = i
    texts = [[word for word in document.split()]for document in documents]

    for item in texts:
        if len(item) > 0:
            for word in item:
                ps = PorterStemmer()
                word = ps.stem(word)
                file.write(word)
                file.write(' ')
            file.write('\n')    




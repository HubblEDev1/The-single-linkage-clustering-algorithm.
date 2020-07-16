import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
spanish_stopwords = set(stopwords.words('spanish'))


def tokens(text, regex=re.compile(r'[a-záéíóúñ]{3,}')) -> list:
    tokens = regex.findall(text) # Todos los tokens que hagan match con la regex
    return [word for word in tokens if word not in spanish_stopwords]


def normalize(mat:np.array,axis=1) -> np.array:
    return mat/np.sum(np.square(mat),axis=axis)[:,np.newaxis]


def sim(a:np.array,b:np.array) -> float:
    return np.dot(a,b)

file_1='manV2.txt'
file_2='womanV2.txt'

with open(file_1,'r',encoding='utf-8') as m, open(file_2,'r',encoding='utf-8') as w:
    texto_m=[line.strip() for line in m]
    texto_w=[line.strip() for line in w]

#Descomentar para saber el numero de documentos.
#print(len(texto_m))
#print(len(texto_w))

train=[(i,'man') for i in texto_m]+[(j,'woman') for j in texto_w]

df = pd.DataFrame(train)

vectorizer = TfidfVectorizer(tokenizer=tokens, stop_words=spanish_stopwords, lowercase=True)
mat = vectorizer.fit_transform(df[0]).toarray()

#elimina las filas de 0, numpy all regresa true o false si todas las filas o columnas de una 
mat = mat[~np.all(mat == 0, axis=1)]

mat_tfdif = normalize(mat)

def simple_hac(d:np.array):
    N = d.shape[0]
    print(N)
    C = np.zeros(N*N)
    I = np.ones(N)
    A = []
    indices = np.array([(i,j) for i in range(N) for j in range(N)])
    for n in range(N):
        for i in range(N):
            C[n*N + i] = sim(d[n],d[i])
    for k in range(N - 1):
        #argmax
        sorted_i = np.argsort(C)
        for i,m in indices[sorted_i][::-1]:
            if i==m or not I[i] or not I[m]:
                continue
            break
        A.append((i,m))
        for j in range(N):
            temp_1 = sim(d[i],d[j])
            temp_2 = sim(d[m],d[j])
            sim_m = max(temp_1,temp_2)
            C[i*N + j] = sim_m
            C[j*N + i] = sim_m
        I[m] = False
    
    return A
    

final = simple_hac(mat_tfdif)
final

    

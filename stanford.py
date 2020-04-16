import requests
import json
import os

url="http://localhost:9000"
properties={'annotators':'tokenize,ssplit,pos','outputFormat':'json'}
params={'properties':str(properties)}



def getpos(f1,f2):
    with open(f1, 'rt', encoding='utf-8') as data_in:
        with open(f2, 'wt', encoding='utf-8') as file2:
           for data in data_in:
                data = data.strip("\n")
                data = data.strip()
                result = requests.post(url, data, params=params).json()
                sentences=result.get('sentences')
                wordpos = []
                sentence=sentences[0]
                words=sentence.get('tokens')
                for i in range(len(words)):
                    pos=words[i].get('pos')
                    wordpos.append(pos)
                file2.write(str(wordpos))
                file2.write("\n")
        file2.close()
    data_in.close()











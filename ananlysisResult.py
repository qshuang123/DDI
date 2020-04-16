from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pathlib import Path
import string

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from lxml import etree
import warnings

from stanford import getpos
import re
def printwronginstance(f1,f2,f3,f4):
    predict = []
    instance = []
    result = []
    wrong = []
    countadvise=0
    countmechanism=0
    counteffect=0
    countint=0
    countnone=0
    with open(f1, 'rt', encoding='utf-8') as File1:
        for line1 in File1:
            predict.append(line1)
        print("predict length:", len(predict))
    File1.close()
    with open(f2, 'rt', encoding='utf-8') as File2:
        for line2 in File2:
            lines = line2.split("$")
            result.append(lines[1])
        print("result length:", len(result))
    File2.close()
    with open(f3, 'rt', encoding='utf-8') as File3:
        for line3 in File3:
            instance.append(line3)
        print("instence :", len(instance))
    File3.close()
    with open(f4, 'wt', encoding='utf-8') as File4:
        for i in range(len(predict)):
            if str(predict[i]).strip("\n") != str(result[i]).strip("\n"):
                wrong.append(result[i] + "/" + str(predict[i]).strip("\n") + "/" + instance[i])
                if str(result[i]).strip("\n")=="advise":
                    countadvise=countadvise+1
                if str(result[i]).strip("\n") == "mechanism":
                    countmechanism=countmechanism+1
                if str(result[i]).strip("\n") == "effect":
                    counteffect=counteffect+1
                if str(result[i]).strip("\n") == "int":
                    countint=countint+1
                if str(result[i]).strip("\n") == "none":
                    countnone=countnone+1
        print("countadvise:",countadvise)
        print("countmechanism:", countmechanism)
        print("counteffect:", counteffect)
        print("countint:", countint)
        print("countnone:", countnone)
        for i in wrong:
            File4.write(i)
    File4.close()
def deletedrug0(f1,f2):
    sentence = []
    sentences=[]
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            line=line.replace("drug0","")
            sentence.append(line)
    data_in.close()

    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for line in sentence:
        line = re.sub(r, '', line)
        for i in range(6):
            while (line.find("  ") != -1):
                index8 = line.find("  ")
                sline = line[index8:index8 + 2]
                line = line.replace(sline, " ")
        sentences.append(line)

    data_in.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in sentences:
            line=str(line).strip(" ")
            file2.write(line)

    file2.close()






def main():
    printwronginstance(".\\predicts\\ENDSEMATTENTION+POSITION+10+0.69\\predictions-taskEpoch64.txt", ".\\Test2013\\testIstance.txt", ".\\Test2013\\testCsentence_token.txt", ".\\evaluation\\inputAttentionModelepoch-64F-0.69.txt")
    # deletedrug0(".\\Train2013\\trainCsentence_token.txt",".\\Train2013\\trainCsentence_token(deletedrug0).txt")
    # deletedrug0(".\\Test2013\\testCsentence_token.txt",".\\Test2013\\testCsentence_token(deletedrug0).txt")

if __name__ == "__main__":
    main()
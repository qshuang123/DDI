from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings
import re
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np


maxlen=100
distanceListnum=199
vec_size=50

PositionListPath=".\\Train2013\\PositionList.txt"  #train+test
AllPositionListPath=".\\Train2013\\AllPositionList.txt"
AllPositionMatricxPath=".\\Train2013\\AllPositionMatricx.txt"

traindatapath="Train2013/"
trainIstanceResultpath=".\\Train2013\\trainIstanceResult.txt"
ProcessedtrainIstanceResultpath=".\\Train2013\\ProcessedtrainIstanceResult.txt"
trainIstancepath=".\\Train2013\\trainIstance.txt"
ProcessedtrainIstancepath=".\\Train2013\\ProcessedtestIstance.txt"
trainIstanceDrugpath=".\\Train2013\\trainCsentence_token.txt"
trainIstanceDrugPosition1path=".\\Train2013\\trainIstanceDrugPosition1.txt"
trainIstanceDrugPosition1Processed=".\\Train2013\\ProcessedtrainIstanceDrugPosition1.txt"
trainIstanceDrugPosition2path=".\\Train2013\\trainIstanceDrugPosition2.txt"
trainIstanceDrugPosition2Processed=".\\Train2013\\ProcessedtrainIstanceDrugPosition2.txt"

testIstancepath=".\\Test2013\\testIstance.txt"
testIstanceDrugPosition1path=".\\Test2013\\testIstanceDrugPosition1.txt"
testIstanceDrugPosition1Processed=".\\Test2013\\ProcessedtestIstanceDrugPosition1.txt"
testIstanceDrugPosition2path=".\\Test2013\\testIstanceDrugPosition2.txt"
testIstanceDrugPosition2Processed=".\\Test2013\\ProcessedtestIstanceDrugPosition2.txt"
ProcessedtestIstancepath=".\\Test2013\\ProcessedtestIstance.txt"



deepProcessedtestIstancepath=".\\Test2013\\ProcessedtestIstance.txt"
testIstanceResultpath=".\\Test2013\\testIstanceResult.txt"
ProcessedtestIstanceResultpath=".\\Test2013\\ProcessedtestIstanceResult.txt"
testIstanceDrugpath=".\\Test2013\\testCsentence_token.txt"

testIstanceDrugPositionweighted=".\\Test2013\\testPosition1weighted.txt"
testIstanceDrugPositionweightedProcessed=".\\Test2013\\ProcessedtestPositionweighted.txt"

trainIstanceDrugPositionweighted=".\\Train2013\\trainPosition1weighted.txt"
trainIstanceDrugPositionweightedProcessed=".\\Train2013\\ProcessedtrainPositionweighted.txt"

def preprocess(fp,fw):
    with open(fp, 'rt', encoding='utf-8') as txtFile:
        with open(fw, 'wt', encoding='utf-8') as writeFile:
            for line in txtFile:
                index1=line.find("[")
                if index1!=-1:
                    line=line.replace("[","")
                index2=line.find("'")
                if index2!=-1:
                    line=line.replace("'","")
                index3=line.find("]")
                if index3!=-1:
                    line=line.replace("]","")
                index4 = line.find(",")
                if index4 != -1:
                    line = line.replace(",", " ")
                index5 = line.find("$")
                if index5 != -1:
                    line = line.replace("$", " ")
                index6 = line.find(":")
                if index6 != -1:
                    line = line.replace(":", " ")
                index7 = line.find("\"\"")
                if index7 != -1:
                    line = line.replace("\"\"", " ")
                index9 = line.find("``")
                if index9 != -1:
                    line = line.replace("``", " ")
                for i in range(5):
                    while (line.find("  ") != -1):
                        index8 = line.find("  ")
                        sline = line[index8:index8 + 2]
                        line = line.replace(sline, " ")
                writeFile.write(line)
        writeFile.close()
    txtFile.close()

def find_max_sentence(fp):
    with open(fp, 'rt', encoding='utf-8') as txtFile:
        maxsentence = 0
        maxtxt = str(0)
        j=0
        i=0
        count=0
        for line in txtFile:

            index1=line.find("drug1")
            index2=line.find("drug2")
            if index1==-1 or index2==-1:
                print(i)
                print(line)
                count=count+1
            lines = line.split(" ")

            j=len(lines)
            if maxsentence < j:
                maxsentence = j
                maxtxt = line

            i=i+1
        print("最大句子长度为：", maxsentence)
        print("最大句子为：", maxtxt)
        print(count)
    txtFile.close()

def getsubposition(fp,f1,f2,f3,f4,f5,f6):
    with open(fp, 'rt', encoding='utf-8') as txtFile:
        with open(f1, 'wt', encoding='utf-8') as File1:
            with open(f2, 'wt', encoding='utf-8') as File2:
                with open(f3, 'wt', encoding='utf-8') as File3:
                    with open(f4, 'wt', encoding='utf-8') as File4:
                        with open(f5, 'wt', encoding='utf-8') as File5:
                            with open(f6, 'wt', encoding='utf-8') as File6:
                                for line in txtFile:
                                    PositionToDrug1Listsub1 = []
                                    PositionToDrug1Listsub2 = []
                                    PositionToDrug1Listsub3 = []
                                    PositionToDrug2Listsub1 = []
                                    PositionToDrug2Listsub2 = []
                                    PositionToDrug2Listsub3 = []
                                    line = line.replace("\n", "")
                                    line = line.strip()
                                    lines = line.split(" ")
                                    i = 0
                                    for e in lines:
                                        i = i + 1
                                        if e == "drug1":
                                            key1 = i
                                        if e == "drug2":
                                            key2 = i
                                    j = 0
                                    for e in lines:
                                        j = j + 1
                                        distance1 = j - key1
                                        if distance1 == 0:
                                            distance1 = 100
                                        distance2 = j - key2
                                        if distance2 == 0:
                                            distance2 = 100

                                        if j < key1:
                                            PositionToDrug1Listsub1.append(distance1)
                                        if j >= key1 and j <= key2 :
                                            PositionToDrug1Listsub2.append(distance1)
                                        if j > key2:
                                            PositionToDrug1Listsub3.append(distance1)

                                        if j < key1:
                                            PositionToDrug2Listsub1.append(distance2)
                                        if j >= key1 and j <= key2 :
                                            PositionToDrug2Listsub2.append(distance2)
                                        if j > key2:
                                            PositionToDrug2Listsub3.append(distance2)


                                    File1.write(str(PositionToDrug1Listsub1))
                                    File1.write("\n")
                                    File2.write(str(PositionToDrug1Listsub2))
                                    File2.write("\n")
                                    File3.write(str(PositionToDrug1Listsub3))
                                    File3.write("\n")
                                    File4.write(str(PositionToDrug2Listsub1))
                                    File4.write("\n")
                                    File5.write(str(PositionToDrug2Listsub2))
                                    File5.write("\n")
                                    File6.write(str(PositionToDrug2Listsub3))
                                    File6.write("\n")
                            File6.close()
                        File5.close()
                    File4.close()
                File3.close()
            File2.close()
        File1.close()
    txtFile.close()

def getPosition(fp,f1,f2):
    with open(fp, 'rt', encoding='utf-8') as txtFile:
        with open(f1, 'wt', encoding='utf-8') as writeFile1:
            with open(f2, 'wt', encoding='utf-8') as writeFile2:

                for line in txtFile:
                    PositionToDrug1List = []
                    PositionToDrug2List = []
                    line=line.replace("\n","")
                    line=line.strip()
                    lines = line.split(" ")
                    i=0
                    for e in lines:
                        i=i+1
                        if e == "drug1":
                            key1 = i
                        if e == "drug2" :
                            key2 = i
                    j=0
                    for e in lines:
                        j=j+1
                        distance1 = j - key1
                        if distance1==0:
                            distance1=100
                        distance2 = j - key2
                        if distance2==0:
                            distance2=100
                        PositionToDrug1List.append(distance1)
                        PositionToDrug2List.append(distance2)

                    writeFile1.write(str(PositionToDrug1List))
                    writeFile1.write("\n")
                    writeFile2.write(str(PositionToDrug2List))
                    writeFile2.write("\n")
            writeFile2.close()
        writeFile1.close()
    txtFile.close()

def getPositionWeighted(fp,f1):
    with open(fp, 'rt', encoding='utf-8') as txtFile:
        with open(f1, 'wt', encoding='utf-8') as writeFile1:

                for line in txtFile:
                    Weight = []
                    line=line.replace("\n","")
                    line=line.strip()
                    lines = line.split(" ")
                    i=0
                    for e in lines:
                        i=i+1
                        if e == "drug1":
                            key1 = i
                        if e == "drug2" :
                            key2 = i
                    j=0
                    for e in lines:
                        j=j+1
                        distance1 = abs(j - key1)
                        distance2 = abs(j - key2)
                        distance = min(distance1, distance2)
                        w = 1 - (distance / i)
                        w = round(w,5)
                        Weight.append(w)
                    writeFile1.write(str(Weight))
                    writeFile1.write("\n")

        writeFile1.close()
    txtFile.close()

def getPositionRange(f1,f2,f3,f4,fc):
    with open(f1, 'rt', encoding='utf-8') as txtFile1:
        with open(f2, 'rt', encoding='utf-8') as txtFile2:
            with open(f3, 'rt', encoding='utf-8') as txtFile3:
                with open(f4, 'rt', encoding='utf-8') as txtFile4:
                    with open(fc, 'wt', encoding='utf-8') as writeFile1:
                        distanceList = []
                        count = 0
                        for line in txtFile1:
                            lines = line.split(",")
                            for d in lines:
                                d=d.strip(" ")
                                d=d.strip("\n")
                                if d not in distanceList:
                                    count=count+1
                                    distanceList.append(d)
                        for line in txtFile2:
                            lines = line.split(",")
                            for d in lines:
                                d=d.strip(" ")
                                d = d.strip("\n")
                                if d not in distanceList:
                                    count=count+1
                                    distanceList.append(d)
                        for line in txtFile3:
                            lines = line.split(",")
                            for d in lines:
                                d=d.strip(" ")
                                d = d.strip("\n")
                                if d not in distanceList:
                                    count=count+1
                                    distanceList.append(d)
                        for line in txtFile4:
                            lines = line.split(",")
                            for d in lines:
                                d = d.strip(" ")
                                d = d.strip("\n")
                                if d not in distanceList:
                                    count=count+1
                                    distanceList.append(d)
                        for p in distanceList:
                            writeFile1.write("{}\n".format(p))
                        print("count:", count)
                    writeFile1.close()
                txtFile4.close()
            txtFile3.close()
        txtFile2.close()
    txtFile1.close()

def delete0(f1):
    with open(f1, 'rt', encoding='utf-8') as txtFile1:
        with open(f1, 'wt', encoding='utf-8') as writeFile:
            distance=[]
            for line in txtFile1:
                lines = line.split(",")
                for d in lines:
                    if d=="0":
                        d="100"
                    distance.append(d)
                writeFile(str(distance))
                writeFile.write("\n")
        writeFile.close()
    txtFile1.close()

def creativePositionList(fc):
    distanceList=[]
    with open(fc, 'wt', encoding='utf-8') as writeFile1:
        for i in range(-99,101):
            distanceList.append(i)
        for p in distanceList:
            writeFile1.write("{}\n".format(p))
    writeFile1.close()

def get_Position_matrix(fm,vec_size, distanceListnum):
    length = distanceListnum
    distanceListnum_matrix = np.random.rand(length, vec_size)
    wf = open(fm, 'w', encoding='utf-8')
    for i in range(len(distanceListnum_matrix)):
        for j in range(len(distanceListnum_matrix[i])):
            wf.write(str(distanceListnum_matrix[i][j]) + " ")
        wf.write("\n")
    wf.close()

def getposlist(f1,f2,f3):
    posList = []
    with open(f1, 'rt', encoding='utf-8') as File1:
        for line in File1:
            line=line.strip("\n")
            lines=line.split(" ")
            for word in lines:
                if word not in posList:
                    posList.append(word)
    File1.close()
    with open(f2, 'rt', encoding='utf-8') as File2:
        for line in File2:
            line = line.strip("\n")
            lines=line.split(" ")
            for word in lines:
                if word not in posList:
                    posList.append(word)
    print(len(posList))
    File2.close()
    with open(f3, 'wt', encoding='utf-8') as File3:
        for line in posList:
            File3.write(line)
            File3.write("\n")
    File3.close()

def makeinstance(f1,f2,f3):
    entity1=[]
    entity2=[]
    sent=[]
    with open(f1, 'rt', encoding='utf-8') as file1:
        for line in file1:
            lines=line.split(",")
            entity1.append(lines[0])
            entity2.append(lines[1])
    file1.close()
    with open(f2, 'rt', encoding='utf-8') as file2:
        for line in file2:
            sent.append(line)
    file2.close()
    instance=[]
    for i in range(len(sent)):
        inst=entity1[i]+"$"+entity2[i]+"$"+sent[i]
        instance.append(inst)
    with open(f3, 'wt', encoding='utf-8') as file3:
        for line in instance:
            file3.write(line)

    file3.close()

def makeresultandinstance(f1,f2,f3):
    instance=[]
    result=[]
    with open(f1, 'rt', encoding='utf-8') as txtFile1:
        for line in txtFile1:
            instance.append(line)
    txtFile1.close()
    with open(f2, 'rt', encoding='utf-8') as txtFile2:
        for line in txtFile2:
            line=line.strip("\n")
            result.append(line)
    txtFile2.close()
    example=[]
    for i in range(len(instance)):
        eg=result[i]+"$"+instance[i]
        example.append(eg)
    with open(f3, 'wt', encoding='utf-8') as file3:
        for line in example:
            file3.write(line)
    file3.close()

def changeSpelling(f1,f2):
    lines=[]
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            line=line.lower()
            lines.append(line)
    data_in.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in lines:
            file2.write(line)
    file2.close()

def replaceNumber(f1,f2):

    writeline=[]
    with open(f1, 'rt', encoding='utf-8') as data_in:
        with open(f2, 'wt', encoding='utf-8') as file2:
            for line in data_in:
                line=line.split("$")
                result=line[0].strip(" ")
                entity1=line[1].strip(" ")
                entity2=line[2].strip(" ")
                sent=line[3]
                sentence=sent.split(" ")
                for i in range(len(sentence)):
                    word=sentence[i]
                    if word!=entity1 and word!=entity2 :
                        word1=word.strip("%()*+-<=>@[\]^_`{|}~ ")
                        if word1.isdigit():
                            sent=sent.replace(word,"number")
                writel=result+"$"+entity1+"$"+entity2+"$"+sent
                file2.write(writel)
        file2.close()
    data_in.close()

def deletepair(f1,f2):
    instance = []
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            lines = line.split("$")
            ent1 = lines[2].strip((" "))
            ent2 = lines[3].strip((" "))
            if ent1 != ent2:
                instance.append(line)
    data_in.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in instance:
            file2.write(line)
    file2.close()

def deletepairfinal(f1,f2):
    instance = []
    count = 0
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            lines = line.split("$")
            ent1 = lines[2].strip((" "))
            ent2 = lines[3].strip((" "))
            sentence = lines[4]
            result = lines[1]
            # "drug1 , drug2"  ("drug2 , drug1"dont exist)
            if "drug1 , drug2" in sentence :
                if result!="none":
                    print("have wrong instance in drug1,drug2:",result)
                else:
                    count=count+1
            else :
                instance.append(line)

            # "drug1 ( drug2 )"
            # if "drug1 ( drug2 )" in sentence:
            #     if result!="none":
            #         print("have wrong instance in drug1,drug2:",result)
            #     else:
            #         count=count+endsematten+newwordvector
            # else :
            #     instance.append(line)

            # "drug1 , drug2 , and drug0" "drug1 , drug0 , and drug2"  "drug0 , drug1 , and drug2"
            # if "drug1 , drug2 , and drug0" in sentence or "drug1 , drug0 , and drug2" in sentence or "drug0 , drug1 , and drug2" in sentence:
            #     if result != "none":
            #         print("have wrong instance in drug1,drug2:", result)
            #     else:
            #         count = count + endsematten+newwordvector
            # else:
            #     instance.append(line)

            # "drug1 such as drug2"
            # if "drug1 such as drug2" in sentence :
            #     if result != "none":
            #         print("have wrong instance in drug1,drug2:", result)
            #     else:
            #         count = count + endsematten+newwordvector
            # else:
            #     instance.append(line)

            # if "drug1 such as drug0 or drug2" in sentence :
            #     if result != "none":
            #         print("have wrong instance in drug1 such as drug0 or drug2:", result)
            #     else:
            #         count = count + endsematten+newwordvector
            # else:
            #     instance.append(line)

            # sentence=sentence.strip(" ")
            # index1 = sentence.find("drug1 :")
            # index2 = sentence.find("drug2 :")
            # if index1 == 0 or index2 == 0:
            #     if result!="none":
            #         print("have wrong instance in drug1,drug2:",result)
            #     else:
            #         count=count+endsematten+newwordvector
            # else :
            #     instance.append(line)
    print("count:",count)
    data_in.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in instance:
            file2.write(line)
    file2.close()
def loadsentence(f1,f2):
    instance = []
    result=[]
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            lines = line.split("$")
            # result.append(lines[0])
            sentence=lines[3]
            instance.append(sentence)
    data_in.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in instance:
            file2.write(line)

    file2.close()

def cleanfuhao(f1,f2):
    sentence=[]
    r ="drug0"
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            line = line.replace(r," ")
            for i in range(4):
                while (line.find("  ") != -1):
                    index8 = line.find("  ")
                    sline = line[index8:index8 + 2]
                    line = line.replace(sline, " ")
            sentence.append(line)
    data_in.close()

    with open(f2, 'wt', encoding='utf-8') as file2:
        for line in sentence:
            file2.write(line)
    file2.close()
    # sentence = []
    # r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    # with open(f1, 'rt', encoding='utf-8') as data_in:
    #     for line in data_in:
    #         line = re.sub(r, '', line)
    #         for i in range(4):
    #             while (line.find("  ") != -endsematten+newwordvector):
    #                 index8 = line.find("  ")
    #                 sline = line[index8:index8 + 2]
    #                 line = line.replace(sline, " ")
    #         sentence.append(line)
    # data_in.close()
    #
    # with open(f2, 'wt', encoding='utf-8') as file2:
    #     for line in sentence:
    #         file2.write(line)
    # file2.close()

def makefile(f1,f2):
    sentence = []
    with open(f1, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            sentence.append(line)
    data_in.close()
    with open(f2, 'a', encoding='utf-8') as file2:
        for line in sentence:
            file2.write(line)
            file2.write("\n")
    file2.close()

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
            file2.write(line)
            file2.write("\n")
    file2.close()

def getsubsection(f1,f2,f3,f4):
    subsentence1=[]
    subsentence2=[]
    subsentence3=[]
    with open(f1, 'rt', encoding='utf-8') as file1:
        for line in file1:
            index1=line.find("drug1")
            index2=line.find("drug2")
            line1=line[0:index1]
            line2=line[index1:index2+6]
            line3=line[index2+6:]
            line1=line1.strip(" ")
            line1=line1.strip("\n")
            line2 = line2.strip(" ")
            line2 = line2.strip("\n")
            line3 = line3.strip(" ")
            line3 = line3.strip("\n")
            subsentence1.append(line1)
            subsentence2.append(line2)
            subsentence3.append(line3)
    file1.close()
    with open(f2, 'wt', encoding='utf-8') as file2:
        for i in subsentence1:
            file2.write(i)
            file2.write("\n")
    file2.close()
    with open(f3, 'wt', encoding='utf-8') as file3:
        for i in subsentence2:
            file3.write(i)
            file3.write("\n")
    file3.close()
    with open(f4, 'wt', encoding='utf-8') as file4:
        for i in subsentence3:
            file4.write(i)
            file4.write("\n")
    file4.close()


def model_weigted():
    i=0
    F=0.737
    model_weighted_attention_dot=[]
    model_weighted_attention=[]
    model_weighted_attention_dense=[]
    model_weighted_sen_sdp_matric=[]

    model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_attention_dot" + ".txt", "r")
    for line in model_weight_filepath:
        model_weighted_attention_dot.append(line.strip("\n"))
    model_weight_filepath.close()
    model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_attention" + ".txt", "r")
    for line in model_weight_filepath:
        model_weighted_attention.append(line.strip("\n"))
    model_weight_filepath.close()
    model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_attention_dense" + ".txt", "r")
    for line in model_weight_filepath:
        model_weighted_attention_dense.append(line.strip("\n"))
    model_weight_filepath.close()
    model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_sen_sdp_matric" + ".txt", "r")
    model_weighted_sen_sdp_matric.append(line.strip("\n"))
    model_weight_filepath.close()

    model_weighted_attention_dot=str(model_weighted_attention_dot)
    model_weighted_attention_dot=model_weighted_attention_dot.split("]")
    print(model_weighted_attention_dot[0])

    print(model_weighted_attention_dot[473])
    print(model_weighted_attention_dot[474])
    print(model_weighted_attention_dot[475])
    print(model_weighted_attention_dot[-1])



def main():
    model_weigted()
    # preprocess(".\\Train2013\\analysis\\trainSentence_text.txt",".\\Train2013\\analysis\\processedtrainSentence_text.txt")
    # preprocess(".\\Test2013\\analysis\\testSentence_text.txt", ".\\Test2013\\analysis\\processedtestSentence_text.txt")
    # makeinstance(".\Train2013\\oldprocessed\\ProcessedtrainIstance.txt",".\Train2013\\oldprocessed\\trainIstance.txt",".\Train2013\\traindrugIstance.txt")
    # makeinstance(".\Test2013\\oldprocesseddata\\ProcessedtestIstance.txt",".\Test2013\\oldprocesseddata\\testIstance.txt", ".\Test2013\\testdrugIstance.txt")
    # deletepair(".\Train2013\\alltrainIstance.txt",".\Train2013\\processedalltrainIstance.txt")
    # preprocess(trainIstancepath, ProcessedtrainIstancepath)
    # find_max_sentence(trainIstanceDrugpath)#80
    # changeSpelling(".\Test2013\\testIstance.txt",".\Test2013\\Istance.txt")
    # changeSpelling(".\Train2013\\trainIstance.txt", ".\Train2013\\Istance.txt")

    # replaceNumber(".\Test2013\\Istance.txt",".\Test2013\\NIstance.txt")
    # replaceNumber(".\Train2013\\Istance.txt",".\Train2013\\NIstance.txt")

    # deletepairfinal(".\Train2013\\dIstance.txt", ".\Train2013\\dIstance1.txt")
    # deletepair(".\Train2013\\Istance.txt", ".\Train2013\\dIstance.txt")
    # getPosition(trainIstanceDrugpath, trainIstanceDrugPosition1path, trainIstanceDrugPosition2path)
    # getsubposition(trainIstanceDrugpath, ".\\Train2013\\trainTodrug1PositionSub1List.txt", ".\\Train2013\\trainTodrug1PositionSub2List.txt", ".\\Train2013\\trainTodrug1PositionSub3List.txt", ".\\Train2013\\trainTodrug2PositionSub1List.txt", ".\\Train2013\\trainTodrug2PositionSub2List.txt", ".\\Train2013\\trainTodrug2PositionSub3List.txt")
    # getsubposition(testIstanceDrugpath, ".\\Test2013\\testTodrug1PositionSub1List.txt", ".\\Test2013\\testTodrug1PositionSub2List.txt", ".\\Test2013\\testTodrug1PositionSub3List.txt", ".\\Test2013\\testTodrug2PositionSub1List.txt", ".\\Test2013\\testTodrug2PositionSub2List.txt", ".\\Test2013\\testTodrug2PositionSub3List.txt")
    # preprocess(".\\Train2013\\trainTodrug1PositionSub1List.txt", ".\\Train2013\\protrainTodrug1PositionSub1List.txt")
    # preprocess(".\\Train2013\\trainTodrug1PositionSub2List.txt", ".\\Train2013\\protrainTodrug1PositionSub2List.txt")
    # preprocess(".\\Train2013\\trainTodrug1PositionSub3List.txt", ".\\Train2013\\protrainTodrug1PositionSub3List.txt")
    # preprocess(".\\Train2013\\trainTodrug2PositionSub1List.txt", ".\\Train2013\\protrainTodrug2PositionSub1List.txt")
    # preprocess(".\\Train2013\\trainTodrug2PositionSub2List.txt", ".\\Train2013\\protrainTodrug2PositionSub2List.txt")
    # preprocess(".\\Train2013\\trainTodrug2PositionSub3List.txt", ".\\Train2013\\protrainTodrug2PositionSub3List.txt")
    # preprocess(".\\Test2013\\testTodrug1PositionSub1List.txt",".\\Test2013\\protestTodrug1PositionSub1List.txt")
    # preprocess(".\\Test2013\\testTodrug1PositionSub2List.txt", ".\\Test2013\\protestTodrug1PositionSub2List.txt")
    # preprocess(".\\Test2013\\testTodrug1PositionSub3List.txt", ".\\Test2013\\protestTodrug1PositionSub3List.txt")
    # preprocess(".\\Test2013\\testTodrug2PositionSub1List.txt", ".\\Test2013\\protestTodrug2PositionSub1List.txt")
    # preprocess(".\\Test2013\\testTodrug2PositionSub2List.txt", ".\\Test2013\\protestTodrug2PositionSub2List.txt")
    # preprocess(".\\Test2013\\testTodrug2PositionSub3List.txt", ".\\Test2013\\protestTodrug2PositionSub3List.txt")
    # preprocess(".\\Test2013\\testIstanceResult.txt", ".\\Test2013\\protestIstanceResult.txt")



    # getPosition(testIstanceDrugpath, testIstanceDrugPosition1path, testIstanceDrugPosition2path)
    # preprocess(trainIstanceDrugPosition1path,trainIstanceDrugPosition1Processed)
    # preprocess(trainIstanceDrugPosition2path, trainIstanceDrugPosition2Processed)
    # preprocess(testIstanceDrugPosition1path, testIstanceDrugPosition1Processed)
    # preprocess(testIstanceDrugPosition2path, testIstanceDrugPosition2Processed)
    # getPositionRange(trainIstanceDrugPosition1Processed,trainIstanceDrugPosition2Processed,testIstanceDrugPosition1Processed,testIstanceDrugPosition2Processed,PositionListPath)
    # getPositionWeighted(testIstanceDrugpath, testIstanceDrugPositionweighted)
    # getPositionWeighted(trainIstanceDrugpath, trainIstanceDrugPositionweighted)
    # preprocess(testIstanceDrugPositionweighted,testIstanceDrugPositionweightedProcessed)
    # preprocess(trainIstanceDrugPositionweighted, trainIstanceDrugPositionweightedProcessed)


    # creativePositionList(AllPositionListPath)
    # get_Position_matrix(AllPositionMatricxPath, vec_size, distanceListnum=200)
    # makeresultandinstance(".\Test2013\\testdrugIstance.txt",ProcessedtestIstanceResultpath,".\Test2013\\testresultIstance.txt" )
    # makeresultandinstance(".\Train2013\\traindrugIstance.txt",ProcessedtrainIstanceResultpath,".\Train2013\\trainresultIstance.txt" )

    # preprocess(trainIstanceResultpath, ProcessedtrainIstanceResultpath)
    # preprocess(testIstanceResultpath, ProcessedtestIstanceResultpath)
    #最大句子长度为94
    # preprocess(testIstancepath, ProcessedtestIstancepath)
    # find_max_sentence(testIstanceDrugpath)#87

    # loadsentence(".\Train2013\\trainIstanceD.txt", ".\Train2013\\trainCsentence_token.txt")
    # loadsentence(".\Train2013\\Csentence.txt", ".\Train2013\\Ksentence.txt")

    # getpos(".\Test2013\\sentence.txt",".\Test2013\\testpos.txt")
    # getpos(".\Train2013\\sentence.txt", ".\Train2013\\trainpos.txt")
    # preprocess(".\Test2013\\testpos.txt",".\Test2013\\pos.txt")
    # preprocess(".\Train2013\\trainpos.txt", ".\Train2013\\pos.txt")
    # getposlist(".\Test2013\\pos.txt",".\Train2013\\pos.txt",".\Train2013\\AllposList.txt")
    # get_Position_matrix(".\Train2013\\AllposMatrix.txt", vec_size=50, distanceListnum=31)
    # cleanfuhao(".\wordvector\Apubmed_All_Ab.txt", ".\wordvector\processedApubmed_All_Ab.txt")
    # changeSpelling(".\wordvector\processedApubmed_All_Ab.txt", ".\wordvector\cpApubmed_All_Ab.txt")
    # cleanfuhao(".\Test2013\\Ksentence.txt",".\Test2013\\Kdsentence.txt")
    # cleanfuhao(".\Train2013\\Ksentence.txt", ".\Train2013\\Kdsentence.txt")
    # makefile(".\\word.txt", ".\\wordvector\\cpApubmed_All_Ab.txt")
    # deletedrug0(".\\Train2013\\trainCsentence_token.txt",".\\Train2013\\trainCsentence_token(deletedrug0).txt")
    # deletedrug0(".\\Test2013\\testCsentence_token.txt",".\\Test2013\\testCsentence_token(deletedrug0).txt")
    # getsubsection(".\\Test2013\\testCsentence_token.txt",".\\Test2013\\testCsentence_token1.txt",".\\Test2013\\testCsentence_token2.txt",".\\Test2013\\testCsentence_token3.txt")
    # getsubsection(".\\Train2013\\trainCsentence_token.txt",".\\Train2013\\trainCsentence_token1.txt",".\\Train2013\\trainCsentence_token2.txt",".\\Train2013\\trainCsentence_token3.txt")

    # find_max_sentence(".\\Test2013\\testCsentence_token.txt")
    # find_max_sentence(".\\Test2013\\testCsentence_token2.txt")
    # find_max_sentence(".\\Test2013\\testCsentence_token3.txt")
    # find_max_sentence(".\\Train2013\\trainCsentence_token1.txt")
    # find_max_sentence(".\\Train2013\\trainCsentence_token2.txt")
    # find_max_sentence(".\\Train2013\\trainCsentence_token3.txt")
    # find_max_sentence(".\\Train2013\\trainCsentence_token.txt")
    # find_max_sentence(".\\Test2013\\testCsentence_token.txt")
if __name__ == "__main__":
    main()

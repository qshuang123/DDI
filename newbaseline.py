from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import os
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF

from guo73 import liter_dimension, liter_dimension_output_shape, changeshapek, changeshapek_output_shape, \
    weightedwordvector, weightedwordvector_output_shape

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
from keras.optimizers import RMSprop
from keras import Input, models
from keras.layers import Dense, Embedding, Bidirectional, GRU, TimeDistributed, Lambda, Dropout,Flatten, LSTM
from keras import backend as K
from keras.models import Model
import warnings
from Evalution import loadResult, getpart_prf, get_prf
from wordtovector import loadInstance, get_wordvector_input, get_embedding_wordvector, loadInstancePosition, \
    get_embedding_Positionvector, get_position_input, get_entityvector_input, get_positionweight_input, produce_matrix, \
    loaddependecySentence, getdependencysentence
# from keras.activations import relu
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(level=logging.INFO, filename='BASELSTM_POSTIONWEIGHT.log')
#全局变量
maxlen=154
max_features=20000
EMBEDDING_DIM=200
batch_size=256
#路径
wordVectorPath = "./wordvector/public/wikipedia-pubmed-and-PMC-w2v"

trainIstanceDrugpath = "./Train2013/dIstance.txt"
testIstanceDrugpath = "./Test2013/Istance.txt"

trainSentencePath = "./Train2013/trainCsentence_token.txt"
testSentencePath = "./Test2013/testCsentence_token.txt"

trainIstanceDrugPosition1Processed = "./Train2013/ProcessedtrainIstanceDrugPosition1.txt"
trainIstanceDrugPosition2Processed = "./Train2013/ProcessedtrainIstanceDrugPosition2.txt"
testIstanceDrugPosition1Processed = "./Test2013/ProcessedtestIstanceDrugPosition1.txt"
testIstanceDrugPosition2Processed = "./Test2013/ProcessedtestIstanceDrugPosition2.txt"

trainSDPpath="./Train2013/train_gdep11.txt_sdp"
testSDPpath="./Test2013/test_gdep1.txt_sdp"

testIstanceDrugPositionweightedProcessed="./Test2013/ProcessedtestPositionweighted.txt"
trainIstanceDrugPositionweightedProcessed="./Train2013/ProcessedtrainPositionweighted.txt"


def buildmodel():
    #模型输入
    position_input = Input(shape=(154,), dtype='float32', name='position_input')  # (?,154)
    print("position_input:", position_input.shape)
    position_weight = Lambda(changeshapek, output_shape=changeshapek_output_shape)(position_input)  # (?,154,1)
    print("positionweight:", position_weight.shape)
    weight = Lambda(liter_dimension, output_shape=liter_dimension_output_shape)(position_weight)  # (?,154,200)
    print("weight:", weight.shape)
    # wordvector input
    main_input = Input(shape=(154,), dtype='float32', name='main_input')#(?,154)
    embedding_layer=Embedding(num_word + 1, 200, mask_zero=True,trainable=True, weights=[embedding_matrix])
    wordVector = embedding_layer(main_input)  # (?,154,200)
    print("wordVector:",wordVector.shape)

    # position weughted wordvector

    x=keras.layers.Multiply()([wordVector,weight])


    # lstm
    z = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, activation='relu'))(x)  # (?,?,600)

    print("biGRU z:", z.shape)

    main_output = Dense(5, activation='softmax', name='main_output')(z)   # (?,5)
    print("main_output:", main_output.shape)
    model = Model(inputs= [main_input,position_input], outputs=main_output)

    model.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def onehotDecoder(predicted):
    predict=[]
    for p in predicted:
        q = p.tolist()
        i = q.index(max(q))
        if i == 0:
            instanceResult = "none"
        if i == 1:
            instanceResult = "mechanism"
        if i == 2:
            instanceResult = "effect"
        if i == 3:
            instanceResult = "advise"
        if i == 4:
            instanceResult = "int"
        predict.append(instanceResult)
    return predict

def trainModel(model, Traininput, traininstanceResult, Testinput,testinstanceResult,TrainInstancePosition,TestInstancePosition):
    for i in range(100):
        print("迭代次数：", i+1)
        model.fit({'main_input':Traininput,'position_input':TrainInstancePosition}, {'main_output': traininstanceResult}, epochs=1, batch_size=batch_size)

        print("predict!!")
        predicted = model.predict({'main_input':Testinput, 'position_input':TestInstancePosition}, verbose=1)
        print("write predict!!")

        predict=onehotDecoder(predicted)
        trueresult=onehotDecoder(testinstanceResult)

        print("get effect prf:")
        getpart_prf("effect", trueresult, predict)
        print("get mechanism prf:")
        getpart_prf("mechanism", trueresult, predict)
        print("get advise prf:")
        getpart_prf("advise", trueresult, predict)
        print("get int prf:")
        getpart_prf("int", trueresult, predict)
        print("get none prf:")
        getpart_prf("none", trueresult, predict)
        print("liu prf:")  # 这是预测ddi的prf
        p, r, f = get_prf(trueresult, predict)

        if f > 0.7:

            f = str(f)
            ModelFName = './model/baseModel' + "epoch-" + str(i) + "F-" + f + ".h5"
            MODELsaveFILEM = open(ModelFName, "w")
            model.save_weights(ModelFName)

            FNAME = 'predictions-task' + "Epoch" + str(i) + '.txt'
            PREDICTIONSFILE = open("./predicts/base" + FNAME, "w")
            for p in predicted:
                q = p.tolist()
                j = q.index(max(q))
                if j == 0:
                    instanceResult = "none"
                if j == 1:
                    instanceResult = "mechanism"
                if j == 2:
                    instanceResult = "effect"
                if j == 3:
                    instanceResult = "advise"
                if j == 4:
                    instanceResult = "int"
                PREDICTIONSFILE.write("{}\n".format(instanceResult))
            PREDICTIONSFILE.close()

def predict(Testinput):
    for i in range(1):
        ModelFName = '.\\loadmodel\\inputAttentionModelepoch-20F-0.7041564792176038.h5'
        print("load model !!")
        PREDICTIONSFILEM = open(ModelFName, "rt")
        model.load_weights(ModelFName)
        print("predict!!")
        predicted = model.predict({'main_input': Testinput}, verbose=0)
        print("write predict!!")
        FNAME = 'predictions-task' +  "0.704" + str(i) + '.txt'
        PREDICTIONSFILE = open(".\\loadpredicts\\" + FNAME, "w")
        for p in predicted:
            q=p.tolist()
            i=q.index(max(q))
            if i==0 :
                instanceResult="none"
            if i==1 :
                instanceResult="mechanism"
            if i==2 :
                instanceResult="effect"
            if i==3 :
                instanceResult="advise"
            if i==4 :
                instanceResult="int"
            PREDICTIONSFILE.write("{}\n".format(instanceResult))
        PREDICTIONSFILE.close()
    return predicted

def evalution(predicted,testinstanceResult):
    predict = onehotDecoder(predicted)
    trueresult = onehotDecoder(testinstanceResult)

    print("get effect prf:")
    getpart_prf("effect", trueresult, predict)
    print("get mechanism prf:")
    getpart_prf("mechanism", trueresult, predict)
    print("get advise prf:")
    getpart_prf("advise", trueresult, predict)
    print("get int prf:")
    getpart_prf("int", trueresult, predict)
    print("get none prf:")
    getpart_prf("none", trueresult, predict)
    print("liu prf:")  # 这是预测ddi的prf
    F = get_prf(trueresult, predict)
    return F

if __name__ == "__main__":

# 加载实例
    print("加载实例：")
    traininstance, traininstanceResult,entity1train,entity2train = loadInstance(trainIstanceDrugpath,trainSentencePath)
    print("instance:", traininstance[0])
    print("instanceResult:", traininstanceResult[0])
    print("entity1train:",entity1train[0])
    print("entity2train:",entity2train[0])
    # 加载测试集实例
    testinstance, testinstanceResult,entity1test,entity2test = loadInstance(testIstanceDrugpath,testSentencePath)
    print("instance:", testinstance[0])
    print("instanceResult:", testinstanceResult[0])
    print("entity1test:", entity1test[0])
    print("entity2test:", entity2test[0])

    # 加载位置信息
    print("加载位置信息：")
    TrainInstancePosition=loadInstancePosition(trainIstanceDrugPositionweightedProcessed)
    TestInstancePosition=loadInstancePosition(testIstanceDrugPositionweightedProcessed)

    # 获取词输入
    print("获取词输入：")
    Traininput, Testinput, word_index,tk = get_wordvector_input(traininstance,testinstance)
    print("traininput shape:", Traininput.shape)
    print("Traininput[0]:", Traininput[0])
    print("testinput shape:", Testinput.shape)
    print("Testinput[0]:", Testinput[0])



    # 获取词向量embedding input
    print("获取词向量embedding input:")
    embedding_matrix, num_word = produce_matrix(tk)
    print("embedding_matrix shape:",embedding_matrix.shape)


    # build simple model
    model = buildmodel()
    trainModel(model, Traininput, traininstanceResult, Testinput,testinstanceResult,TrainInstancePosition,TestInstancePosition)
    # 根据模型预测
    # predicted = predict(Testinput)

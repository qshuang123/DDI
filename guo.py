from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
import os
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
from keras.optimizers import RMSprop
from keras import Input, models
from keras.layers import Dense, Embedding, Bidirectional, GRU, TimeDistributed, Lambda, Dropout,Flatten
from keras import backend as K
from keras.activations import softmax
from keras.models import Model,load_model
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D
import warnings
from Evalution import loadResult, getpart_prf, get_prf
from wordtovector import loadInstance, get_wordvector_input, get_embedding_wordvector, loadInstancePosition, \
    get_embedding_Positionvector, get_position_input, get_entityvector_input, get_positionweight_input, produce_matrix, \
    loaddependecySentence, getdependencysentence

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(level=logging.INFO, filename='GUO73_POS_WEIGHT.log')
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

def batchdot(multituple):
    d=K.batch_dot(multituple[0],multituple[1],axes=[1,1])

    return d

def batchdot_output_shape(input_shape):
    shape = list(input_shape[1])
    shape=list(shape)
    shape[1]=600
    assert len(shape) == 3
    shape.pop(1)
    return tuple(shape)

def batchdotA(multituple):
    d=K.batch_dot(multituple[0],multituple[1],axes=[2,2])
    print('ddddddddd',d.shape)
    #d=K.tanh(d)
    return d

def batchdotA_output_shape(input_shape):
    shape = list(input_shape[1])
    shape=list(shape)
    shape[1]=28
    shape[2]=154
    assert len(shape) == 3
    return tuple(shape)

def batchdotB(multituple):
    d=K.batch_dot(multituple[0],multituple[1],axes=[2,1])
    return d

def batchdotB_output_shape(input_shape):
    shape = list(input_shape[1])
    shape=list(shape)
    shape[1]=154
    shape[2]=600
    assert len(shape) == 3
    return tuple(shape)

def batchdotC(multituple):
    d = multituple[0] * multituple[1]
    return d

def batchdotC_output_shape(input_shape):
    shape = list(input_shape[1])
    shape=list(shape)
    assert len(shape) == 3
    return tuple(shape)

def softmaxattention(x):
    x=K.softmax(x)
    print("softmaxattention shape:", x.shape)
    return x
def softmax_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2
    return tuple(shape)

def softmaxA(x):
    x = K.permute_dimensions(x,[0,2,1])
    x = K.softmax(x)
    x = K.permute_dimensions(x, [0, 2, 1])
    print("softmaxA shape:", x.shape)
    return x
def softmaxA_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return tuple(shape)

def multidot(multituple):
    d = multituple[0] * multituple[1]
    print("d shape:", d.shape)
    return d
def multidot_output_shape(input_shape):
    shape = list(input_shape)
    shape=list(shape[0])
    assert len(shape) == 3
    return tuple(shape)

def liter_dimension(x):
    res = K.repeat_elements(x, 200, axis=2)
    print("repeat res:",res.shape)
    return res
def liter_dimension_output_shape(input_shape):
    shape = list(input_shape)
    shape[2]=200
    assert len(shape) == 3
    return tuple(shape)

def changeshape(x):
    x=K.reshape(x,[-1,154,600])
    return x
def changeshape_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = 154
    assert len(shape) == 3
    return tuple(shape)

def changeshapeU(x):
    x=K.reshape(x,[-1,154])
    return x
def changeshapeU_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape.pop(1)
    return tuple(shape)

def weightedwordvector(multituple):
    d=multituple[0] * multituple[1]
    print("d:",d.shape)
    return d

def weightedwordvector_output_shape(input_shape):
    shape = list(input_shape[1])
    assert len(shape) == 3
    return tuple(shape)

def changeshapek(x):
    x=K.reshape(x, [-1, 154, 1])
    return x
def changeshapek_output_shape(input_shape):
    shape = list(input_shape)
    shape.append(1)
    assert len(shape) == 3
    return tuple(shape)



def changeshapeC(x):
    x = K.repeat_elements(x, 154*600, axis=-1)
    x=K.reshape(x,[-1,154,600])
    return x
def changeshapeC_output_shape(input_shape):
    shape = list(input_shape)
    shape[1]=154
    shape.append(600)
    assert len(shape) == 3
    return tuple(shape)



def changeshapeC2(x):
    x = K.repeat_elements(x, 28*600, axis=-1)
    x=K.reshape(x,[-1,28,600])
    return x
def changeshapeC_output_shape2(input_shape):
    shape = list(input_shape)
    shape[1]=28
    shape.append(600)
    assert len(shape) == 3
    return tuple(shape)


def get_g2(x):
    print(x.shape[0])
    y=1-x
    y=K.repeat_elements(y, 28*600, axis=-1)
    y=K.reshape(y,[-1,28,600])
    return y

def get_g2_output_shape(input_shape):
    shape = list(input_shape)
    shape[1]=28
    shape.append(600)
    assert len(shape) == 3
    return tuple(shape)


def myconn(x):
    print(x[0])
    print(x[1])
    y=K.concatenate([x[0],x[1]],1)
    print(y)
    return y

def myconn_output_shape(input_shape):
    shape = list(input_shape)
    shape = list(shape[0])
    shape[1]=182
    shape[2]=600
    assert len(shape) == 3
    return tuple(shape)

def buildmodel():
    #模型输入
    position_input = Input(shape=(154,), dtype='float32', name='position_input')# (?,154)
    print("position_input:",position_input.shape)
    position_weight = Lambda(changeshapek, output_shape=changeshapek_output_shape)(position_input)  # (?,154,1)
    print("positionweight:", position_weight.shape)
    weight = Lambda(liter_dimension, output_shape=liter_dimension_output_shape)(position_weight)  # (?,154,200)
    print("weight:",weight.shape)
    # wordvector input
    main_input = Input(shape=(154,), dtype='float32', name='main_input')#(?,154)
    embedding_layer=Embedding(num_word + 1, 200, mask_zero=False,trainable=False, weights=[embedding_matrix])
    wordVector = embedding_layer(main_input)  # (?,154,200)
    print("wordVector:",wordVector.shape)

    #SDP input
    SDP_input=Input(shape=(28,), dtype='float32', name='SDP_input')
    SDPvector=embedding_layer(SDP_input)

    x=keras.layers.Multiply()([wordVector,weight])


    # lstm
    z = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, return_sequences=True,activation='relu'))(x)  # (?,?,600)
    print("biGRU z:", z.shape)


    s = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, activation='relu',return_sequences=True))(SDPvector)  # (?,?,600)
    print("s.GRU:", s.shape)

    #矩阵
    multituple13 = [s,z]
    a = Lambda(batchdotA, output_shape=batchdotA_output_shape,name="sen_sdp_matric")(multituple13)
    a=keras.layers.Reshape([154,28])(a)
    a=Dense(1,activation='tanh',name="attention_dense")(a)# (?,154,28)
    a=keras.layers.Reshape([154])(a)
    a=Dense(154,activation='softmax',name="attention")(a)
    d1 =keras.layers.Dot(axes=[1,1])([a,z])#(?,28,600)
    d1 =keras.layers.Reshape([600],name="attention_dot")(d1)


    output = Dropout(0.5)(d1)

    main_output = Dense(5, activation='softmax', name='main_output')(output)   # (?,5)
    print("main_output:", main_output.shape)
    model = Model(inputs=[position_input, main_input,SDP_input], outputs=main_output)

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

def trainModel(model, Traininput,Train_Positionweight_input, traininstanceResult, Testinput, Test_Positionweight_input,testinstanceResult,trainSDPinput, testSDPinput):
    for i in range(100):
        print("迭代次数：", i+1)
        logging.info('epoch' + str(i+1))
        model.fit({'position_input': Train_Positionweight_input, 'main_input':Traininput,'SDP_input':trainSDPinput}, {'main_output': traininstanceResult}, epochs=1, shuffle=True, batch_size=batch_size)

        # ModelFName = '.\\model\\positionweighted+endsematten\\inputAttentionModelepoch-57F-0.6907378335949764.h5'
        # print("load model !!")
        # PREDICTIONSFILEM = open(ModelFName, "rt")
        # model.load_weights(ModelFName)
        print("predict!!")
        predicts = model.predict({'position_input': Test_Positionweight_input,'main_input':Testinput,'SDP_input':testSDPinput}, verbose=0)
        print("write predict!!")


        predict=onehotDecoder(predicts)
        trueresult=onehotDecoder(testinstanceResult)

        # print("get effect prf:")
        # logging.info("get effect prf:")
        # p, r, f = getpart_prf("effect", trueresult, predict)
        # logging.info("Epoch: " + str(i) + "p-" + str(p))
        # logging.info("Epoch: " + str(i) + "r-" + str(r))
        # logging.info("Epoch: " + str(i) + "F-" + str(f))
        #
        # print("get mechanism prf:")
        # logging.info("get mechanism prf:")
        # p, r, f = getpart_prf("mechanism", trueresult, predict)
        # logging.info("Epoch: " + str(i) + "p-" + str(p))
        # logging.info("Epoch: " + str(i) + "r-" + str(r))
        # logging.info("Epoch: " + str(i) + "F-" + str(f))
        #
        # print("get advise prf:")
        # logging.info("get advise prf:")
        # p, r, f = getpart_prf("advise", trueresult, predict)
        # logging.info("Epoch: " + str(i) + "p-" + str(p))
        # logging.info("Epoch: " + str(i) + "r-" + str(r))
        # logging.info("Epoch: " + str(i) + "F-" + str(f))
        #
        # print("get int prf:")
        # logging.info("get int prf:")
        # ss="int"
        # p,r,f = getpart_prf(ss, trueresult, predict)
        # logging.info("Epoch: " + str(i) + "p-" + str(p))
        # logging.info("Epoch: " + str(i) + "r-" + str(r))
        # logging.info("Epoch: " + str(i) + "F-" + str(f))
        #
        # print("get none prf:")
        # logging.info("get none prf:")
        # p, r, f = getpart_prf("none", trueresult, predict)
        # logging.info("Epoch: " + str(i) + "p-" + str(p))
        # logging.info("Epoch: " + str(i) + "r-" + str(r))
        # logging.info("Epoch: " + str(i) + "F-" + str(f))

        print("liu prf:")  # ????ddi?prf
        logging.info("get final prf:")
        p, r, f = get_prf(trueresult, predict)
        logging.info("Epoch: " + str(i) + "p-" + str(p))
        logging.info("Epoch: " + str(i) + "r-" + str(r))
        logging.info("Epoch: " + str(i) + "F-" + str(f))

        if f>0.7:
            ModelFName = '.\model\Model' + "F-" + str(f) + "epoch-" + str(i) + ".h5"
            MODELsaveFILEM = open(ModelFName, "w")
            model.save_weights(ModelFName)

            FNAME = 'predictions-task' + "Epoch" + str(i) + '.txt'
            PREDICTIONSFILE = open(".\predicts\\" + FNAME, "w")

            predicted = predicts
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

        # if F > 0.73 :
            # model_weighted_attention_dot = predicts[-1]
            # model_weighted_attention = predicts[-2]
            # model_weighted_attention_dense = predicts[-3]
            # model_weighted_sen_sdp_matric = predicts[-4]

            # print("model_weighted_attention_dot:",model_weighted_attention_dot)
            # print("model_weighted_attention:", model_weighted_attention)
            # print("model_weighted_attention_dense:", model_weighted_attention_dense)
            # print("model_weighted_sen_sdp_matric:", model_weighted_sen_sdp_matric)

            # model_weight_filepath = open(".\\modelweight\\"+str(F)+"model_weighted_attention_dot"+".txt","w")
            # model_weight_filepath.writelines(model_weighted_attention_dot)
            # model_weight_filepath.close()
            # model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_attention" + ".txt", "w")
            # model_weight_filepath.writelines(model_weighted_attention)
            # model_weight_filepath.close()
            # model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_attention_dense" + ".txt", "w")
            # model_weight_filepath.writelines(model_weighted_attention_dense)
            # model_weight_filepath.close()
            # model_weight_filepath = open(".\\modelweight\\" + str(F) + "model_weighted_sen_sdp_matric" + ".txt", "w")
            # model_weight_filepath.writelines(model_weighted_sen_sdp_matric)
            # model_weight_filepath.close()


def predict(Test_Positionweight_input,Testinput, testSDPinput):
    for i in range(1):
        ModelFName = './loadmodel/bestModelepoch-90F-0.7371631926792069.h5'
        print("load model !!")
        PREDICTIONSFILEM = open(ModelFName, "rt")
        model.load_weights(ModelFName)
        print("predict!!")
        predicts = model.predict({'position_input': Test_Positionweight_input,'main_input':Testinput,'SDP_input':testSDPinput}, verbose=0)

        print("write predict!!")
        FNAME = 'predictions-task' +  "0.7370000" + str(i) + '.txt'
        PREDICTIONSFILE = open("./loadpredicts/" + FNAME, "w")
        predicted = predicts
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
        # model_weighted_attention_dot = predicts[-1][473]
        # model_weighted_attention = predicts[-2][473]
        # model_weighted_attention_dense = predicts[-3][473]
        # model_weighted_sen_sdp_matric = predicts[-4][473]
        #
        # print("model_weighted_attention_dot:", model_weighted_attention_dot)
        # print("model_weighted_attention:", model_weighted_attention)
        # print("model_weighted_attention_dense:", model_weighted_attention_dense)
        # print("model_weighted_sen_sdp_matric:", model_weighted_sen_sdp_matric)


    return predicts

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

    # 加载SDP
    trainSDP = loaddependecySentence(trainSDPpath)
    print("trainSDP:",len(trainSDP))
    testSDP = loaddependecySentence(testSDPpath)
    print("testSDP:",len(testSDP))


    # 获取词输入
    print("获取词输入：")
    Traininput, Testinput, word_index,tk = get_wordvector_input(traininstance,testinstance)
    print("traininput shape:", Traininput.shape)
    print("Traininput[0]:", Traininput[0])
    print("testinput shape:", Testinput.shape)
    print("Testinput[0]:", Testinput[0])
    # 获取位置信息的输入
    print("获取位置的输入：")
    Train_Positionweight_input = loadInstancePosition(trainIstanceDrugPositionweightedProcessed)
    Test_Positionweight_input = loadInstancePosition(testIstanceDrugPositionweightedProcessed)



    # 获取词向量embedding input
    print("获取词向量embedding input:")
    embedding_matrix, num_word = produce_matrix(tk)
    print("embedding_matrix shape:",embedding_matrix.shape)

    # SDP input
    trainSDPinput, testSDPinput = getdependencysentence(tk, trainSDP, testSDP)
    print("trainSDPinput:",trainSDPinput.shape)
    print("testSDPinput:",testSDPinput.shape)
    # build simple model
    model = buildmodel()
    trainModel(model, Traininput,Train_Positionweight_input, traininstanceResult, Testinput, Test_Positionweight_input,testinstanceResult,trainSDPinput, testSDPinput)
    # 根据模型预测
    # predict(Test_Positionweight_input, Testinput, testSDPinput)
    # # evaltuion
    # # for i in range(1):
    # evalution(predicted, testinstanceResult)

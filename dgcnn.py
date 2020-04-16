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
logging.basicConfig(level=logging.INFO, filename='log.log')

maxlen=154
max_features=20000
EMBEDDING_DIM=200
batch_size=256
#Â·¾¶
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


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback


class OurLayer(Layer):
    """¶¨ÒåÐÂµÄLayer£¬Ôö¼Óreuse·½·¨£¬ÔÊÐíÔÚ¶¨ÒåLayerÊ±µ÷ÓÃÏÖ³ÉµÄ²ã
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class AttentionPooling1D(OurLayer):
    """Í¨¹ý¼ÓÐÔAttention£¬½«ÏòÁ¿ÐòÁÐÈÚºÏÎªÒ»¸ö¶¨³¤ÏòÁ¿
    """
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if self.h_dim is None:
            self.h_dim = input_shape[0][-1]
        self.k_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o_dense = Dense(1, use_bias=False)
    def call(self, inputs):
        xo, mask = inputs
        x = xo
        x = self.reuse(self.k_dense, x)
        x = self.reuse(self.o_dense, x)
        x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        return K.sum(x * xo, 1)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][-1])


class DilatedGatedConv1D(OurLayer):
    """ÅòÕÍÃÅ¾í»ý£¨DGCNN£©
    """
    def __init__(self,
                 o_dim=None,
                 k_size=3,
                 rate=1,
                 skip_connect=True,
                 drop_gate=None,
                 **kwargs):
        super(DilatedGatedConv1D, self).__init__(**kwargs)
        self.o_dim = o_dim
        self.k_size = k_size
        self.rate = rate
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
    def build(self, input_shape):
        super(DilatedGatedConv1D, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[0][-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.rate,
            padding='same'
        )
        if self.skip_connect and self.o_dim != input_shape[0][-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)
    def call(self, inputs):
        xo, mask = inputs
        x = xo * mask
        x = self.reuse(self.conv1d, x)
        x, g = x[..., :self.o_dim], x[..., self.o_dim:]
        if self.drop_gate is not None:
            g = K.in_train_phase(K.dropout(g, self.drop_gate), g)
        g = K.sigmoid(g)
        if self.skip_connect:
            if self.o_dim != K.int_shape(xo)[-1]:
                xo = self.reuse(self.conv1d_1x1, xo)
            return (xo * (1 - g) + x * g) * mask
        else:
            return x * g * mask
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.o_dim,)


def buildmodel():
    #Ä£ÐÍÊäÈë
    # wordvector input
    main_input = Input(shape=(154,), dtype='float32', name='main_input')#(?,154)

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(main_input)

    embedding_layer=Embedding(num_word + 1, 200, trainable=True, weights=[embedding_matrix])

    wordVector = embedding_layer(main_input)  # (?,154,200)

    print("input shape:", wordVector.shape)
    print("mask shape:", mask.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([wordVector, mask])
    print("q1 shape:", q.shape)
    q = DilatedGatedConv1D(rate=2, drop_gate=0.1)([q, mask])
    print("q2 shape:", q.shape)
    q = DilatedGatedConv1D(rate=3, drop_gate=0.1)([q, mask])
    print("q3 shape:", q.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, mask])
    print("q4 shape:", q.shape)
    q = DilatedGatedConv1D(rate=2, drop_gate=0.1)([q, mask])
    print("q5 shape:", q.shape)
    q = DilatedGatedConv1D(rate=3, drop_gate=0.1)([q, mask])
    print("q6 shape:", q.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, mask])
    print("q7 shape:", q.shape)
    q = DilatedGatedConv1D(rate=2, drop_gate=0.1)([q, mask])
    print("q8 shape:", q.shape)
    q = DilatedGatedConv1D(rate=3, drop_gate=0.1)([q, mask])
    print("q9 shape:", q.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, mask])
    print("q10 shape:", q.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, mask])
    print("q11 shape:", q.shape)
    q = DilatedGatedConv1D(rate=1, drop_gate=0.1)([q, mask])
    print("q12 shape:", q.shape)
    p = AttentionPooling1D()([q, mask])
    print("p shape:", p.shape)

    # # lstm
    # z = Bidirectional(GRU(300, dropout=0.5, recurrent_dropout=0.5, activation='relu'))(wordVector)  # (?,?,600)
    #
    # print("biGRU z:", z.shape)

    main_output = Dense(5, activation='softmax', name='main_output')(p)   # (?,5)
    print("main_output:", main_output.shape)
    model = Model(inputs= main_input, outputs=main_output)

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

def trainModel(model, Traininput, traininstanceResult, Testinput,testinstanceResult):
    for i in range(100):
        print("µü´ú´ÎÊý£º", i+1)

        logging.info('epoch' + str(i+1))
        model.fit({'main_input':Traininput}, {'main_output': traininstanceResult}, epochs=1, batch_size=batch_size)

        # ModelFName = '.\\loadmodel\\inputAttentionModelepoch-23F-0.7281648675171737.h5'
        # print("load model !!")
        # PREDICTIONSFILEM = open(ModelFName, "rt")
        # model.load_weights(ModelFName)
        print("predict!!")
        predicted = model.predict({'main_input':Testinput}, verbose=0)

        predict=onehotDecoder(predicted)
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

        if f>0.7 :
            F = str(f)
            ModelFName = './model/base_dgcnn_Model' + "epoch-" + str(i) + "F-" + F + ".h5"
            MODELsaveFILEM = open( ModelFName, "w")
            model.save_weights(ModelFName)

            print("write predict!!")
            FNAME = 'predictions-task' + "Epoch" + str(i) + '.txt'
            PREDICTIONSFILE = open("./predicts/base_dgcnn_" + FNAME, "w")
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
    print("liu prf:")  # ÕâÊÇÔ¤²âddiµÄprf
    F = get_prf(trueresult, predict)
    return F

if __name__ == "__main__":

# ¼ÓÔØÊµÀý
    print("¼ÓÔØÊµÀý£º")
    traininstance, traininstanceResult,entity1train,entity2train = loadInstance(trainIstanceDrugpath,trainSentencePath)
    print("instance:", traininstance[0])
    print("instanceResult:", traininstanceResult[0])
    print("entity1train:",entity1train[0])
    print("entity2train:",entity2train[0])
    # ¼ÓÔØ²âÊÔ¼¯ÊµÀý
    testinstance, testinstanceResult,entity1test,entity2test = loadInstance(testIstanceDrugpath,testSentencePath)
    print("instance:", testinstance[0])
    print("instanceResult:", testinstanceResult[0])
    print("entity1test:", entity1test[0])
    print("entity2test:", entity2test[0])


    # »ñÈ¡´ÊÊäÈë
    print("»ñÈ¡´ÊÊäÈë£º")
    Traininput, Testinput, word_index,tk = get_wordvector_input(traininstance,testinstance)
    print("traininput shape:", Traininput.shape)
    print("Traininput[0]:", Traininput[0])
    print("testinput shape:", Testinput.shape)
    print("Testinput[0]:", Testinput[0])



    # »ñÈ¡´ÊÏòÁ¿embedding input
    print("»ñÈ¡´ÊÏòÁ¿embedding input:")
    embedding_matrix, num_word = produce_matrix(tk)
    print("embedding_matrix shape:",embedding_matrix.shape)


    # build simple model
    model = buildmodel()
    trainModel(model, Traininput, traininstanceResult, Testinput,testinstanceResult)


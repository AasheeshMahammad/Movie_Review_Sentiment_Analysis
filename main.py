import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
from spacy import Vocab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout,SpatialDropout1D,Embedding
import matplotlib.pyplot as plt

USE_GPU = True

def plot(history):
    accuracy = history['accuracy']
    validationAccuracy = history['val_accuracy']
    loss = history['loss']
    validationLoss = history['val_loss']
    #print(accuracy,validationAccuracy,loss,validationLoss)
    plt.plot(accuracy,label='acc')
    plt.plot(validationAccuracy,label='val_acc')
    plt.plot(loss,label='loss')
    plt.plot(validationLoss,label='val_loss')
    plt.legend()
    plt.show()

def predict(text,maxLen,tokenizer,labels,model):
    tokenizedText = tokenizer.texts_to_sequences([text])
    pText = pad_sequences(tokenizedText,maxlen=maxLen)
    pred = int(model.predict(pText).round().item())
    val = labels[1][pred]
    print(val)

def getModel(vocabSize,inputSize,vectorLength=32):
    model = Sequential()
    recurrentDropout = 0.5
    activationFunction = 'sigmoid'
    if USE_GPU:
        recurrentDropout = 0
    model.add(Embedding(vocabSize,vectorLength,input_length=inputSize))
    model.add(SpatialDropout1D(0.25))
    model.add(LSTM(50,dropout=0.5,recurrent_dropout=recurrentDropout))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation=activationFunction))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def pad(encodedTexts):
    maxLen = 0
    for i in encodedTexts:
        maxLen = max(maxLen,len(i))
    padedSequences = pad_sequences(encodedTexts,maxlen=maxLen+5)
    inputSize = maxLen + 5
    return padedSequences,inputSize

def tokenize(sentence):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentence) # convert words to numbers
    #print(tokenizer.word_index) 
    encodedTexts = tokenizer.texts_to_sequences(sentence)  # replace words to token number
    return encodedTexts,len(tokenizer.word_counts),tokenizer

def filter(df):
    reviewDf = df[['text','airline_sentiment']]
    reviewDf = reviewDf.rename(columns = {'airline_sentiment':'sentiment'})
    reviewDf = reviewDf[reviewDf['sentiment'] != 'neutral']
    label = reviewDf['sentiment'].factorize()
    sentence = reviewDf.text.values
    return label,sentence

def test():
    df = pd.read_csv('./data/Tweets.csv')
    labels,sentence = filter(df)
    encodedTexts,vocabSize,tokenizer = tokenize(sentence)
    paddedSequences,inputSize = pad(encodedTexts)
    model = load_model('model')
    text = "the flight experience was enjoyed"
    predict(text,inputSize,tokenizer,labels,model)

def train():
    df = pd.read_csv('./data/Tweets.csv')
    labels,sentence = filter(df)
    encodedTexts,vocabSize,tokenizer = tokenize(sentence)
    paddedSequences,inputSize = pad(encodedTexts)
    model = getModel(vocabSize,inputSize)
    print(model.summary())
    #print(labels)
    history = model.fit(paddedSequences,labels[0],validation_split=0.2,epochs=5,batch_size=32)
    #plot(history.history)
    model.save('model')

def main():
    #train()
    test()

if __name__ == '__main__':
    main()



import re
import json, os
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.layers import Flatten,TimeDistributed,MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.utils import np_utils
from keras.optimizers import Adagrad,Adam
import gensim

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)




def ContinuousInterval(intervalL):  # Returns max continuous interval
    maxInt = []
    tempInt = [intervalL[0]]
    for q in range(1, len(intervalL)):
        if intervalL[q] - intervalL[q - 1] > 1:
            if len(tempInt) > len(maxInt):
                maxInt = tempInt
            tempInt = [intervalL[q]]
        else:
            tempInt.append(intervalL[q])
    if len(maxInt) == 0:
        maxInt = tempInt
    return maxInt



def remove_link(input_text):  # remove hyperlink
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    tweetText = pattern.sub(' ', input_text)
    return tweetText


def remove_pattern(input_txt, pattern):  # remove handles
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, ' ', input_txt)
    return input_txt


def remove_spchar(input_text):  # remove digits
    tweetText1 = re.sub('[0-9]', ' ', input_text)
    return tweetText1


def remove_emoji(input_text):  # remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', input_text)

def fcode():
    mypath = "twitter11"
    Evenlist = listdir(mypath)
    TIME_STEPS = 6
    InputWord = 15
    IMPUT_SIZE = 625
    BATCH_SIZE = 30
    BATCH_INDEX = 0
    OUTPUT_SIZE = 2
    CELL_SIZE = 175
    LR = 0.001

    output_fin = []
    totalData = []
    totalDataLabel = []
    counter = 0
    totalDoc = 0
    totalpost = 0
    tdlist1 = 0
    Pos = 0
    Neg = 0
    maxpost = 0
    minpost = 62827

    for event in Evenlist:

        totalDoc += 1
        print(event)
        fe = open(os.path.join("event", event+".json"), "r")

        EventJson = json.load(fe)
        Label = EventJson["label"]  # store label
        path = os.path.join("twitter11", event)
        TidList = listdir(path)
        TweetList = []

        if len(TidList) == 1:          # only one tweet id present
            tdlist1 += 1
            continue
        if len(TidList) >= maxpost:
            maxpost = len(TidList)
        if len(TidList) <= minpost:
            minpost = len(TidList)

        for TweetId in TidList:
            event = re.sub('.json', '', event)

            ft = open(os.path.join("twitter11", event, TweetId), "r")
            file1 = json.load(ft)
            tweet1 = file1['text']
            text1 = remove_pattern(tweet1, "@[\w]*")
            text2 = remove_link(text1)
            text3 = remove_emoji(text2)
            text4 = remove_spchar(text3)

            totalpost += 1
            Time = file1['time']
            TweetList.append({"text": text4, "time": Time})  # time and text stored in tweetlist

        if Label == 0:
            Pos += 1
        else:
            Neg += 1

        TweetList = sorted(TweetList, key=lambda k: k['time'])  # Sort by time

        TotalTimeLine = TweetList[-1]['time'] - TweetList[0]['time']

        IntervalTime = TotalTimeLine / TIME_STEPS
        k = 0
        PreConInt = []
        while True:
            k += 1
            tweetIndex = 0
            output = []
            if TotalTimeLine == 0:
                output.append(''.join(tweet["text"] for tweet in TweetList))
                break
            Start = TweetList[0]['time']
            Interval = int(TotalTimeLine / IntervalTime)

            Intset = []
            for inter in range(0, Interval):
                empty = 0
                interval = []

                for q in range(tweetIndex, len(TweetList)):  # to store text in that interval
                    if TweetList[q]['time'] >= Start and TweetList[q]['time'] < Start + IntervalTime:
                        empty += 1
                        interval.append(TweetList[q]["text"])

                    elif TweetList[q]['time'] >= Start + IntervalTime:
                        tweetIndex = q - 1
                        break

                if empty == 0:  # empty interval
                    output.append([])
                else:           # add the last tweet
                    if TweetList[-1]['time'] == Start + IntervalTime:
                        interval.append(TweetList[-1]["text"])
                    Intset.append(inter)
                    output.append(interval)
                Start = Start + IntervalTime  # update start

            ConInt = ContinuousInterval(Intset)  # find max continuous interval

            if len(ConInt) < TIME_STEPS and len(ConInt) > len(PreConInt):
                IntervalTime = int(IntervalTime * 0.5)
                PreConInt = ConInt
                if IntervalTime == 0:
                    output = output[ConInt[0]:ConInt[-1] + 1]
                    break
            else:
                output = output[ConInt[0]:ConInt[-1] + 1]
                break
        counter += 1

        fe.close()

        for q in range(0, len(output)):
            output[q] = ''.join(s for s in output[q])



        vectorizer = CountVectorizer(output,
                                     stop_words=["all", "in", "this", "and", "is", "as", "it", "so", "the", "we", "are",
                                                 "via", "you", "your"])

        tf = vectorizer.fit_transform(output)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(tf)

        Allvocabulary = vectorizer.get_feature_names()

        first_doc_vector = tfidf[0]       # get tfidf vec for first doc
        df = pd.DataFrame(first_doc_vector.T.todense(), index=Allvocabulary, columns=["tfidf"])
        #print(df.sort_values(by=["tfidf"], ascending=False))

        Input = []

        for interval in tfidf.toarray():
            interval = sorted(interval, reverse=True)
            while len(interval) < IMPUT_SIZE:
                interval.append(0.0)

            Input.append(interval[:IMPUT_SIZE])

        if len(Input) < TIME_STEPS:
            for q in range(0, TIME_STEPS - len(Input)):
                Input.insert(0, [0.0] * IMPUT_SIZE)

        totalData.append(Input[:TIME_STEPS])
        totalDataLabel.append(Label)

    """
        Input = []
        for interval in tfidf.toarray():
            Wordvector = []
            WordvectorMatrix = []
            NonZeroCount = 0
            interval, Allvocabulary = zip(*sorted(zip(interval, Allvocabulary), reverse=True))
    
            for word, value in zip(Allvocabulary, interval):
                if value != 0.0 and NonZeroCount < InputWord:  # IMPUT_SIZE
                    try:
                        Wordvector = np.append(Wordvector, model[word])
                        NonZeroCount += 1
                    except:
                        print("")
    
            while NonZeroCount < InputWord:
                Wordvector = np.append(Wordvector, ([0.0] * 300))
                NonZeroCount += 1
            Input.append(Wordvector)            # (WordvectorMatrix)
    
        if len(Input) < TIME_STEPS:
            for q in range(0, TIME_STEPS - len(Input)):
                Input.insert(0, [0.0] * 300 * InputWord)
        totalData.append(Input[:TIME_STEPS])
        totalDataLabel.append(Label)"""

    print(totalDataLabel)

    print("totalDoc : " + str(totalDoc))
    print("tdlist1 : " + str(tdlist1))
    print("Pos : " + str(Pos))
    print("Neg : " + str(Neg))
    print("totalpost : " + str(totalpost))
    print("maxpost : " + str(maxpost))
    print("minpost : " + str(minpost))
    print(len(totalData))
    print(counter)

    X_train = np.array(totalData[:int(counter/5*4)])

    y_train = np.array(totalDataLabel[:int(counter/5*4)])

    X_test = np.array(totalData[int(counter/5*4):])
    y_test = np.array(totalDataLabel[int(counter/5*4):])

    # LSTM
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    y_test = np_utils.to_categorical(y_test, num_classes=2)

    model = Sequential()
    """
    model.add(LSTM(CELL_SIZE, return_sequences=True, input_shape=(TIME_STEPS, IMPUT_SIZE)))
    model.add(TimeDistributed(Dense(1)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    """

    model.add(LSTM(CELL_SIZE, input_shape=(TIME_STEPS, IMPUT_SIZE)))
    model.summary()

    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))
    Adagrad = Adam(LR)

    model.compile(optimizer=Adagrad, loss='mean_squared_error', metrics=['accuracy'])

    # train
    print("Training---------")

    history = model.fit(X_train, y_train,validation_split=0.2, epochs=10, batch_size=BATCH_SIZE)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'], loc='upper right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train','test'], loc='upper right')
    plt.show()

    print("\nTesting---------")
    cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
    print('test cost: ', cost)
    print('test accuracy: ', accuracy)

    y_pred= model.predict(X_test)
    matrix = confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
    print(matrix)

    plt.matshow(matrix)
    plt.title('Confusion Matrix Plot')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


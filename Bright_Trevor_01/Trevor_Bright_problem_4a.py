#**************************#
#* Trevor Bright * #
#* CECS 590-01 * #
#* Assignment 1 * #
#************************* #
import numpy as np
import pandas as ps
import math
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
np.random.seed(seed)

def generateDataset():
    inCords = []
    outCords = []
    while len(inCords) < 100 or len(outCords) < 100:
        tempX = np.random.normal(.5, math.sqrt(.08))
        tempY = np.random.normal(.5, math.sqrt(.08))
        if (tempX >= 0 and tempY >= 0 and tempX < 1 and tempY < 1):
            if(tempX <= .5 and tempY <= .5):
                inCords.append([tempX,tempY])
            else:
                outCords.append([tempX,tempY])
    joined = inCords[:100]+outCords[:100]
    random.shuffle(joined)
    return joined


def rankDataset(dataset):
    labels = []
    for cords in dataset:
        if(cords[0] <= .5 and cords[1] <= .5):
            labels.append(1)
        else:
            labels.append(0)
    return labels

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

dataset = generateDataset()
labels = rankDataset(dataset)

np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, dataset, labels, cv=kfold)

print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
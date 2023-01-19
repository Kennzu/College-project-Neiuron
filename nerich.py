import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

#Read the data, turn it into lower case
data = open("C:/Python/Recurent/ottelo.txt").read().lower()

chars = sorted(list(set(data)))

totalChars = len(data)

numberOfUniqueChars = len(chars)


CharsForids = {char:Id for Id, char in enumerate(chars)}

idsForChars = {Id:char for Id, char in enumerate(chars)}

numberOfCharsToLearn = 100


charX = []

y = []

counter = totalChars - numberOfCharsToLearn

for i in range(0, counter, 1):

    theInputChars = data[i:i+numberOfCharsToLearn]

    theOutputChars = data[i + numberOfCharsToLearn]

    charX.append([CharsForids[char] for char in theInputChars])

    y.append(CharsForids[theOutputChars])


X = np.reshape(charX, (len(charX), numberOfCharsToLearn, 1))

X = X/float(numberOfUniqueChars)

y = np_utils.to_categorical(y)

model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=128)
model.save_weights("Othello.hdf5")
model.load_weights("Othello.hdf5")

randomVal = np.random.randint(0, len(charX)-1)
randomStart = charX[randomVal]
for i in range(500):
    x = np.reshape(randomStart, (1, len(randomStart), 1))
    x = x/float(numberOfUniqueChars)
    pred = model.predict(x)
    index = np.argmax(pred)
    randomStart.append(index)
    randomStart = randomStart[1: len(randomStart)]
print("".join([idsForChars[value] for value in randomStart]))

# with h5py.File('Othello.hdf5', 'r') as f:
#     print(f)
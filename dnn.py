from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
X_train=X_train/255.0
X_test=X_test/255.0

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()

#add model layers
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
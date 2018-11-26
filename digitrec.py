# Import keras.
import keras as kr
import numpy as np
import gzip
# For encoding categorical variables.
import sklearn.preprocessing as pre

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()

# Start a neural network, building it by layers.
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=500, activation='sigmoid', input_dim=784))
model.add(kr.layers.Dense(units=400, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=300, activation='relu', input_dim=784))
model.add(kr.layers.Dense(units=150, activation='tanh', input_dim=784))


# Add a three neuron output layer.
model.add(kr.layers.Dense(units=10, activation='sigmoid'))

# Build the graph.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])


    
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)/255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)

inputs = train_img.reshape(60000, 784)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

model.fit(inputs, outputs, epochs=10, batch_size=127)

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()
    
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum())


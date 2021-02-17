import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load data with headers
data = np.genfromtxt(
    "breast-cancer-wisconsin.data.txt",
    delimiter=",",
    names=True,
)

# Define what columns are considered features for
# describing the cells
feature_names = [
    "clump_thickness",
    'uniform_cell_size',
    'uniform_cell_shape',
    'marginal_adhesion',
    'single_epi_cell_size',
    'bare_nuclei',
    'bland_chromation',
    'normal_nucleoli',
    'mitoses'
]

# Gather features into one matrix (this is the "X" matrix in task)
features = []
for feature_name in feature_names:
    features.append(data[feature_name])
features = np.stack(features, axis=1)

# Do same for class
classes = data["class"]

# Remove non-numeric values (appear as NaNs in the data).
# If any feature in a sample is nan, drop that sample (row).
cleaned_features = []
cleaned_classes = []
for i in range(features.shape[0]):
    if not np.any(np.isnan(features[i])):
        cleaned_features.append(features[i])
        cleaned_classes.append(classes[i])
cleaned_features = np.stack(cleaned_features, axis=0)
cleaned_classes = np.array(cleaned_classes)

# Rename to match the exercises
X = cleaned_features
y = cleaned_classes

# Standardize features with standard scaling ([x - mean] / std)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Transform y into {0,1} label array.
# Turn 2 into 0 and 4 into 1
y = (y == 4).astype(np.int64)

# Split data into training and testing sets.
# You could also use e.g. scikit-learn to do this.
# Lets split the data 50-50.
indeces = np.arange(X.shape[0])
#np.random.shuffle(indeces)
# Take one half for training, second half for testing
#half_point = X.shape[0] // 2
train_indeces = indeces[:580]
test_indeces = indeces[580:]

X_train = X[train_indeces]
y_train = y[train_indeces]
print (y_train[0:4])
y_train = keras.utils.to_categorical(y_train, num_classes=2, dtype="float32")
print (X_train[0:4])
print (y_train[0:4])
X_test = X[test_indeces]
y_test = y[test_indeces]
y_test = keras.utils.to_categorical(y_test, num_classes=2, dtype="float32")

# single layer with 5, 10, 30 nodes
def singlelayer():
    score = []
    score1 = []
    ind = 0
    for i in [5,10,30]:
        nnmodel = keras.Sequential()
        nnmodel.add(keras.layers.Dense(i, activation="relu", name = "hlayer1"))
        nnmodel.add(keras.layers.Dense(2,activation='softmax'))
        nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        nnmodel.compile(optimizer = nnsgd, loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
        nnmodel.fit(X_train, y_train, batch_size=64, epochs=500)
        score.append(nnmodel.evaluate(X_train, y_train, batch_size=64))
        score1.append(nnmodel.evaluate(X_test, y_test, batch_size=64))
        ind = ind + 1
    print("the loss and binary accuracy in training for 5,10,30 respectively are: ",score)
    print("the loss and binary accuracy in testing for 5,10,30 respectively are: ",score1)
    
# 10layers with 5neurons in the first layer while 10 in each next hidden layers
def multilayer():
    nnmodel = keras.Sequential()
    nnmodel.add(keras.layers.Dense(5, activation="relu", name = "hlayer1"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer2"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer3"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer4"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer5"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer6"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer7"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer8"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer9"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer10"))
    nnmodel.add(keras.layers.Dense(2,activation='softmax'))
    nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    nnmodel.compile(optimizer = nnsgd, loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
    nnmodel.fit(X_train, y_train, batch_size=64, epochs=2000)
    score = nnmodel.evaluate(X_train, y_train, batch_size=64)
    score1 = nnmodel.evaluate(X_test, y_test, batch_size=64)
    print("the loss and binary accuracy in training are: ",score)
    print("the loss and binary accuracy in testing are: ",score1)
    

#using dropout from 0% 10 40%
def dropouts():
    d = 0
    score2 = []
    score3 = []
    
    for i in range(0, 4):
        nnmodel2 = keras.Sequential()
        nnmodel2.add(keras.layers.Dense(10, activation="relu", name = "hlayer1"))
        nnmodel2.add(keras.layers.Dropout(d))
        nnmodel2.add(keras.layers.Dense(10, activation="relu", name = "hlayer2"))
        nnmodel2.add(keras.layers.Dropout(d))
        nnmodel2.add(keras.layers.Dense(10, activation="relu", name = "hlayer3"))
        nnmodel2.add(keras.layers.Dropout(d))
        nnmodel2.add(keras.layers.Dense(10, activation="relu", name = "hlayer4"))
        nnmodel2.add(keras.layers.Dropout(d))
        nnmodel2.add(keras.layers.Dense(2,activation='softmax'))
        nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        nnmodel2.compile(optimizer = nnsgd, loss=keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
        nnmodel2.fit(X_train, y_train, batch_size=None, epochs=500)
        score2.append(nnmodel2.evaluate(X_train, y_train, batch_size=64))
        score3.append(nnmodel2.evaluate(X_test, y_test, batch_size=64))
        d = d + 0.1
    print("the loss and binary accuracy in training for the 4 dropoutrates are: ",score2)
    print("the loss and binary accuracy in testing for the 4 dropoutrates are: ",score3)

# using Relu activation function
def relu():
    nnmodel = keras.Sequential()
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer9"))
    nnmodel.add(keras.layers.Dense(10, activation="relu", name = "hlayer10"))
    nnmodel.add(keras.layers.Dense(2,activation='softmax'))
    nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    nnmodel.compile(optimizer = nnsgd, loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
    nnmodel.fit(X_train, y_train, batch_size=64, epochs=500)
    score = nnmodel.evaluate(X_train, y_train, batch_size=64)
    score1 = nnmodel.evaluate(X_test, y_test, batch_size=64)
    print("score on training data ", score)
    print("score on test data ", score1)

# using sigmoid activation function
def sigmd():
    nnmodel = keras.Sequential()
    nnmodel.add(keras.layers.Dense(10, activation="sigmoid", name = "hlayer9"))
    nnmodel.add(keras.layers.Dense(10, activation="sigmoid", name = "hlayer10"))
    nnmodel.add(keras.layers.Dense(2,activation='softmax'))
    nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    nnmodel.compile(optimizer = nnsgd, loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
    nnmodel.fit(X_train, y_train, batch_size=64, epochs=500)
    score = nnmodel.evaluate(X_train, y_train, batch_size=64)
    score1 = nnmodel.evaluate(X_test, y_test, batch_size=64)
    print("score on training data ", score)
    print("score on test data ", score1)

# using tanh activation function
def tanh():
    nnmodel = keras.Sequential()
    nnmodel.add(keras.layers.Dense(10, activation="tanh", name = "hlayer9"))
    nnmodel.add(keras.layers.Dense(10, activation="tanh", name = "hlayer10"))
    nnmodel.add(keras.layers.Dense(2,activation='softmax'))
    nnsgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    nnmodel.compile(optimizer = nnsgd, loss = keras.losses.CategoricalCrossentropy(), metrics = [keras.metrics.BinaryAccuracy()])
    nnmodel.fit(X_train, y_train, batch_size=64, epochs=500)
    score = nnmodel.evaluate(X_train, y_train, batch_size=64)
    score1 = nnmodel.evaluate(X_test, y_test, batch_size=64)
    print("score on training data ", score)
    print("score on test data ", score1)
from __future__ import print_function
import keras.backend as K
session = K.get_session()
K.set_session(session)
import numpy as np
import h5py
from scipy import ndimage
import scipy
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import History
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
K.set_image_dim_ordering('tf')

np.random.seed(1)

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_orig = train_x_orig/255.

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
print(num_px)
# Reshape the training and test examples
train_x_flatten = train_x_orig   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten
test_x = test_x_flatten/255.

train_y = train_y.reshape(-1, train_y.shape[0])
test_y = test_y.reshape(-1, test_y.shape[0])

labelencoder_Y_1 = LabelEncoder()
labelencoder_Y_1.fit(train_y)
train_y = labelencoder_Y_1.transform(train_y)
train_y = to_categorical(train_y)
test_y = labelencoder_Y_1.transform(test_y)
test_y = to_categorical(test_y)

train_y = train_y.reshape(len(train_y), 2)
test_y = test_y.reshape(len(test_y), 2)

# model parameters
NB_EPOCH = 100
BATCH_SIZE = 32
VERBOSE = 1
NB_CLASSES = 2
OPTIMIZER = Adam()
N_HIDDEN = 64

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax', name="result"))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))

history = History()
model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, callbacks=[history])

score = model.evaluate(test_x, test_y, verbose=VERBOSE)

my_image = "my_image3.jpg"  # my test image
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px, num_px, 3))
my_image = my_image/255.
my_predict = model.predict_classes(my_image)
if my_predict == 1:
    print("cat")
else:
    print("not cat")

model.save("trained_model.h5", include_optimizer=True)





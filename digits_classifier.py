"""
In this file we train a model that deal with digits classification

once the file finish his running, in output directory will be:
1. acc_and_loss: accuracy and loss plots of the training and validation data
2. best_model_weights_initial_CNN_model_mnist1: weights for initial using of the model
3. confusion_matrix: confusion_matrix of the predications
4. digit_classifier: the weights that evaluated form model and save after the fitting
5. test_predications: some plots of the test predications
6. training_predications some plots of the training predications

We will use the digit_classifier to predict the digits in the sudoku from the image.
"""
# ------------------ imports ------------------ #
import argparse
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer

import warnings
warnings.catch_warnings()
warnings.simplefilter("ignore")
# ------------------ imports ------------------ #

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model after training")
args = vars(ap.parse_args())

# download data from MNIST
print("[INFO] get data from MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# lets look at the data size:
print('X_train shape is: {} \n\
y_train shape is: {}'.format(X_train.shape, y_train.shape))
print()
print('X_test shape is: {} \n\
y_test shape is: {}'.format(X_test.shape, y_test.shape))

# scale data to the range of [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(y_train)
testLabels = le.transform(y_test)

# print some images and predictions
def plot_multiple_imgs(X, y, nrow=2, ncol=2, figsize=(13, 7), preds=None, skip=0, file_name=""):
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(nrow * ncol):
        ax[i // ncol, i % ncol].imshow(X[skip + i], cmap='binary')
        ax[i // ncol, i % ncol].set_xticks([])
        ax[i // ncol, i % ncol].set_yticks([])
        if preds is not None:
            ax[i // ncol, i % ncol].text(0.85, 0.1, str(preds[skip + i]), transform=ax[i // ncol, i % ncol].transAxes,
                                         color='green' if y[skip + i] == preds[skip + i] else 'red', weight='bold')
            ax[i // ncol, i % ncol].text(0.05, 0.1, str(y[skip + i]), color='blue',
                                         transform=ax[i // ncol, i % ncol].transAxes, weight='bold')
        else:
            ax[i // ncol, i % ncol].text(0.05, 0.1, str(y[skip + i]), color='blue',
                                         transform=ax[i // ncol, i % ncol].transAxes, weight='bold')
    plt.savefig('output/' + file_name + '.png')
    plt.show()


plot_multiple_imgs(X_train, y_train, 10, 10, (12, 8), file_name="training_predications")


# set the callbacks for the compilation of the model
def set_callbacks(description='run1', es_patience=10, rlop_patience=7):
    cp = ModelCheckpoint('output/best_model_weights_{}.h5'.format(description), save_best_only=True)
    es = EarlyStopping(patience=es_patience, monitor='val_acc')
    rlop = ReduceLROnPlateau(patience=rlop_patience)
    tb = TensorBoard()
    cb = [cp, es, rlop, tb]
    return cb


# Build our model
print("[INFO] building model...")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# compile and fit
print("[INFO] compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print("[INFO] training CNN...")
history = model.fit(np.expand_dims(X_train, axis=-1), to_categorical(y_train), validation_split=0.2, shuffle=True,
                    epochs=10,
                    callbacks=set_callbacks('initial_CNN_model_mnist1'))

# Plot training & validation accuracy values
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title('Model accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Test'], loc='upper left')
plt.savefig('output/acc_and_loss.png')
plt.show()

# Plot confusion matrix
print("[INFO] evaluating CNN...")
preds = model.predict(np.expand_dims(X_test, axis=-1))
pred_cat = np.argmax(preds, axis=1)
print('model accuracy on test set is: {0:.2f}%'.format(accuracy_score(y_test, pred_cat) * 100))
sns.heatmap(confusion_matrix(y_test, pred_cat), cmap='Greens', annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('True label')
plt.title('mnist Convolutional model \n classification results on test set')
plt.savefig('output/confusion_matrix.png')

# plot some predications
plot_multiple_imgs(X_test, y_test, 10, 10, (12, 12), pred_cat, skip=400, file_name="test_predications")

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")

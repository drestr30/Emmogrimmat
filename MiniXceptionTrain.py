from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.utils import to_categorical
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.vis_utils import plot_model
import os
import cv2
import numpy as np
from Filtro import fourier_transform, inverse_fourier_transform, laguerre_gauss_filter
import scipy

def resize_img(img, shape):
    new_img = scipy.misc.imresize(img, shape)
    return new_img

def balance(faces, emotions, num):
    unique, counts = np.unique(emotions, return_counts=True)
    print(dict(zip(unique, counts)))
    emotions_id =emotions # np.argmax(emotions, axis=1)

    index_list = list()
    for i in unique:
        index = np.where(emotions_id == i)[0]
        if np.shape(index)[0] > num:
            if i == 6:  ### special balance for any class
                index_list.append(index[num:])
            else:
                index_list.append(index[num:])

    index_4_delete = np.concatenate(index_list)

    balanced_faces = np.delete(faces,index_4_delete, axis=0)
    balanced_emotions = np.delete(emotions,index_4_delete, axis=0)

    # balanced_emotions = np.where(balanced_emotions == 4, 3, balanced_emotions)
    # balanced_emotions = np.where(balanced_emotions == 6, 4, balanced_emotions)

    unique, counts = np.unique(balanced_emotions, return_counts=True)
    print(dict(zip(unique, counts)))
    return balanced_faces, balanced_emotions

def fix_labels(emotions):
    new_labels = []
    for emotion in emotions:
        if emotion == 0:
            new_labels.append(emotion+5)
        elif emotion == 5:
            new_labels.append(emotion - 5)
        elif emotion == 1:
            new_labels.append(emotion +1)
        elif emotion == 2:
            new_labels.append(emotion-1)
        else:
            new_labels.append(emotion)
    return new_labels

def filterImages(images):

    # fil_images = np.empty(np.shape(images))

    # for n, image in zip(range(np.shape(images)[0]),images):
    #     fil_images[n,:,:,:] = image
    image = np.squeeze(images, axis=2)
    height, width = np.shape(image)
    kernel = laguerre_gauss_filter(height, width, 0.3)
    lenna_fourier = fourier_transform(image)
    kernel_fourier = fourier_transform(kernel)

    out = np.multiply(kernel_fourier, lenna_fourier)
    out = np.abs(inverse_fourier_transform(out))

    fil_img = out / out.max()  # normalize the data to 0 - 1
    fil_img = 255 * fil_img  # Now scal by 255
    fil_img = fil_img.astype(np.uint8)
        # fil_images[n,:,:,0] = fil_img

    return np.expand_dims(fil_img, axis=2)

def plotImages(batch_data, n_images=(4, 4)):
    fer_labels = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    axes = axes.flatten()
    images = batch_data[0]
    for n, ax in zip(range(16), axes):
        img = images[n, :, :, :]
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(fer_labels[np.argmax(batch_data[1][n])])
    plt.tight_layout()
    plt.show()


# parameters
#
# dataset = "facesDB"
# # #
# data_root = "/home/fourier/Documentos/Datasets_emociones/{}/train".format(dataset)
# val_root = "/home/fourier/Documentos/Datasets_emociones/{}/validation".format(dataset)
# test_root = "/home/fourier/Documentos/Datasets_emociones/{}/test".format(dataset)

dataset_path = '/home/fourier/Documentos/Datasets_emociones/'

batch_size = 32
num_epochs = 15
input_shape = (100, 100, 1)
verbose = 1
num_classes = 7
patience = 5
base_path = 'models/'
l2_regularization = 0.01

image_size = (100,100)

def load_fer2013():

    data = pd.read_csv(dataset_path + "fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    emo_inds = np.argmax(emotions, axis=1)
    faces = []
    emo_coded = []
    for pixel_sequence, emotion in zip(pixels, emo_inds):
        # if emotion == 3 or emotion == 5 :
        #     continue
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
        emo_coded.append(emotion)
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    return faces,np.asarray(emo_coded)

def load_raf(gray=False, test=False):
    raf_path = "RAFalinied/train/"
    data = sorted(os.listdir(dataset_path + raf_path))
    # width, height = 100, 100

    faces = []
    emo_coded = []
    with open(dataset_path + 'RAFalinied/list_patition_label.txt') as label_data:
        for image_path, label_line in zip(data, label_data):
            face = cv2.imread(dataset_path + raf_path + image_path)
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if gray:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, image_size)
                face = np.expand_dims(face, axis=-1)
            faces.append(face)
            emotion = label_line[-2]
            emotion = int(emotion)-1   ### get emotions from 0 to 6
            emo_coded.append(emotion)
        faces = np.asarray(faces)

    if test:
        raf_test_path = "RAFalinied/test/"
        data = sorted(os.listdir(dataset_path + raf_test_path))
        # width, height = 100, 100

        test_faces = []
        test_emo_coded = []
        with open(dataset_path + 'RAFalinied/test_patition_label.txt') as label_data:
            for image_path, label_line in zip(data, label_data):
                face = cv2.imread(dataset_path + raf_test_path + image_path)
                if gray:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, image_size)
                    face = np.expand_dims(face, axis=-1)
                test_faces.append(face)
                emotion = label_line[-2]
                emotion = int(emotion) - 1  ### get emotions from 0 to 6
                test_emo_coded.append(emotion)
            test_faces = np.asarray(test_faces)

        return faces, np.asarray(emo_coded), test_faces, np.asarray(test_emo_coded)

    return faces,np.asarray(emo_coded)

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    # x = filterImages(x)
    x = x/ 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

faces, emotions = load_fer2013()
# plt.imshow(faces[0,:,:,:])
# plt.show()
# faces, emotions = balance(un_faces, un_emotions, 5000)
#
# faces, emotions, xtest, ytest = load_raf(gray=True, test= True)

# emotions = fix_labels(emotions)
# ytest = fix_labels(ytest)

# values, counts = np.unique(emotions, return_counts=True)
# for v,c in zip(values,counts):
#     print('value: %s, counts: %s'%(v,c))

faces, emotions = balance(faces, emotions, 1000)
# xtest, ytest = balance(xtest, ytest, 100)

# filtered_image = filterImages(faces)
# procesed_faces = preprocess_input(faces)

xtrain, xval, ytrain, yval = train_test_split(faces, to_categorical(emotions),test_size=0.1,shuffle=True)
# xtrain, xtest, ytrain, ytest = train_test_split(_xtrain, _ytrain, test_size=0.1, shuffle= True)

# print(len(xtrain), len(xval), len(xtest))

# data generators
data_generator = ImageDataGenerator(preprocessing_function= preprocess_input,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    brightness_range= (0.5,1.5),
                                    shear_range=0.1)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# #
# image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size,
#                                                 color_mode="grayscale")
# val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size,
#                                               color_mode="grayscale")
# test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size,
#                                                color_mode="grayscale")

image_data = data_generator.flow(xtrain,ytrain,batch_size)
val_data = test_generator.flow(xval,yval, batch_size)

## Check if batch info is OK.

one_batch = next(iter(image_data))
plotImages(one_batch)

# val_batch = next(iter(val_data))
# plotImages(val_batch)

# xtest = preprocess_input(xtest)
# plt.imshow(np.squeeze(xtest[0], axis=2), cmap='gray', vmin=-1, vmax=1)
plt.show()

# model parameters
regularization = l2(l2_regularization)

# model arquitecture
 #base
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# module 1
residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 2
residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 3
residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 4
residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name='predictions')(x)

model = Model(img_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

## Reinstantiete previous trained model
#
# retrain_model = 'mXC.ferBalance.19-0.62'
# model = load_model('./Mx_models/{}.hdf5'.format(retrain_model))
# score = model.evaluate_generator(val_data, steps=5)
# print('Reinstatiete model loss:', score)
#
# print(model.summary())

# callbacks

#### Logging per batch

class Batch_logger(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracy.append(logs.get('val_acc'))

logging = Batch_logger()

model_name = 'mXC'
data_base_name = 'raf'
parameter = 'x100_1kBal'

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = '%s%s.%s_%s' %(base_path,model_name, data_base_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr, logging]

plot_model(model,'./{}_model.png'.format(trained_models_path), show_shapes=True)

model.fit_generator(generator=image_data,
                    steps_per_epoch= len(xtrain) /batch_size,# image_data.samples/batch_size
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps= len(xval)/ batch_size #val_data.samples/val_data.batch_size
                    )

## ploting trainig stadistics

# summarize history for loss per epoch
plt.figure()
plt.plot(logging.val_losses)
plt.plot(logging.losses)
plt.title('%s %s model losses' %(model_name, data_base_name))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation', 'train'], loc='upper left')
plt.savefig('./{}_loss.jpg'.format(trained_models_path))
#plt.show()

# summarize history for acc per epoc
plt.figure()
plt.plot(logging.val_accuracy)
plt.plot(logging.accuracies)
plt.title('%s %s model accuracy' %(model_name, data_base_name))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation', 'train'], loc='upper left')
plt.savefig('./{}_acc.jpg'.format(trained_models_path))
#plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Confution Matrix and Classification Report

predicted_classes = model.predict(xtest)
#
# test_steps = test_data.samples/test_data.batch_size
# predicted_classes = model.predict_generator(test_data,steps=test_steps)

predicted_labels = np.argmax(predicted_classes, axis=1)

# 2.Get ground-truth classes and class-labels

true_classes = ytest
# true_classes = test_data.classes
# class_labels = list(test_data.class_indices.keys())
class_labels = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]      ### FER labels
# class_labels = ["surprise","fear","disgust", "happiness","sadness", "anger","neutral"]    ### RAF lables
# 3. Use scikit-learn to get statistics

report = classification_report(true_classes, predicted_labels, target_names=class_labels)
print('Classification Report')
print(report)
print('Confusion matrix')
cnf_matrix = confusion_matrix(true_classes, predicted_labels)
np.set_printoptions(precision=2)
print(cnf_matrix)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels, normalize=True,
                      title='%s %s model confusion matrix'%(model_name, data_base_name))
plt.savefig('./{}_cm.jpg'.format(trained_models_path))

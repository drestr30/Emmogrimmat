import matplotlib.pyplot as plt
import scipy
import numpy as np
from Filtro import laguerre_gauss_filter, fourier_transform, inverse_fourier_transform
import itertools
import os
import pandas as pd
import cv2
from keras.callbacks import Callback

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


def resize_img(img, shape):
    new_img = scipy.misc.imresize(img, shape)
    return new_img

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

def plotImages(batch_data, n_images=(4, 4), gray=True ):
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    facesdb_labels=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    axes = axes.flatten()
    images = batch_data[0]#.astype(np.uint8)
    for n, ax in zip(range(n_images[0]*n_images[1]), axes):
        img = images[n, :, :, :]
        if gray:
            ax.imshow(np.squeeze(img, axis=2), cmap='gray', vmin=-1, vmax=1)
        else: ax.imshow(img.astype(np.uint8), vmin=-3, vmax=3)#cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(facesdb_labels[np.argmax(batch_data[1][n])])
    plt.tight_layout()
    plt.show()

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

class Data_sets():

    def __init__(self):
        self.dataset_path = '/home/fourier/Documentos/Datasets_emociones/'

    def load_fer2013(self, image_size= (48,48)):

        data = pd.read_csv(self.dataset_path + "fer2013/fer2013.csv")
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

    def load_raf(self, image_size=(160,160), gray=False, test=False):
        raf_path = "RAFalinied/train/"
        data = sorted(os.listdir(self.dataset_path + raf_path))
        # width, height = 100, 100

        faces = []
        emo_coded = []
        with open(self.dataset_path + 'RAFalinied/list_patition_label.txt') as label_data:
            for image_path, label_line in zip(data, label_data):
                face = cv2.imread(self.dataset_path + raf_path + image_path)
                face = cv2.resize(face, image_size)
                if gray:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = np.expand_dims(face, axis=-1)
                faces.append(face)
                emotion = label_line[-2]
                emotion = int(emotion)-1   ### get emotions from 0 to 6
                emo_coded.append(emotion)
            faces = np.asarray(faces)

        if test:
            raf_test_path = "RAFalinied/test/"
            data = sorted(os.listdir(self.dataset_path + raf_test_path))
            # width, height = 100, 100

            test_faces = []
            test_emo_coded = []
            with open(self.dataset_path + 'RAFalinied/test_patition_label.txt') as label_data:
                for image_path, label_line in zip(data, label_data):
                    face = cv2.imread(self.dataset_path + raf_test_path + image_path)
                    face = cv2.resize(face, image_size)
                    if gray:
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face = np.expand_dims(face, axis=-1)
                    test_faces.append(face)
                    emotion = label_line[-2]
                    emotion = int(emotion) - 1  ### get emotions from 0 to 6
                    test_emo_coded.append(emotion)
                test_faces = np.asarray(test_faces)

            return faces, np.asarray(emo_coded), test_faces, np.asarray(test_emo_coded)

        return faces,np.asarray(emo_coded)
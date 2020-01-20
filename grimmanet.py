from keras.callbacks import  ModelCheckpoint, EarlyStopping
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
from keras.layers import MaxPooling2D, concatenate
from keras.layers import SeparableConv2D
from keras import layers
from keras.utils import to_categorical
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from Filtro import fourier_transform, inverse_fourier_transform, laguerre_gauss_filter
import scipy
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from keras.utils.vis_utils import plot_model
import os
import cv2


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

def plotImages(batch_data, n_images=(4, 4)):
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    axes = axes.flatten()
    images = batch_data[0]
    for n, ax in zip(range(16), axes):
        img = images[n, :, :, :]
        # ax.imshow(np.squeeze(img, axis=2), cmap='gray', vmin=-1, vmax=1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(batch_data[1][n])
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

dataset_path = "/home/fourier/Documentos/Datasets_emociones/"

def load_raf(gray=False, test=False):
    raf_path = "RAFalinied/train/"
    data = sorted(os.listdir(dataset_path + raf_path))
    # width, height = 100, 100

    faces = []
    emo_coded = []
    with open(dataset_path + 'RAFalinied/list_patition_label.txt') as label_data:
        for image_path, label_line in zip(data, label_data):
            face = cv2.imread(dataset_path + raf_path + image_path)
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
    # x = filterImages(x)
    x = x.astype('float32')
    # x = filterImages(x)
    x = x/ 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
# parameters

#
# dataset = "facesDB"
# # #
# data_root = "/home/fourier/Documentos/Datasets_emociones/{}/train".format(dataset)
# val_root = "/home/fourier/Documentos/Datasets_emociones/{}/validation".format(dataset)
# test_root = "/home/fourier/Documentos/Datasets_emociones/{}/test".format(dataset)
#

batch_size = 32
num_epochs = 5
input_shape = (100, 100, 1)
verbose = 1
num_classes = 7
patience = 3
base_path = 'grimmanet_models/'
l2_regularization = 0.01

image_size = (100, 100)

faces, emotions, xtest, ytest = load_raf(gray=False, test= True)

emotions = fix_labels(emotions)
ytest = fix_labels(ytest)

faces, emotions = balance(faces, emotions, 700)
xtest, ytest = balance(xtest, ytest, 100)

xtrain, xval, ytrain, yval = train_test_split(faces, to_categorical(emotions),test_size=0.1,shuffle=True)

# data generators
data_generator = ImageDataGenerator(preprocessing_function= preprocess_input,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    #brightness_range= (0.5,1.2),
                                    shear_range=0.1)
#
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# # # Flows
# image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size)
#                                                 #color_mode="grayscale")
# val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size)
#                                               # color_mode="grayscale")
# test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size)
#                                                # color_mode="grayscale")

image_data = data_generator.flow(xtrain,ytrain,batch_size)
val_data = test_generator.flow(xval,yval, batch_size)

## Check if batch info is OK.

one_batch = next(iter(image_data))
plotImages(one_batch)

# val_batch = next(iter(val_data))
# plotImages(val_batch)

# model parameters
regularization = l2(l2_regularization)

### model arquitecture
 #base
img_input = Input(input_shape)
#


def incep_tower(layer_in, f_size=(3,3), f_number = 16, r_number = 32, batch_norm =True):
    tower = Conv2D(r_number, (1, 1), padding='same', activation='relu')(layer_in)
    tower = Conv2D(f_number, f_size,  padding='same',activation='relu')(tower)
    if batch_norm:
        tower = BatchNormalization()(tower)
    return tower

def incep_factorized(layer_in, f_size=(3,3), f_number = 16, batch_norm= True):
    # tower = Conv2D(r_number, (1, 1), padding='same', activation='relu')(layer_in)
    tower = Conv2D(f_number, (1, f_size[0]),  padding='same',activation='relu')(layer_in)
    tower = Conv2D(f_number, (f_size[0], 1), padding='same', activation='relu')(tower)
    if batch_norm:
        tower = BatchNormalization()(tower)
    return tower

#
# conv1 = Conv2D(64, (3,3), strides=(2,2), activation='relu')(img_input)
# conv1 = MaxPooling2D((3,3), strides=(2,2))(conv1)
# conv1 = BatchNormalization()(conv1)
#
# tower_1 = Conv2D(64, (1,1), activation='relu')(conv1)
# tower_2 = Conv2D(32, (1,1), activation='relu')(conv1)
# tower_2 = incep_factorized(tower_2,(3,3), 32)
# tower_3 = incep_tower(conv1, (3,3), 16, 32, batch_norm=False)
# tower_3 = incep_factorized(tower_3, (3,3), 16)
# tower_4 = MaxPooling2D((3,3), strides= (1,1),padding='same')(conv1)
# tower_4 = Conv2D(16, (1,1), padding='same', activation='relu')(tower_4)
# incept1 = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
# pool1 = MaxPooling2D((3,3), strides=(2,2))(incept1)
#
# tower2_1 = Conv2D(128, (1,1), activation='relu')(pool1)
#
# tower2_2 = Conv2D(64, (1,1), padding='same', activation='relu')(pool1)
# tower2_2a = Conv2D(32, (1,3), padding='same', activation='relu')(tower2_2)
# tower2_2a = BatchNormalization()(tower2_2a)
# tower2_2b = Conv2D(32, (3,1), padding='same', activation='relu')(tower2_2)
# tower2_2b = BatchNormalization()(tower2_2b)
#
# tower2_3 = Conv2D(64, (1,1), padding='same', activation='relu')(pool1)
# tower2_3 = incep_factorized(tower2_3, (3,3), 32, batch_norm=False)
# tower2_3a = Conv2D(16, (1,3), padding='same', activation='relu')(tower2_3)
# tower2_3a = BatchNormalization()(tower2_3a)
# tower2_3b = Conv2D(16, (3,1), padding='same', activation='relu')(tower2_3)
# tower2_3b = BatchNormalization()(tower2_3b)
#
# tower2_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(pool1)
# tower2_4 = Conv2D(32, (1,1), padding='same', activation='relu')(tower2_4)
#
# incept2 = concatenate([tower2_1, tower2_2a, tower2_2b, tower2_3a,tower2_3b, tower2_4], axis= 3)
# pool2 = MaxPooling2D((3,3), strides=(2,2))(incept2)
#
# # tower3_1 = Conv2D(32, (1,1), activation='relu')(pool2)
# # tower3_2 = incep_tower(pool2, (3,3), 32, 16)
# # tower3_3 = incep_tower(pool2, (5,5), 32, 16)
# # tower3_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(pool2)
# # tower3_4 = Conv2D(32, (1,1), padding='same', activation='relu')(tower3_4)
# # incept3 = concatenate([tower3_1, tower3_2,tower3_3, tower3_4], axis= 3)
# # pool3 = MaxPooling2D((3,3), strides=(2,2))(incept3)
#
# output = Conv2D(num_classes, (1, 1), padding='same')(pool2)
# output = GlobalAveragePooling2D()(output)
# output = Activation('softmax', name='predictions')(output)
#
#
# model = Model(img_input, output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# plot_model(model,show_shapes=True, show_layer_names=True)

# Reinstantiete previous trained model

retrain_model = 'IncepV2Raf_1K_x100p.10-0.55'
model = load_model('./{}/{}.hdf5'.format(base_path, retrain_model))
score = model.evaluate_generator(val_data, steps=5)
print('Reinstatiete model loss:', score)

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

model_name = 'IncepV2'
data_base_name = 'Raf_1K'
parameter = 'reTrain700'

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = '%s%s%s_%s' %(base_path, model_name, data_base_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr, logging]

# Save model summary and model plot.
with open("./{}_summary.txt".format(trained_models_path),"w+") as model_file:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: model_file.write(x + '\n'))
model_file.close()

plot_model(model,'./{}_model.png'.format(trained_models_path))

model.fit_generator(generator=image_data,
                    steps_per_epoch=  len(xtrain)/batch_size, #'#image_data.samples/batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=  len(xval)/batch_size) #val_data.samples/val_data.batch_size)

plt.show()

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

# Confution Matrix and Classification Report
#
# test_steps = test_data.samples/test_data.batch_size
# predicted_classes = model.predict_generator(test_data,steps=test_steps)
# predicted_labels = np.argmax(predicted_classes, axis=1)

# 2.Get ground-truth classes and class-labels

# true_labels = test_data.classes
# class_labels = list(test_data.class_indices.keys())

predicted_classes = model.predict(xtest)
predicted_labels = np.argmax(predicted_classes, axis=1)
true_labels = ytest
class_labels = ["angry","disgust","scaredq", "happy", "sad", "surprised","neutral"]      ### FER labels


# 3. Use scikit-learn to get statistics

report = classification_report(true_labels, predicted_labels, target_names=class_labels)
print('Classification Report')
print(report)
print('Confusion matrix')
cnf_matrix = confusion_matrix(true_labels, predicted_labels)
np.set_printoptions(precision=2)
print(cnf_matrix)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels, normalize=True,
                      title='%s %s model confusion matrix'%(model_name, data_base_name))
plt.savefig('./{}_cm.jpg'.format(trained_models_path))
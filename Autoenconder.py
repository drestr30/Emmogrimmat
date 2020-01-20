from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential, load_model, save_model
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from Filtro import fourier_transform, inverse_fourier_transform, laguerre_gauss_filter
import scipy
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import cv2 as cv
from keras.utils.vis_utils import plot_model

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

def filterImages(image):

    # fil_images = np.empty(np.shape(images))

    # for n, image in zip(range(np.shape(images)[0]),images):
    #     fil_images[n,:,:,:] = image
    image = np.squeeze(image, axis=2)
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

    return fil_img

def plotImages(batch_data,title, n_images=16, save=False, field=False ):
    fig, axes = plt.subplots(int(np.sqrt(n_images)), int(np.sqrt(n_images)), figsize=(9, 9))
    # fig.suptitle(title, y=0.9)
    axes = axes.flatten()
    images = batch_data
    for n, ax in zip(range(n_images), axes):
        if field:
            img = images[0,:,:,n]
            ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        else:
            img = images[n, :, :, :]
            ax.imshow(np.squeeze(img, axis=2), cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks(())
        ax.set_yticks(())
        # ax.set_title(batch_data[1][n])
    plt.tight_layout()
    # plt.show()
    if save: plt.savefig('./{}{}.jpg'.format(base_path, title))

def preprocess_input(x, v2=True):
    x = filterImages(x)
    x = x.astype('float32')

    x =(255- x)/ 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return np.expand_dims(x, axis=2)

# parameters
dataset = "facesDB"

data_root = "/home/fourier/Documentos/Datasets_emociones/{}/train".format(dataset)
val_root = "/home/fourier/Documentos/Datasets_emociones/{}/validation".format(dataset)
test_root = "/home/fourier/Documentos/Datasets_emociones/{}/test".format(dataset)

batch_size = 64
input_shape = (192, 192, 1)
verbose = 1
num_classes = 7
base_path = 'autoencoder/'
l2_regularization = 0.01

image_size =(192,192)# np.squeeze(input_shape, axis=2)

# Data generators
data_generator = ImageDataGenerator(preprocessing_function= preprocess_input,
                                    rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    brightness_range= (0.5,1.5),
                                    shear_range=0.1)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Autoencoder generators
image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size,
                                                color_mode="grayscale", class_mode='input')
val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size,
                                              color_mode="grayscale", class_mode='input')
test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size,
                                               color_mode="grayscale", class_mode='input') #grayscale
# Check if batch info is OK.

one_batch = next(iter(image_data))
plotImages(one_batch[0], 'train_batch_data')
plt.show()

# Model arquitecture functions

def vgg_dim_reduce(layer_in, n_filters):
    layer_in = Conv2D(n_filters, (1, 1))(layer_in)
    # layer_in = BatchNormalization()(layer_in)
    layer_in = Activation('relu')(layer_in)
    return layer_in

def vgg_block(layer_in, n_filters, f_size = (3,3), pooling= True):
    # add convolutional layers
    layer_in = Conv2D(n_filters, f_size, padding='same')(layer_in)
    layer_in = Activation('relu')(layer_in)
    layer_in = BatchNormalization()(layer_in)

    # add max pooling layer
    if pooling:
        layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)

    return layer_in

def vgg_transpose_block(layer_in, n_filters, stride=2 ):
    # for _ in range(n_conv):
    layer_in = Conv2DTranspose(n_filters, (3, 3), strides=stride, padding='same')(layer_in)
    layer_in = BatchNormalization()(layer_in)
    layer_in = Activation('relu')(layer_in)

    return layer_in

# Auto Encoder

img_input = Input(input_shape)
#
encoder = vgg_block(img_input,16, (5,5), pooling=False)
# encoder = vgg_dim_reduce(encoder, 16)
encoder = vgg_block(encoder, 16, (5,5), pooling=True)
# encoder = vgg_dim_reduce(encoder,32)
encoder = vgg_block(encoder,32, (3,3), pooling=False)
encoder = vgg_dim_reduce(encoder, 16)
encoder = vgg_block(encoder, 32, (3,3), pooling=True)
# encoder = vgg_dim_reduce(encoder,32)
# encoder = vgg_block(encoder, 64, (3,3), pooling=True)
# encoder = vgg_dim_reduce(encoder, 32)
# encoder = vgg_block(encoder, 128, (3,3), pooling=True)

encoded = vgg_dim_reduce(encoder, 64)  #### (6,6,128) latent representation.

# decoder = vgg_transpose_block(encoded, 128, stride=2)
# decoder = vgg_dim_reduce(decoder, 32)
# decoder = vgg_transpose_block(decoder, 64, stride= 2)
# decoder = vgg_dim_reduce(decoder, 32)
decoder = vgg_transpose_block(encoded, 32, stride=1)
decoder = vgg_dim_reduce(decoder, 16)
decoder = vgg_transpose_block(decoder, 32, stride= 2)
decoder = vgg_transpose_block(decoder, 16, stride=1)
# decoder = vgg_dim_reduce(decoder, 16)
decoder = vgg_transpose_block(decoder, 16, stride= 2)

decoded_output = Conv2D(1,(3,3), activation='sigmoid', padding='same', name='encoded_img')(decoder)

autoencoder = Model(img_input, decoded_output)
opt = Adam(lr=0.01)
autoencoder.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

autoencoder.summary()

early_stop = EarlyStopping('val_loss', patience=3)
autoencoder.fit_generator(generator=image_data,
                    steps_per_epoch= 300, #image_data.samples/batch_size,
                    epochs=7, verbose=1, callbacks= [early_stop],
                    validation_data= val_data,
                    validation_steps= val_data.samples/val_data.batch_size)


test_steps = test_data.samples / test_data.batch_size
test_batch = next(iter(test_data))
plotImages(test_batch[0], 'test_d02', save=True)
#
pred_batch = autoencoder.predict_on_batch(test_batch[0])
plotImages(pred_batch, 'recostructed_d02', save=True)

##############################
# Reinstantiete previous trained model
#
# retrain_model = 'ancoderFacesFn_deep_01.04-0.13'
# base_model = load_model('./{}/{}.hdf5'.format(base_path,retrain_model))
# base = base_model.layers[-43].output
##############################
# Classifer data generator
image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size,
                                                color_mode="grayscale")
val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size,
                                              color_mode="grayscale")
test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size,
                                           color_mode="grayscale") #grayscale

# Train classifier, freeze latent space.
base_model = encoded
encoder_model = Model(autoencoder.input, encoded)
encoder_model.summary()
save_model(encoder_model, '%sencoder48-d02.hdf5')
plotImages(encoder_model.predict(test_batch[0]), 'Encoded_activation_field',field = True, save=True)

vgg = vgg_block(base_model, 128, True)
vgg = vgg_dim_reduce(vgg, 64)
vgg = vgg_block(vgg, 128, True)
classifier = Conv2D(num_classes, (3,3), padding='same', name='classifier')(vgg)
classifier = GlobalAveragePooling2D()(classifier)
classifier = Activation('softmax', name='predictions')(classifier)
model = Model(autoencoder.input, classifier)

for layer in model.layers[:-13]:
    layer.trainable = False
    print(layer.name, layer.trainable)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks and train settings
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

model_name = 'Ncoder'
data_base_name = 'Fn'
parameter = 'd02'
model_full_name = '%s%s_%s' %(model_name, data_base_name, parameter)
#
with open("./autoencoder/{}.txt".format(model_full_name),"w+") as model_file:
    # Pass the file handle in as a lambda function to make it callable
    model_file.write('Batch size:{}'.format(batch_size))
    autoencoder.summary(print_fn=lambda x: model_file.write(x + '\n'))
    encoder_model.summary(print_fn=lambda x: model_file.write(x + '\n'))
    model.summary(print_fn=lambda x: model_file.write(x + '\n'))
model_file.close()
plot_model(model, '%sautoencoder_classifier.png'%base_path, show_shapes=True, show_layer_names=False)

num_epochs = 10
patience = 5

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)
trained_models_path = '%s%s%s_%s' %(base_path, model_name, data_base_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr, logging]


model.fit_generator(generator=image_data,
                    steps_per_epoch= 300, #image_data.samples/batch_size,
                    epochs=num_epochs, verbose=1,
                    validation_data= val_data, callbacks=callbacks,
                    validation_steps= val_data.samples/val_data.batch_size)

# ploting trainig stadistics
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

#
# image = cv.imread('./sadness.jpg')
# image = cv.resize(image,image_size)
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
# feed = preprocess_input(np.expand_dims(gray, axis=2))
# plt.imshow(np.squeeze(feed, axis=2) ,cmap='gray',  vmin= -1, vmax=1)
# # plt.show()
#
# reconstructed = model.predict(np.expand_dims(feed,axis=0))
# plotImages(np.squeeze(reconstructed ,axis=0), 'feature map', n_images=16, filters=True)
# plt.imshow(np.squeeze(np.squeeze(reconstructed,axis=3),axis=0), cmap='gray')
# plt.savefig('./{}_acc.jpg'.format(trained_models_path))
# plt.show()

### classification report and confusion matrix
#
# confusion matriz
test_steps_per_epoch = test_data.samples / test_data.batch_size
pred_classes = model.predict_generator(test_data, steps= test_steps_per_epoch)
pred_labels = np.argmax(pred_classes, axis=1)

# 2.Get ground-truth classes and class-labels

true_labels = test_data.classes
class_labels = list(test_data.class_indices.keys())
#
# 3. Use scikit-learn to get statistics

report = classification_report(true_labels, pred_labels, target_names=class_labels)
print('Classification Report')
print(report)
print('Confusion matrix')
cnf_matrix = confusion_matrix(true_labels,pred_labels)
np.set_printoptions(precision=2)
print(cnf_matrix)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels, normalize=True,
                      title='{} model normalized confusion matrix'.format(model_name))
plt.savefig('{}.jpg'.format(trained_models_path))


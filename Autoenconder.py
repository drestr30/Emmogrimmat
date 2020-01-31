from keras.callbacks import  ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential, load_model, save_model
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import cv2 as cv
from keras.utils.vis_utils import plot_model
from Utils import plot_confusion_matrix, resize_img, filterImages, plotImages

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
    if save: plt.savefig('./{}/{}.png'.format(base_path, title))

def preprocess_input(x, v2=True):
    x = filterImages(x)
    x = x.astype('float32')

    x =(255- x)/ 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# # Model arquitecture functions

def incep_tower(layer_in, f_size=(3,3), f_number = 16, r_number = 32, batch_norm =True):
    tower = Conv2D(r_number, (1, 1), padding='same', activation='relu')(layer_in)
    tower = Conv2D(f_number, f_size,  padding='same',activation='relu')(tower)
    if batch_norm:
        tower = BatchNormalization()(tower)
    return tower

def vgg_dim_reduce(layer_in, n_filters):
    layer_in = Conv2D(n_filters, (1, 1))(layer_in)
    # layer_in = BatchNormalization()(layer_in)
    layer_in = Activation('relu')(layer_in)
    return layer_in

def incep_factorized(layer_in, f_size=(3,3), f_number = 16, batch_norm= True):
    # tower = Conv2D(r_number, (1, 1), padding='same', activation='relu')(layer_in)
    tower = Conv2D(f_number, (1, f_size[0]),  padding='same',activation='relu')(layer_in)
    tower = Conv2D(f_number, (f_size[0], 1), padding='same', activation='relu')(tower)
    if batch_norm:
        tower = BatchNormalization()(tower)
    return tower

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
#
    return layer_in

# # parameters
dataset = "facesDB"
root = '/home/meyer/Descargas'

data_root = '{}/{}/train'.format(root, dataset)
val_root = '{}/{}/validation'.format(root, dataset)
test_root = '{}/{}/test'.format(root, dataset)
#
batch_size = 64
input_shape = (192, 192, 1)
verbose = 1
num_classes = 7
base_path = './saved_models/autoencoder'
l2_regularization = 0.01

model_name = 'Ncoder'
data_base_name = 'Fn'
parameter = 'test'
model_full_name = '%s%s_%s' %(model_name, data_base_name, parameter)

image_size =(192,192) # np.squeeze(input_shape, axis=2)

# # Data generators
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

# Auto Encoder

img_input = Input(input_shape)
encoder = vgg_block(img_input,8, (5,5), pooling=False)
encoder = vgg_block(encoder, 8, (5,5), pooling=True)
encoder = vgg_block(encoder,16, (3,3), pooling=False)
encoder = vgg_block(encoder, 16, (3,3), pooling=True)

encoded = vgg_dim_reduce(encoder, 32)

decoder = vgg_transpose_block(encoded, 16, stride=1)
decoder = vgg_transpose_block(decoder, 16, stride= 2)
decoder = vgg_transpose_block(decoder, 8, stride=1)
decoder = vgg_transpose_block(decoder, 8, stride= 2)
decoded_output = Conv2D(1,(3,3), activation='sigmoid', padding='same', name='encoded_img')(decoder)

autoencoder = Model(img_input, decoded_output)
opt = Adam(lr=0.01)
autoencoder.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
autoencoder.summary()

early_stop = EarlyStopping('val_loss', patience=3)
autoencoder.fit_generator(generator=image_data,
                    steps_per_epoch= 1,#300, #image_data.samples/batch_size,
                    epochs=1, verbose=1, callbacks= [early_stop],
                    validation_data= val_data,
                    validation_steps=1)# val_data.samples/val_data.batch_size)


test_steps = test_data.samples / test_data.batch_size
test_batch = next(iter(test_data))
plotImages(test_batch[0], 'test_%s'%parameter, save=True)

pred_batch = autoencoder.predict_on_batch(test_batch[0])
plotImages(pred_batch, 'recostructed_%s'%parameter, save=True)

encoder_model = Model(autoencoder.input, encoded)
encoder_model.trainable = False
encoder_model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
encoder_model.summary()
save_model(encoder_model, '%s/Encoder_%s.hdf5'%(base_path,parameter))
plotImages(encoder_model.predict(test_batch[0]), 'Encoded_activation_field_%s'%parameter,field = True, save=True)

#############################
# #Reinstantiete previous trained model
#
#
# retrain_model = 'Encoder_d03'
# encoder_model = load_model('./{}/{}.hdf5'.format(base_path,retrain_model))
# encoder_model.summary()
# base_model = encoder_model.layers[-1].output
##############################

# Classifer data generator
image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size,
                                                color_mode="grayscale")
val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size,
                                              color_mode="grayscale")
test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size,
                                           color_mode="grayscale") #grayscale

# Train classifier, freeze latent space.

base_model = encoder_model.layers[-1].output
tower_1 = Conv2D(32, (1,1), activation='relu')(base_model)
tower_2 = Conv2D(32, (1,1), activation='relu')(base_model)
tower_2 = incep_factorized(tower_2,(3,3), 16)
tower_3 = incep_tower(base_model, (3,3), 8, 16, batch_norm=False)
tower_3 = incep_factorized(tower_3, (3,3), 8)
tower_4 = MaxPooling2D((3,3), strides= (1,1),padding='same')(base_model)
tower_4 = Conv2D(8, (1,1), padding='same', activation='relu')(tower_4)
incept1 = concatenate([tower_1, tower_2, tower_3, tower_4], axis = 3)
pool1 = MaxPooling2D((3,3), strides=(2,2))(incept1)

tower2_1 = Conv2D(32, (1,1), activation='relu')(pool1)
tower2_2 = Conv2D(32, (1,1), activation='relu')(pool1)
tower2_2 = incep_factorized(tower2_2,(3,3), 16)
tower2_3 = incep_tower(pool1, (3,3), 8, 16, batch_norm=False)
tower2_3 = incep_factorized(tower2_3, (3,3), 8)
tower2_4 = MaxPooling2D((3,3), strides= (1,1),padding='same')(pool1)
tower2_4 = Conv2D(8, (1,1), padding='same', activation='relu')(tower2_4)
incept2= concatenate([tower2_1, tower2_2, tower2_3, tower2_4], axis = 3)
pool2 = MaxPooling2D((3,3), strides=(2,2))(incept2)

tower3_1 = Conv2D(64, (1,1), activation='relu')(pool2)
tower3_2 = Conv2D(32, (1,1), padding='same', activation='relu')(pool2)
tower3_2a = Conv2D(16, (1,3), padding='same', activation='relu')(tower3_2)
tower3_2a = BatchNormalization()(tower3_2a)
tower3_2b = Conv2D(16, (3,1), padding='same', activation='relu')(tower3_2)
tower3_2b = BatchNormalization()(tower3_2b)
tower3_3 = Conv2D(32, (1,1), padding='same', activation='relu')(pool2)
tower3_3 = incep_factorized(tower3_3, (3,3), 32, batch_norm=False)
tower3_3a = Conv2D(8, (1,3), padding='same', activation='relu')(tower3_3)
tower3_3a = BatchNormalization()(tower3_3a)
tower3_3b = Conv2D(8, (3,1), padding='same', activation='relu')(tower3_3)
tower3_3b = BatchNormalization()(tower3_3b)
tower3_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(pool2)
tower3_4 = Conv2D(16, (1,1), padding='same', activation='relu')(tower3_4)

incept3 = concatenate([tower3_1, tower3_2a, tower3_2b, tower3_3a,tower3_3b, tower3_4], axis= 3)
pool3 = MaxPooling2D((3,3), strides=(2,2))(incept3)

# vgg = vgg_block(base_model, 128, True)
# vgg = vgg_dim_reduce(vgg, 64)
# vgg = vgg_block(vgg, 128, True)
classifier = Conv2D(num_classes, (5,5), padding='same', name='classifier')(pool3)
classifier = GlobalAveragePooling2D()(classifier)
classifier = Activation('softmax', name='predictions')(classifier)

model = Model(encoder_model.input, classifier)

for layer in model.layers[:17]:
    layer.trainable = False

for layer in model.layers:
    print("{}: {}".format(layer, layer.trainable))

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
#
with open("{}/{}.txt".format(base_path,model_full_name),"w+") as model_file:
    # Pass the file handle in as a lambda function to make it callable
    model_file.write('Batch size:{}'.format(batch_size))
    autoencoder.summary(print_fn=lambda x: model_file.write(x + '\n'))
    encoder_model.summary(print_fn=lambda x: model_file.write(x + '\n'))
    model.summary(print_fn=lambda x: model_file.write(x + '\n'))
model_file.close()
plot_model(model, '{}/encoder_classifier_{}.png'.format(base_path,parameter), show_shapes=True, show_layer_names=False)

num_epochs = 1
patience = 5

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)
trained_models_path = '%s/%s%s_%s' %(base_path, model_name, data_base_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, early_stop, reduce_lr, logging]

model.fit_generator(generator=image_data,
                    steps_per_epoch=1,# 300, #image_data.samples/batch_size,
                    epochs=num_epochs, verbose=1,
                    validation_data= val_data, callbacks=callbacks,
                    validation_steps= 1)#val_data.samples/val_data.batch_size)

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


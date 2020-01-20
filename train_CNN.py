import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, BatchNormalization, Activation, Flatten, MaxPooling2D, AveragePooling1D
from keras.applications import MobileNetV2
from keras.applications import MobileNet
from keras.applications.mobilenetv2 import preprocess_input
# from keras.models import load_model
# from keras.applications import InceptionV3
# from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, SGD
from Filtro import fourier_transform, inverse_fourier_transform, laguerre_gauss_filter

def preprocess_and_filter(image):
    image = filterImages(image)
    return preprocess_input(image)

def plotImages(batch_data, n_images=(4, 4), dim = 0):
    fig, axes = plt.subplots(n_images[0], n_images[1], figsize=(12, 12))
    axes = axes.flatten()
    images = batch_data[0]
    for n, ax in zip(range(16), axes):
        img = images[n, :, :, dim]
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(batch_data[1][n])
    plt.tight_layout()
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gray2rgb(image):
    height, width = np.shape(image)
    rgb = np.zeros((height,width,3))
    rgb[:,:,0] = image
    rgb[:, :, 1] = image
    rgb[:, :, 2] = image
    return rgb

def filterImages(image):

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = rgb2gray(image)
    height, width = np.shape(image)
    kernel = laguerre_gauss_filter(height, width, 0.7)
    lenna_fourier = fourier_transform(image)
    kernel_fourier = fourier_transform(kernel)

    out = np.multiply(kernel_fourier, lenna_fourier)
    out = np.abs(inverse_fourier_transform(out))

    fil_img = out / out.max()  # normalize the data to 0 - 1
    fil_img = 255 - 255 * fil_img  # Now scal by 255
    fil_img = fil_img.astype(np.uint8)
        # fil_images[n,:,:,0] = fil_img

    return gray2rgb(fil_img)

#### Prepare the data
# datasets = {'MUG':"MUG/MUG_faces",
#             'FERG': 'FERG/FERG_faces',
#             'facesDB':'facesDB'}
#
# for dataset_name, dataset_path in datasets.items():

dataset_name = 'facesDB'

data_root = "/home/fourier/Documentos/Datasets_emociones/{}/train".format(dataset_name)
val_root = "/home/fourier/Documentos/Datasets_emociones/{}/validation".format(dataset_name)
# img_path = "/home/fourier/Documentos/Datasets_emociones/filtered_DB/anger/001angertake00030.jpg"

image_generator = ImageDataGenerator(preprocessing_function=preprocess_and_filter, rotation_range=15,
            width_shift_range= .15,
            height_shift_range=.15,
            brightness_range= (1,3),
            shear_range= 15,
            zoom_range= .1 ,
            horizontal_flip= True)

validation_generator = ImageDataGenerator(preprocessing_function=preprocess_and_filter, rotation_range=20,
            width_shift_range= .15,
            height_shift_range=.15,
            brightness_range= (1,3),
            shear_range= 15,
            zoom_range= .2 ,
            horizontal_flip= True)

#aug_test_generator = ImageDataGenerator(rotation_range=15,
 #           width_shift_range= .15,
  #          height_shift_range=.15,
   #         brightness_range= (1,3),
    #        shear_range= 15,
     #       zoom_range= .2 ,
         #   horizontal_flip= True)

batch_size = 16
img_size = (224,224)  ## (299,299) for Inception  (224,224) for MobileNet
input_shape = (224,224,3)
image_data = image_generator.flow_from_directory(str(data_root), target_size=img_size,
                                                 batch_size=batch_size)
val_data = validation_generator.flow_from_directory(str(val_root), target_size=img_size,
                                                    batch_size= batch_size)

### Ploting image data with augmentation
augmented_images = next(iter(image_data))
plotImages(augmented_images)

for image_batch,label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

### Instance the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape= input_shape)  # imports the mobilenet model
#print(base_model.summary())
x = base_model.output  #Discard the last couple of layers
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='relu')(x)
#x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)
classifierLayer = Dense(1, activation='sigmoid')(x)  # final clasification layer
#
model = Model(inputs=base_model.input, outputs=classifierLayer)
opt = SGD(lr=0.001, momentum=0.9)

for layer in model.layers[:-3]:
    layer.trainable = False
    print(layer.name, layer.trainable)

model.compile(
  optimizer= opt, #Adam(),
  loss='binary_crossentropy',
  metrics=['accuracy'])

model_name = 'Mb2c'

### Reinstantiete previous trained model
# model = load_model('./saved_models/{}.h5'.format(model_name))
# score = model.evaluate_generator(val_data, steps=5)
# print('Reinstatiete model loss:', score)

# print(model.summary())

# Set layers to be trainable


print(model.summary())
#
# for i,layer in enumerate(model.layers):
#   print(i,layer.name, 'trainable:',layer.trainable)

# Test run a single batch, to see that the result comes back with the expected shape.
#result = model.predict(image_batch)
#print(result.shape)

###### Train the model


# Use compile to configure the training process:


# lr_finder = LRFinder(model)
# lr_finder.find_generator(image_data, 0.0001, 1, epochs=5)
#
# lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
# lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))
#

base_path = '/home/fourier/Documentos/David Restrepo/Tesis/DeepModels/Mobilnet2c/'
parameter = 'F10f10f'
trained_models_path = '%s%s.%s%s' %(base_path, model_name, dataset_name, parameter)
fig_model_name = '%s%s.%s' %(model_name, dataset_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

epochs = 30
patience = 10
steps_per_epoch =  image_data.samples/image_data.batch_size
val_steps_per_epoch =val_data.samples / val_data.batch_size


#### Logging per batch

class Batch_logger(Callback):
    def on_train_begin(self,logs={}):
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
checkpoint = ModelCheckpoint(model_names, verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience= int(patience/4), factor= 0.2)
callbacks = [logging, checkpoint, early_stop, reduce_lr]

history = model.fit_generator(generator= image_data,
                    epochs=epochs,
                    steps_per_epoch= steps_per_epoch,
                    validation_data=val_data,
                    validation_steps= val_steps_per_epoch,
                    verbose= 1,
                    callbacks=callbacks)

## ploting trainig stadistics

# summarize history for loss per epoch
plt.figure()
plt.plot(logging.val_losses)
plt.plot(logging.losses)
plt.title('{} model losses'.format(model_name,dataset_name,parameter))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation', 'train'], loc='upper left')
plt.savefig('{}_loss.jpg'.format(trained_models_path))
# plt.show()

# summarize history for acc per epoc
plt.figure()
plt.plot(logging.val_accuracy)
plt.plot(logging.accuracies)
plt.title('{} model accuracy'.format(model_name,dataset_name,parameter))
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['validation', 'train'], loc='upper left')
plt.savefig('{}_acc.jpg'.format(trained_models_path))
# plt.show()
# logging.release()

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


#Confution Matrix and Classification Report
test_path = "/home/fourier/Documentos/Datasets_emociones/facesDB/2cl/{}/test".format(dataset_name)
test_data_generator = ImageDataGenerator(preprocessing_function= preprocess_and_filter)
test_data = test_data_generator.flow_from_directory(str(test_path), target_size=img_size,
                                                    batch_size= batch_size, shuffle=False, class_mode='binary')

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
                      title='{} model normalized confusion matrix'.format(model_name,dataset_name,parameter))
plt.savefig('{}.jpg'.format(trained_models_path))
# plt.show()















from keras.models import load_model, Model
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from Utils import fix_labels, Data_sets,Batch_logger, plot_confusion_matrix, plotImages, balance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def preprocess_input(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    return face_pixels

def get_raf_generetors(balance_to = 2000):
    data_sets = Data_sets()
    faces, emotions, xtest, ytest = data_sets.load_raf(image_size=(160, 160), gray=False, test=True)
    emotions = fix_labels(emotions)
    ytest = fix_labels(ytest)

    faces, emotions = balance(faces, emotions, balance_to)
    xtest, ytest = balance(xtest, ytest, 100)

    xtrain, xval, ytrain, yval = train_test_split(faces, to_categorical(emotions), test_size=0.1, shuffle=True)

    # data generators
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True,
                                        brightness_range=(0.5, 1.2),
                                        shear_range=0.1)
    #
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    image_data = data_generator.flow(xtrain, ytrain, batch_size)
    val_data = test_generator.flow(xval, yval, batch_size)

    return image_data, val_data, xtest, ytest

def get_croppedfaces_generators():
    dataset = "facesDB"

    data_root = "/home/fourier/Documentos/Datasets_emociones/{}/train".format(dataset)
    val_root = "/home/fourier/Documentos/Datasets_emociones/{}/validation".format(dataset)
    test_root = "/home/fourier/Documentos/Datasets_emociones/{}/test".format(dataset)

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                        rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True,
                                        brightness_range= (0.5,1.2),
                                        shear_range=0.1)

    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    # # Flows
    image_size = (160, 160)
    image_data = data_generator.flow_from_directory(str(data_root), target_size=image_size, batch_size=batch_size)
                                                    #color_mode="grayscale")
    val_data = data_generator.flow_from_directory(str(val_root), target_size= image_size, batch_size=batch_size)
                                                  # color_mode="grayscale")
    test_data = test_generator.flow_from_directory(str(test_root), target_size=image_size, batch_size=batch_size)
                                                   # color_mode="grayscale")

    return image_data, val_data, test_data

batch_size = 32

image_data, val_data, test_data = get_croppedfaces_generators()

## Check if batch info is OK.

one_batch = next(iter(image_data))
plotImages(one_batch, gray=False)

base_path = './FaceNet'
base_model = load_model('%s/FaceNetModel/facenet_keras.h5'%base_path)
# summarize input and output shape
# print(model.inputs)
# print(model.outputs)
# model.summary()
# plot_model(model,'%s/FaceNetModel/facenet.png'%base_path, show_shapes=True)

x = base_model.layers[-4].output #Discard the last couple of layers
x = Dropout(0.6)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False)(x)
classifierLayer = Dense(7, activation='sigmoid')(x)  # final clasification layer
#
model = Model(inputs=base_model.input, outputs=classifierLayer)
opt = SGD(lr=0.001, momentum=0.9)

for layer in base_model.layers: # model.layers[:-20]:
    layer.trainable = False
    print(layer.name, layer.trainable)

model.compile(optimizer= opt, #Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model_name = 'FacenetEmo'
data_base_name = 'facesDB_1K'
parameter = 'Fully128FT'

logging = Batch_logger()

num_epochs = 10
patience = 5

reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = '%s/%s%s_%s' %(base_path, model_name, data_base_name, parameter)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, reduce_lr, logging]

# Save model summary and model plot.
with open("./{}_summary.txt".format(trained_models_path),"w+") as model_file:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: model_file.write(x + '\n'))
model_file.close()

plot_model(model,'./{}_model.png'.format(trained_models_path))

model.fit_generator(generator=image_data,
                    steps_per_epoch=  image_data.samples/batch_size, #len(xtrain)/batch_size, #
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=  val_data.samples/val_data.batch_size) #len(xval)/batch_size) #

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
test_steps = test_data.samples/test_data.batch_size
predicted_classes = model.predict_generator(test_data,steps=test_steps)
predicted_labels = np.argmax(predicted_classes, axis=1)

# 2.Get ground-truth classes and class-labels

true_labels = test_data.classes
class_labels = list(test_data.class_indices.keys())

# predicted_classes = model.predict(xtest)
# predicted_labels = np.argmax(predicted_classes, axis=1)
# true_labels = ytest
# class_labels = ["angry","disgust","scaredq", "happy", "sad", "surprised","neutral"]      ### FER labels


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
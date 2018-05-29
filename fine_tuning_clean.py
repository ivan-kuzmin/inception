# Libraries
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras import backend as K
import tensorflow as tf

import os
path = "."
os.chdir(path)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#import seaborn

# Dimensions of our images.
img_width, img_height = 150, 150

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#sess = tf.Session(config=config)
gpu_id = 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads= 1
sess = tf.Session(config=config)

# Parameters
train_data_dir = 'clean_karies/train'
validation_data_dir = 'clean_karies/test'
nb_train_samples = 30000
nb_validation_samples = 3000
epochs = 30
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_tensor = Input(shape=(3, img_width, img_height))
    #input_shape = (3, img_width, img_height)
else:
    input_tensor = Input(shape=(img_width, img_height, 3))
    #input_shape = (img_width, img_height, 3)

# Create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
x = Dense(64, activation='selu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data preparation

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range = 30)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# train the model on the new data for a few epochs

history_pre = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    #epochs = 2,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#model.save_weights('first_try.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layer
history_post = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    #epochs = 3,
    epochs=3*epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


plt.figure(1)
plt.plot(history_pre.history['acc'])
plt.plot(history_pre.history['val_acc'])
plt.title('Pre_training; nadam; bs = 32')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('ft_pre_clean.pdf')
plt.clf()
# summarize history for loss
plt.figure(2)
plt.plot(history_post.history['acc'])
plt.plot(history_post.history['val_acc'])
plt.title('Fine_tuning; nadam; bs = 32')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('ft_post_clean.pdf')
plt.clf()

model.save_weights('clean_karies_optimal_weights')

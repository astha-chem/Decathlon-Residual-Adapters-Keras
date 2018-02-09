import glob
import numpy as np
import sklearn.metrics as metrics
import os
import math
#from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils import plot_model
#import pydotplus as pydot
from keras import backend as K
import importlib
import resnet28_RAM as resnet_RAM
importlib.reload(resnet_RAM)


decathlon_data_folder = "/home/paperspace/nbs/data/"
task = 'cifar100'
modeltype = "RAM_scratch_1"
classes = 100
decay = 0.0005
weights = "weights_" + task + "_" + modeltype + ".h5"
traininglog = "training_" + task + "_" + modeltype + ".log"

batch_size = 64
nb_epoch = 15
img_rows, img_cols = 64,64


#cwd = os.getcwd()

data_folder = decathlon_data_folder + "/" + task + "/"

train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
valid_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
train_generator = train_datagen.flow_from_directory(data_folder + "train", target_size=(img_rows, img_cols),batch_size=batch_size)
valid_generator = valid_datagen.flow_from_directory(data_folder + "val", target_size=(img_rows, img_cols),batch_size=batch_size)

train_files = glob.glob(data_folder + "/train/*/*")
train_steps = len(train_files)//batch_size
val_files = glob.glob(data_folder + "/val/*/*")
val_steps = len(val_files)//batch_size
print("TVsteps:", train_steps, val_steps)

init_shape = (3,img_rows, img_cols ) if K.image_dim_ordering() == 'th' else (img_rows, img_cols ,3)
model = resnet_RAM.create_resnet_RAM(init_shape, filters=32, factor=1, nb_classes=classes, N=4, verbose=1, learnall = True, name = task)
model.summary()
#plot_model(model, to_file = "ResNet28_RAM.png")




lrs = [0.1, 0.01, 0.001]
load = False
for lr in lrs:
    sgd_opt = optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=sgd_opt, metrics=["acc"])
    print("Finished compiling")
    print("Allocating GPU memory")
    if load:     
        model.load_weights(weights)
        print("Model loaded.")
    csv_logger = callbacks.CSVLogger(traininglog, separator = ',', append = True)
    early_stopper = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.005, patience=2, verbose=1, mode='auto')
    model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_steps=val_steps, validation_data=valid_generator, epochs = nb_epoch, callbacks = [early_stopper, csv_logger, callbacks.ModelCheckpoint(weights, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)])       
    load = True

#model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, nb_epoch=nb_epoch,callbacks = [callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)],validation_data=(testX, testY),validation_steps=testX.shape[0] // batch_size,)
#model.fit_generator(train_generator, steps_per_epoch= len(train_generator), validation_data=valid_generator, validation_steps = len(valid_generator), epochs = 5, callbacks = [callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)])



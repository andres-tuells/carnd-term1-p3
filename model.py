import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, ELU
from keras.regularizers import l2
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers.noise import GaussianNoise

from random import random, choice

#from keras import backend as K
#K.set_image_dim_ordering('th')

#correction_factor = 0.04
correction_factor = 0.3

def create_model():
    ch, row, col = 3, 160, 320  # camera format

    # model = Sequential()

    # # Normalize
    # model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # # noise input
    # # percent_noise = 0.1
    # # noise = percent_noise
    # # model.add(GaussianNoise(noise))


    # # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    # model.add(Conv2D(24, (5, 5), strides=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    # model.add(ELU())
    # model.add(Conv2D(36, (5, 5), strides=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    # model.add(ELU())
    # model.add(Conv2D(48, (5, 5), strides=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    # model.add(ELU())

    # #model.add(Dropout(0.50))
    
    # # Add two 3x3 convolution layers (output depth 64, and 64)
    # model.add(Conv2D(64, (3, 3), border_mode='valid', W_regularizer=l2(0.001)))
    # model.add(ELU())
    # model.add(Conv2D(64, (3, 3), border_mode='valid', W_regularizer=l2(0.001)))
    # model.add(ELU())

    # # Add a flatten layer
    # model.add(Flatten())

    # model.add(Dense(1164, W_regularizer=l2(0.001)))
    # model.add(ELU())
    # # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    # model.add(Dense(100, W_regularizer=l2(0.001)))
    # model.add(ELU())
    # #model.add(Dropout(0.50))
    # model.add(Dense(50, W_regularizer=l2(0.001)))
    # model.add(ELU())
    # #model.add(Dropout(0.50))
    # model.add(Dense(10, W_regularizer=l2(0.001)))
    # model.add(ELU())
    # #model.add(Dropout(0.50))

    # # Add a fully connected output layer
    # model.add(Dense(1))
    # model.compile(optimizer="adam", loss="mse")

    # return model

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(ELU())
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(128))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def load_samples():
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)[1:]

def change_bright(img):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute a random brightness value and apply to the image
    brightness = .25 + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness
    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

def generator(samples, batch_size=8):
    num_samples = len(samples)
    images = []
    angles = []
    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for sample in samples:
            i = choice([0,0,1,2])
            name = './data/IMG/'+sample[i].split('/')[-1]
            image = cv2.imread(name)
            angle = float(sample[3])
            image = change_bright(image)
            if i==1:angle += correction_factor
            if i==2:angle -= correction_factor
            if random()>0.5:
                images.append(image)
                angles.append(angle)
            else:
                images.append(np.fliplr(image))
                angles.append(-angle)
            if len(images)>=batch_size:
                X_train = np.array(images)
                y_train = np.array(angles)
                images=[]
                angles=[]
                yield shuffle(X_train, y_train)


def plot_history_object( history_object ):
    ### print the keys contained in the history object
    print(history_object.history.keys())
    for i, loss, val_loss in zip(range(1,1+len(history_object.history['loss'])),history_object.history['loss'], history_object.history['val_loss']):
        print("epoch", i)
        print("loss", loss)
        print("val_loss", val_loss)

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def clean_samples(samples):
    rs = []
    for sample in samples:
        if float(sample[3])==0. and random()<=0.1:continue
        rs.append(sample)
    return rs
    
def main():
    global correction_factor
    print("Starting training")
    samples = load_samples()
    #samples = clean_samples(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.1)
    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    # 7. Define model architecture
    model = create_model()
    print("Correction", correction_factor)

    # 9. Fit model on training data
    history_object = model.fit_generator(train_generator, 
        verbose=1, 
        validation_steps=len(validation_samples), 
        epochs=1, 
        validation_data=validation_generator, 
        steps_per_epoch=len(train_samples)
    )
 
    model.save('model.h5')
    #plot_history_object(history_object)



if __name__=='__main__':
    main()


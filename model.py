import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, ELU
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


#from keras import backend as K
#K.set_image_dim_ordering('th')

def create_model():
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
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
        return list(reader)

def generator(samples, batch_size=8):
    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0,3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    if i==0:
                        correction=0
                    elif i==1:
                        correction=0.3
                    elif i==2:
                        correction=-0.3
                    angle = float(batch_sample[3])+correction
                    images.append(image)
                    angles.append(angle)
                    images.append(np.fliplr(image))
                    angles.append(-angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
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
    
def main():
    print("Starting training")
    samples = load_samples()

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    # 7. Define model architecture
    model = create_model()
 
    # 9. Fit model on training data
    history_object = model.fit_generator(train_generator, 
        verbose=1, 
        validation_steps=len(validation_samples)*6, 
        epochs=1, 
        validation_data=validation_generator, 
        steps_per_epoch=len(train_samples)*6
    )

 
    model.save('model.h5')
    #plot_history_object(history_object)



if __name__=='__main__':
    main()
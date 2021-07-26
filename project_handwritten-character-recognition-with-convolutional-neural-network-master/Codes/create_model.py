# import required packages
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Define callback function to stop training when model get 99% accuracy for both train and test
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.98 and logs.get('val_accuracy') > 0.95:
            print("\nReached 98% accuracy of training and 95% accuracy for validation so training is stop !")
            self.model.stop_training = True


# Model's layer definition
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 28 X 28 with 1 byte color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 64 neuron hidden layer
    tf.keras.layers.Dense(64, activation="relu"),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Model Summary
print(model.summary())

# Compile the Model
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# declare callback function
callback = myCallback()


# Create Data_generator which get image directly from directory
# and label them according to its distinct directories for
# for both training and test data set

train_dir = './train'
test_dir = './test'

# rescale images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        # all images are of single channel
        color_mode='grayscale',
        # load only 7351 image in one batch
        batch_size=7351,
        # use categorical crossenotropy loss, so need of categorical label
        class_mode='categorical'
    )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        # load only 2450 image in one batch
        batch_size=2450,
        # all images are of single channel
        color_mode='grayscale',
        # use categorical crossenotropy loss, so need of categorical label
        class_mode='categorical'
    )


# Fit the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    validation_steps=38,
    steps_per_epoch=38,
    epochs=25,
    callbacks=callback,
    verbose=1
    )

# Save the model
model.save('model')


# line chart how train and test accuracy changes with epochs
def visualiztion():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'y', label='Validation loss')
    plt.title('Training and validation accuracy - Training and validation loss')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
    plt.savefig('train_and_test_accuracy_loss_VS_Epochs.png')


visualiztion()

#

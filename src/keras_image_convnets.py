########################################################
# module: keras_image_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for keras image classification
#             nets for project 1
########################################################

import tensorflow as tf
from tensorflow import keras

from keras_load_data import (
    num_classes,
    train_X,
    train_Y,
    test_X,
    test_Y,
    valid_X,
    valid_Y,
)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

tf.random.set_seed(1234)

### change as necessary; this is the path where ConvNet models
### are persisted.
MODEL_PATH = (
    "/home/palani/Projects/School/inteligent-systems-cs5600/data/nets_project01/keras"
)
MODEL_NAME = "keras_img_model"

# Define a simple sequential model with Keras
def create_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.3):
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                16, (16, 16), padding="same", activation="relu", input_shape=(64, 64, 3)
            ),
            keras.layers.MaxPooling2D((4, 4)),
            keras.layers.Conv2D(4, (4, 4), padding="same", activation="relu"),
            keras.layers.MaxPooling2D((4, 4)),
            keras.layers.Flatten(),
            keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay),
            ),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2, epochs=5):
    # Create a basic model instance
    model = create_keras_model(learning_rate, weight_decay, dropout)

    # Display the model's architecture
    model.summary()

    checkpoint_filepath = MODEL_PATH + "/" + MODEL_NAME
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    # Fit the model
    model.fit(
        train_X,
        train_Y,
        validation_data=(test_X, test_Y),
        batch_size=64,
        epochs=epochs,
        callbacks=[model_checkpoint_callback],
    )
    model.save(MODEL_PATH + "/" + "keras_img_model" + ".h5")


def load_keras_model():
    km = create_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2)
    km.load_weights(MODEL_PATH + "/" + MODEL_NAME + ".h5")
    return km


def load_keras_model_from_file(path):
    km = create_keras_model(learning_rate=0.01, weight_decay=0.001, dropout=0.2)
    km.load_weights(path)
    return km


def evaluate_keras_model(model):
    train_loss, train_acc = model.evaluate(train_X, train_Y, verbose=0)
    print("Training Accuracy: {:5.2f}%".format(100 * train_acc))

    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    print("Testing Accuracy: {:5.2f}%".format(100 * test_acc))

    valid_loss, valid_acc = model.evaluate(valid_X, valid_Y, verbose=0)
    print("Validation Accuracy: {:5.2f}%".format(100 * valid_acc))

    return [train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc]


### ========================================================================

if __name__ == "__main__":
    train_keras_model(epochs=50)
    evaluate_keras_model(load_keras_model())

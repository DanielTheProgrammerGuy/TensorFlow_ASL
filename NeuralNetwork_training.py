import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


training_data =np.load("training_data.npy")
training_labels =np.load("training_labels.npy")
test_data = np.load("test_data.npy")
test_labels = np.load("test_labels.npy")

data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.3),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomBrightness(0.3, value_range=(0, 1.0)),
    layers.RandomContrast(0.3, value_range=(0, 1.0)),
])

SIZE =128
if input("load old data?(y/n): ") == "y":
    model = load_model("best_model.keras")
else:
    model = Sequential((
        # --- Convolution Block 1 ---
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # --- Convolution Block 2 ---
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # --- Convolution Block 3 ---
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),


        # --- Convolution Block 4 ---
        #Conv2D(256, (3, 3), activation='relu', padding='same'),
        #BatchNormalization(),
        #MaxPooling2D((2, 2)),
        #Dropout(0.4),

        # --- Dense Head ---
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(27, activation='softmax')
    ))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def labels_to_outputs(labels):
    labels[labels == "Blank"] = "["
    indeces = np.vectorize(ord)(labels) - 65
    outputs = np.zeros((labels.shape[0],27))
    outputs[np.arange(labels.shape[0]),indeces] = 1
    return outputs
def outputs_to_labels(outputs):
    index = np.argmax(outputs,axis=1)
    letters = np.vectorize(chr)(65 + index)
    letters[letters == "["] = "Blank"
    return letters



training_outputs = labels_to_outputs(training_labels)
test_outputs = labels_to_outputs(test_labels)

train_ds = tf.data.Dataset.from_tensor_slices((training_data, training_outputs))
val_ds = tf.data.Dataset.from_tensor_slices((test_data, test_outputs))

# Apply augmentation only to training
train_ds = (
    train_ds
    .shuffle(1000)
    .batch(64)
    .map(lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)

model.fit(train_ds, epochs=40,verbose=1,validation_data=val_ds, callbacks=[checkpoint, early_stop])
import tensorflow as tf
import efficientnet.tfkeras
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from efficientnet.tfkeras import EfficientNetB0, EfficientNetB5
import pandas as pd
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def train_model(CONFIG):
    checkpoint_path = CONFIG['checkpoint_path'].format(experiment_i=CONFIG['experiment_i'])
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 1 epoch
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1,
        save_weights_only=True,
        save_best_only=True)

    #early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min',
    #    baseline=None, restore_best_weights=True
    #)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=CONFIG['log_dir'].format(experiment_i=CONFIG['experiment_i']), 
        update_freq='epoch'
    )

    callbacks = [tensorboard_callback, cp_callback]

    image_dir = CONFIG['image_dir']

    base_model=EfficientNetB0(weights='noisy-student', include_top=False)
    base_model.trainable= True

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1000, activation='relu')(x)
    x=Dropout(rate=0.2)(x)
    preds=Dense(200,activation='softmax')(x)

    model=Model(inputs=base_model.input,outputs=preds)

    train_datagen=ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
    val_datagen=ImageDataGenerator()

    train_generator=train_datagen.flow_from_directory(
                                directory=os.path.join(image_dir, 'train'),
                                batch_size=32,
                                target_size=(224,224),
                                seed=3,
                                shuffle=True,
                                class_mode="categorical")
    val_generator=val_datagen.flow_from_directory(
                                directory=os.path.join(image_dir, 'val'),
                                batch_size=32,
                                target_size=(224,224),
                                seed=3,
                                shuffle=False,
                                class_mode="categorical")

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    step_size_train=train_generator.n//train_generator.batch_size
    step_size_val=val_generator.n//val_generator.batch_size

    model.fit(train_generator,
                    validation_data=val_generator,
                    validation_steps=step_size_val,
                    steps_per_epoch=step_size_train, 
                    epochs=CONFIG['epochs'],
                    callbacks=callbacks
                    )

    path_to_save = CONFIG['saved_model'].format(experiment_i=CONFIG['experiment_i'])
    print(f"Saving model to {path_to_save}...")
    model.save(path_to_save)
    print("Done")

    return model
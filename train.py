import numpy as np
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import os
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import RNNModels
import config as conf

# Project Configs
# ----------------- #
model_name = 'RNN_LSTM_0.2Thresh'
out_path = conf.OUT_PATH
data_path = '0.2Thresh_NewFeat.h5'

optimization_epochs = 15
optimization_trials = 12

def tuneModel(
    model, 
    train, 
    test, 
    model_name, 
    opt_epochs = 15, 
    opt_trials = 12, 
    toggleRLROP = False, 
    out_path = '',
    ):

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = conf.EARLY_STOPPING_PATIENCE,
        restore_best_weights = True
    )

    print("[INFO] Instantiating Hyperoptimization via Bayesian Optimization")

    tuner = kt.BayesianOptimization(
        model,
        objective = kt.Objective("val_loss", direction = 'min'),
        max_trials = opt_trials,
        seed = 42,
        directory = out_path,
        project_name = model_name
    )

    print("[INFO] Performing Hyperoptimization...")
    tuner.search(
        train,
        validation_data = test,
        batch_size = conf.BATCHSIZE,
        callbacks = [early_stop],
        epochs = opt_epochs
    )

    tuner.results_summary()

    print("[INFO] Hyperoptimized Model:")
    best_hps = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.summary()

    print("[INFO] Training Hyperoptimized Model")

    model_save_path = f'{out_path}/{model_name}/Trained_Model'
    print(f'Saving Path: {model_save_path}')
    exists = False
    try:
        os.mkdir(model_save_path)
    except FileExistsError:
        # directory already exists
        print('[INFO] Found existing model at the path')
        exists = True
        pass

    logger = tf.keras.callbacks.CSVLogger(
        f'{model_save_path}/Training.log',
        append = True
    )

    saver = tf.keras.callbacks.ModelCheckpoint(
        f'{model_save_path}/optimized_model.h5',
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True
    )

    CBLIST = [early_stop, logger, saver]

    if toggleRLROP:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-8)
        CBLIST.append(reduce_lr)

    if exists:
        if os.path.exists(f"{model_save_path}/optimized_model.h5"):
            # Load weights into model
            print('[INFO] Loading weights into model')
            model = tf.keras.models.load_model(f'{model_save_path}/optimized_model.h5')  

    else:
        model.fit(
            train,
            validation_data = test,
            batch_size = conf.BATCHSIZE,
            epochs = conf.EPOCHS,
            callbacks = CBLIST,
            verbose = 1
        )

    print("[INFO] Training Metrics:")
    
    history = pd.read_csv(f'{model_save_path}/Training.log', sep=r',', engine='python').set_index('epoch')

    def Acc_graph(history):
        fig = plt.figure()
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        return fig

    accuracy_graph = Acc_graph(history = history)

    def loss_graph(history):
        fig = plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        return fig

    loss_graph = loss_graph(history = history)

    accuracy_graph.savefig(f'{model_save_path}/Optimized_Model_Accuracy_Plot.png')
    loss_graph.savefig(f'{model_save_path}/Optimized_Model_Loss_Plot.png')

    print(f"Max Validation Accuracy: {max(history['val_accuracy'])}\nMax Validation Loss: {min(history['val_loss'])}")

    return max(history['val_accuracy']), min(history['val_loss']), history['val_loss'].idxmin() # For the epoch number

with h5py.File(data_path, 'r') as hf:
    X = np.array(hf['Features'])
    y = np.array(hf['Labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle = True)
X = y = []; del X, y

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
).batch(
    conf.BATCHSIZE
)

test = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(
    conf.BATCHSIZE
)

X_train = X_train = y_train = y_test = []; del X_train, X_test, y_train, y_test

print('[INFO] STARTING...')

model = RNNModels(n = 200, feature_shape = 3).build
best = tuneModel(
    model = model,
    train = train,
    test = test,
    opt_epochs = optimization_epochs,
    opt_trials = optimization_trials,
    model_name = model_name,
    out_path = out_path,
    toggleRLROP = True
)
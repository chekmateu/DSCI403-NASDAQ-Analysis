import tensorflow as tf
import keras_tuner as kt

class RNNModels():
    def __init__(self, n, feature_shape = 5):
        super().__init__()
        self.n = n
        self.feature_shape = feature_shape

    def build(self, hp):
        # Building Tunable Model
        print("[INFO] Compiling Model")
        model = tf.keras.Sequential()
        # Piece it up (hyperparameter) called n (optimise with timesteps (ps) of 150, 500, 1000, 10000, 1000000)
        model.add(tf.keras.Input(shape = (self.n, self.feature_shape)))
        '''
        # GRU - 
        gru_blocks = range(hp.Int('GRU_blocks', 1, 5, default = 1))
        node_selection = [10, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
        for i in gru_blocks:
            # Last layer needs to return sequences.
            if i != max(list(gru_blocks)):
                model.add(
                    tf.keras.layers.GRU(
                        name = f'GRU_layer_{str(i)}',
                        units = hp.Choice(f'GRU_layer_{str(i)}_units', node_selection),
                        return_sequences = True
                    )
                )
            else:
                model.add(
                    tf.keras.layers.GRU(
                        name = f'GRU_layer_{str(i)}_units',
                        units = hp.Choice(f'GRU_layer_{str(i)}_units', node_selection),
                        return_sequences = False
                    )
                )
        '''
        #LSTM 
        lstm_blocks = range(hp.Int('LSTM_blocks', 1, 5, default = 1))
        node_selection = [10, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
        for i in lstm_blocks:
            # Last layer needs to return sequences.
            if i != max(list(lstm_blocks)):
                model.add(
                    tf.keras.layers.LSTM(
                        name = f'LSTM_layer_{str(i)}',
                        units = hp.Choice(f'LSTM_layer_{str(i)}_units', node_selection),
                        return_sequences = True
                    )
                )
            else:
                model.add(
                    tf.keras.layers.LSTM(
                        name = f'LSTM_layer_{str(i)}_units',
                        units = hp.Choice(f'LSTM_layer_{str(i)}_units', node_selection),
                        return_sequences = False
                    )
                )  
        
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dense(3, name = 'output_layer', activation = 'softmax'))
        model.compile(
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 1e-7),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy'],
        )
            
        model.summary()
        return model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.layers import Reshape, BatchNormalization

def build_model(input_shape=(40, 862, 1), num_classes=8):
    """
    Builds a CNN + LSTM model for speech emotion recognition using MFCC input.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels)
        num_classes (int): Number of emotion classes to classify

    Returns:
        model (Model): Compiled Keras model
    """

    inputs = Input(shape=input_shape)

    # CNN Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # CNN Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Reshape CNN output for LSTM input
    # From (batch_size, height, width, channels) to (batch_size, time_steps, features)
    x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    # LSTM Layer
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs, outputs)
    return model

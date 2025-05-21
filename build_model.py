def build_model(
        input_dim,
        hidden_units=[64,64,64],
        activation='relu',
        lr = 1e-4,
        use_adjusted_softmax=False,
        use_threshold_activation=False,
        use_normalized_relu=False,
        final_activation='linear',
        dropout_rate=0.1,
        l2_factor=0.001,
        output_width=1,
        model_name='Model'
):
    
    # Keras / TensorFlow
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from keras.models import Model
    from keras.layers import Dense, Dropout, Input, Lambda
    from keras.regularizers import l2
    from keras.optimizers import Adam
    from keras.layers import BatchNormalization
    from tensorflow.keras.layers import Activation
    import functools
    
    # Builds and compiles a model
    inputs = Input(shape=(input_dim,), name="Input_Layer")
    x = inputs
    for i, hu in enumerate(hidden_units):
        x = Dense(hu,kernel_regularizer=l2(l2_factor), name=f"Dense_Layer_{i+1}")(x)
        x = BatchNormalization(name=f"Batch_Norm_Layer_{i+1}")(x)
        x = Activation(activation, name=f"{activation}_Activation_Layer_{i+1}")(x)
        x = Dropout(dropout_rate, name=f"Dropout_Layer_{i+1}")(x)
    
    if use_adjusted_softmax:
        raw_outputs = Dense(output_width, name="Raw_Output_Layer")(x)
        final_outputs = AdjustedSoftmax(threshold=0.1, name='Softmax_Layer_Threshold_of_0.1')(raw_outputs)
    elif use_threshold_activation:
        raw_outputs = Dense(output_width, name="Raw_Output_Laye")(x)
        final_outputs = ThresholdActivation(threshold=0.1, name="Threshold_of_0.1")(raw_outputs)
    elif use_normalized_relu:
        raw_outputs = Dense(output_width,activation='relu', name="Raw_relu_Output_Layer")(x)
        final_outputs = NormalizedReLU(name="Normalizing_Layer")(raw_outputs)
    else:
        final_outputs = Dense(output_width, activation=final_activation,name=f"{final_activation}_Output_Layer")(x)

    # Define and compile the model
    model = Model(inputs, final_outputs, name=model_name)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_absolute_error',
        metrics=['mae']
    )
    return model

from keras.saving import register_keras_serializable
from keras.layers import Layer
import tensorflow as tf
@register_keras_serializable()
class ThresholdActivation(Layer):
    def __init__(self, threshold=0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        x_thresh = tf.where(inputs < self.threshold, 0.0, inputs)
        x_sum = tf.reduce_sum(x_thresh, axis=1, keepdims=True) + 1e-8
        return x_thresh / x_sum

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config

@register_keras_serializable()
class AdjustedSoftmax(Layer):
    # A custom layer that applies a softmax but enforces a minimum threshold then renormalises so the outputs sum to 1.
    def __init__(self, threshold=0.1, **kwargs):
        super(AdjustedSoftmax, self).__init__(**kwargs)
        self.threshold = threshold 

    def call(self, inputs):
        softmax_output = tf.nn.softmax(inputs)
        adjusted_output =  tf.where(softmax_output < self.threshold, 0.0, softmax_output)
        return adjusted_output / tf.reduce_sum(adjusted_output, axis=1, keepdims=True)

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config

@register_keras_serializable()
class NormalizedReLU(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        # Apply standard ReLU
        relu_output = tf.nn.relu(inputs)
        # Sum along the feature axis
        sum_ = tf.reduce_sum(relu_output, axis=1, keepdims=True)
        # Clamp the sum to prevent division by zero
        sum_clamped = tf.maximum(sum_, self.epsilon)
        # Normalize the output so values sum to 1
        return relu_output / sum_clamped

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config
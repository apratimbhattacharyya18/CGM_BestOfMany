from keras.engine import Layer
from keras import backend as K

class Choose(Layer):
    def __init__(self,  **kwargs):
        super(Choose, self).__init__(**kwargs)
        self.supports_masking = True

    
    def call(self, inputs, training=None):
        nx = K.random_normal(K.shape(inputs));
        return K.in_train_phase(inputs,nx)

    def get_config(self):
        config = {}
        base_config = super(Choose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
import tensorflow as tf


# Instance Normalization Layer
class InstanceNormalization(tf.keras.layers.Layer):
    # Initialization of Objects
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(
            **kwargs)  # calling parent's init
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02), trainable=True)
        self.offset = self.add_weight(
            name='offset', shape=input_shape[-1:],
            initializer='zeros', trainable=True)

    def call(self, x):
        # Compute Mean and Variance, Axes=[1,2] ensures Instance Normalization
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        # Add epsilon to the config dictionary for deserialization
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

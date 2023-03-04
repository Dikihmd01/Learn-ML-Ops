
import tensorflow as tf

# Define the variables
LABEL_KEY = "is_sarcastic"
FEATURE_KEY = "headline"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features
    
    Keyword arguments:
    inputs -- map from feature keys to raw features.
    Return: outputs -> map from feature keys to transformed features
    """

    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
    

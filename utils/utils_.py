import tensorflow as tf

def create_embeddings(vocab_size, depth=512):
    """Creates an embedding variable."""
    return tf.get_variable("embedding", shape = [vocab_size, depth])

def random_replace(tensor, new_value, p):
    """
    Randomly replace each element of a tensor with a new_value with a probability p .
    """
    return tf.map_fn(lambda x: tf.cond(tf.random_uniform([1])[0] < p, lambda : new_value, lambda : x), tensor)

def kl_coeff(i):
    # coeff = (tf.tanh((i - 3500)/1000) + 1)/2
    coeff = (tf.tanh((i - 6000)/2000) + 1)/2
    return tf.cast(coeff, tf.float32)
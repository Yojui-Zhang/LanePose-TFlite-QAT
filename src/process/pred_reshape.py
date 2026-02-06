import tensorflow as tf

def bcn_to_bnc(y_bcn: tf.Tensor) -> tf.Tensor:
    """Convert [B,C,N] to [B,N,C]"""
    return tf.transpose(y_bcn, [0,2,1])



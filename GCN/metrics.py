import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):  # mask 为一个索引向量，1表示该位置训练数据带有标签 
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32) 
    mask /= tf.reduce_mean(mask)
    loss *= mask # cdot
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):   
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)) # tf.equal 比较是否相等
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

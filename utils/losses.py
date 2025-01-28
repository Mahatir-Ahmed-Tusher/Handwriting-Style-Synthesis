import tensorflow as tf

def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output)))
    return real_loss + fake_loss

def generator_loss(fake_char_output, fake_cursive_output):
    char_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_char_output, labels=tf.ones_like(fake_char_output)))
    cursive_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_cursive_output, labels=tf.ones_like(fake_cursive_output)))
    return char_loss + cursive_loss
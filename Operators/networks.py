import tensorflow as tf


def downsampling_block(tensor, name, filters, kernel = [5,5]):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv2', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        pool = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        return pool

def upsampling_block(tensor, name, filters, kernel = [5,5]):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=kernel,
                                          padding="same", name='conv1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        upsample = tf.layers.conv2d_transpose(inputs=conv1, filters=filters, kernel_size=[5, 5],
                                           strides= (2,2), padding="same", name='deconv1',
                                           reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        return upsample


class UNet(object):
    def __init__(self, channels_out):
        self.channels = channels_out

    def net(self, input):
        # same shape conv
        pre1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[5, 5],
                                padding="same", name='pre1', reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        # downsampling 1
        down1 = downsampling_block(tensor=pre1, name='down1', filters=32)

        # downsampling 2
        down2 = downsampling_block(tensor=down1, name='down2', filters=64)

        # downsampling 3
        down3 = downsampling_block(tensor=down2, name='down3', filters=64)

        # downsampling 4
        down4 = downsampling_block(tensor=down3, name='down4', filters=64)

        # upsampling 1
        up1 = upsampling_block(tensor=down4, name='up1', filters=64)
        con1 = tf.concat([up1, down3], axis=3)

        # upsampling 2
        up2 = upsampling_block(tensor=con1, name='up2', filters=64)
        con2 = tf.concat([up2, down2], axis=3)

        # upsampling 3
        up3 = upsampling_block(tensor=con2, name='up3', filters=64)
        con3 = tf.concat([up3, down1], axis=3)

        # upsampling 4
        up4 = upsampling_block(tensor=con3, name='up4', filters=32)
        con4 = tf.concat([up4, pre1], axis=3)

        post1 = tf.layers.conv2d(inputs=con4, filters=16, kernel_size=[5, 5],
                                 padding="same", name='post1',
                                 reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        post2 = tf.layers.conv2d(inputs=post1, filters=self.channels, kernel_size=[5, 5],
                                 padding="same", name='post2',
                                 reuse=tf.AUTO_REUSE)

        return post2


class Linear(object):
    def __init__(self, channels_out):
        self.channels = channels_out

    def net(self, inp):
        shape = inp.shape
        flat = tf.layers.flatten(inp)
        res = tf.layers.dense(inputs=flat, units=shape[1]*shape[2]*int(self.channels),
                              use_bias=False, reuse=tf.AUTO_REUSE, name='dense')
        return tf.reshape(res, shape=[-1, shape[1], shape[2], int(self.channels)])



import tensorflow as tf
from Framework import model_correction
from Operators.networks import UNet


class Unregularized(model_correction):
    linear = False
    batch_size = 16
    experiment_name = 'NoRegularization'

    # the computational model
    def get_network(self, channels):
        return UNet(channels_out=channels)

    def __init__(self, path, data_sets):
        super(Unregularized, self).__init__(path, data_sets)

        # placeholders
        self.approximate_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1]],
                                            dtype=tf.float32)
        self.true_y = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1]], dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        # add a channel dimension
        ay = tf.expand_dims(self.approximate_y, axis=3)
        ty = tf.expand_dims(self.true_y, axis=3)

        # the network output
        self.output = self.UNet.net(ay)

        # loss functional
        loss = tf.sqrt(tf.reduce_sum(tf.square(self.output - ty), axis=(1,2,3)))
        self.l2 = tf.reduce_mean(loss)
        tf.summary.scalar('Loss_L2', self.l2)

        # optimization algorithm
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.l2,
                                                                             global_step=self.global_step)

        # graph for computing the Hessian of the network in a given direction
        self.direction = tf.placeholder(shape=[None, self.measurement_size[0], self.measurement_size[1]],
                                            dtype=tf.float32)
        di = tf.expand_dims(self.direction, axis=3)
        scalar_prod = tf.reduce_sum(tf.multiply(self.output, di))
        self.gradients = tf.gradients(scalar_prod, self.approximate_y)[0]

        # some tensorboard logging
        tf.summary.image('TrueData', ty, max_outputs=2)
        tf.summary.image('ApprData', ay, max_outputs=2)
        tf.summary.image('NetworkOutput', self.output, max_outputs=2)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.path + 'Logs/', self.sess.graph)

        # initialize variables
        tf.global_variables_initializer().run()

        # load in existing model
        print(self.path)
        self.load()

    def evaluate(self, y):
        y, change = self.feedable_format(y)
        result = self.sess.run(self.output, feed_dict={self.approximate_y: y})[...,0]
        if change:
            result = result[0,...]
        return result

    def differentiate(self, point, direction):
        location, change = self.feedable_format(point)
        direction, _ = self.feedable_format(direction)
        result = self.sess.run(self.gradients, feed_dict={self.approximate_y: location, self.direction: direction})
        if change:
            result = result[0,...]
        return result

    def train(self, learning_rate):
        appr, true, image = self.data_sets.train.next_batch(self.batch_size)
        self.sess.run(self.optimizer, feed_dict={self.approximate_y: appr, self.true_y: true,
                                                 self.learning_rate: learning_rate})

    def log(self):
        appr, true, image = self.data_sets.test.next_batch(self.batch_size)
        iteration, summary = self.sess.run([self.global_step, self.merged],
                      feed_dict={self.approximate_y: appr, self.true_y: true})
        self.writer.add_summary(summary, iteration)
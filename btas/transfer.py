import numpy as np
import tensorflow as tf


class TransferEncoder(object):
    def __init__(self, n_hidden, n_steps):
        """
        The transfer encoder learns a mapping between source domain and target domain samples.
        """
        self.n_hidden = n_hidden
        self.n_steps = n_steps

    def fit(self, X, Z, batch_size=np.inf):
        """
        X is the list of samples in the source domain and Z is the list of samples in the target domain. X and Z should
         have equal length. Model parameters are determined using gradient descent optimization with n_steps as
         specified in the constructor.
        """
        input_dim = X[0].shape[0]

        x = tf.placeholder('float', [None, input_dim])
        z = tf.placeholder('float', [None, input_dim])
        weights = tf.placeholder('float', [None])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(
            tf.random_uniform([input_dim, self.n_hidden], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim),
                              seed=np.random.randint(0, 1e9)))

        # Initialize b to zero
        b1 = tf.Variable(tf.zeros([self.n_hidden]))

        y = tf.nn.tanh(tf.matmul(x, W) + b1)

        # Use tied weights
        W2 = tf.transpose(W)
        b2 = tf.Variable(tf.zeros([input_dim]))

        z_hat = tf.nn.tanh(tf.matmul(y, W2) + b2)

        error = tf.reduce_sum(tf.square(z - z_hat), 1)
        cost = tf.sqrt(tf.reduce_mean(error))

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        n_samples = len(X)
        idx = np.random.permutation(np.arange(n_samples))
        X = X[idx]
        Z = Z[idx]

        batch_size = min(n_samples, batch_size)

        for epoch in range(self.n_steps):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = X[i * batch_size:(i * batch_size + batch_size)]
                batch_zs = Z[i * batch_size:(i * batch_size + batch_size)]

                _, c = sess.run([train_step, cost], feed_dict={
                    x: batch_xs,
                    z: batch_zs,
                    weights: np.ones(batch_size)
                })
                avg_cost += c / n_samples * batch_size

                # if epoch % 100 == 0:
                #     print('Epoch', epoch, 'cost ', avg_cost)

        self.x = x
        self.z_hat = z_hat
        self.sess = sess
        return

    def transfer(self, X):
        """
        Reconstruct X in the target domain.
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        return self.sess.run(self.z_hat, feed_dict={self.x: X})

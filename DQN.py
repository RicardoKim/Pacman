
import util
import numpy as np
import tensorflow as tf
from ops import linear, conv2d, clipped_error

class DQN:
    def __init__(self, params):
        self.params = params
        self.network_name = 'qnet'

        self.x = tf.placeholder('float', [None, params['width'],params['height'], params['depth']],name=self.network_name + '_x')
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t')
        self.actions = tf.placeholder("float", [None, params['numActions']], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')

    def build_network(self):
        Input = tf.keras.layers.Input(shape = [None, params['width'] * params['height'], params['depth']] )
        layer = tf.keras.layers.Dense(units = 512, activation = 'relu')(Input)
        layer = tf.keras.layers.Dense(units = 256, activation = 'relu')(layer)
        layer = tf.keras.layers.Dense(units = 64, actviation = 'relu')(layer)
        output = tf.keras.layers.Dense(units = params['numActions'], actvation = 'linear' )(layer)
        model = tf.keras.models.Model(Input, output)
        model.compile(loss = 'mse', optimizer = tf.keras.layers.optimizers.Adame(), metrics=["accuracy"])
        return model


        #Q,Cost,Optimizer
        self.discount = tf.constant(self.params['discount'])
        # Bellman equation
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t)))
        # gives Q with action taken into consideration
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1)
        # Loss function
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))

        if self.params['load_file'] is not None:
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False)
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Gradient descent on loss function
        self.rmsprop = tf.train.RMSPropOptimizer(self.params['lr'],epsilon=self.params['rms_eps']).minimize(self.cost,global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.params['load_file'] is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.params['load_file'])

        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r):

        feed_dict = {self.x: bat_n}
        q_t = self.sess.run(self.y,feed_dict=feed_dict)
        q_t = np.amax(q_t, axis=1)
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}
        _,cnt,cost = self.sess.run([self.rmsprop,self.global_step,self.cost],feed_dict=feed_dict)
        return cnt, cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)

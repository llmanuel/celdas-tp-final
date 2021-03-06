import tensorflow as tf

class DQNetwork:
  SCOPE_NAME = 'FlappyBirdDQNetwork'
  STATE_SIZE = [84, 84, 4]
  ACTION_SIZE = 2           # Jump, do nothing
  LEARNING_RATE = 0.000001   # Alpha (aka learning rate)

  def __init__(self, learningRate = LEARNING_RATE):
    self.learningRate = learningRate

    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.variable_scope(self.SCOPE_NAME):
      # placeholders
      # *STATE_SIZE means that we take each elements of STATE_SIZE in tuple hence is like if we wrote
      # [None, 84, 84, 4]
      self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.STATE_SIZE], name="inputs_")
      self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.ACTION_SIZE], name="actions_")
      
      self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")
      
      """
      First convnet:
      CNN
      BatchNormalization
      ELU
      """
      # Input is 84x84x4
      self.conv1 = tf.compat.v1.layers.conv2d(inputs = self.inputs_,
                                    filters = 32,
                                    kernel_size = [8,8],
                                    strides = [4,4],
                                    padding = "VALID",
                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    name = "conv1")
      
      self.conv1_batchnorm = tf.compat.v1.layers.batch_normalization(self.conv1,
                                              training = True,
                                              epsilon = 1e-5,
                                              name = 'batch_norm1')
      
      self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
      ## --> [20, 20, 32]
      
      
      """
      Second convnet:
      CNN
      BatchNormalization
      ELU
      """
      self.conv2 = tf.compat.v1.layers.conv2d(
        inputs = self.conv1_out,
        filters = 64,
        kernel_size = [4,4],
        strides = [2,2],
        padding = "VALID",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        name = "conv2"
      )
  
      self.conv2_batchnorm = tf.compat.v1.layers.batch_normalization(
        self.conv2, training = True, epsilon = 1e-5, name = 'batch_norm2'
      )

      self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
      ## --> [9, 9, 64]
      
      
      """
      Third convnet:
      CNN
      BatchNormalization
      ELU
      """
      self.conv3 = tf.compat.v1.layers.conv2d(
        inputs = self.conv2_out,
        filters = 128,
        kernel_size = [4,4],
        strides = [2,2],
        padding = "VALID",
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        name = "conv3"
      )
  
      self.conv3_batchnorm = tf.compat.v1.layers.batch_normalization(
        self.conv3, training = True, epsilon = 1e-5, name = 'batch_norm3'
      )

      self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
      ## --> [3, 3, 128]
      
      
      self.flatten = tf.compat.v1.layers.flatten(self.conv3_out)
      ## --> [1152]
      
      
      self.fc = tf.compat.v1.layers.dense(
        inputs = self.flatten, units = 512, activation = tf.nn.elu, kernel_initializer=tf.initializers.GlorotUniform(), name="fc1"
      )
      
      
      self.output = tf.compat.v1.layers.dense(
        inputs = self.fc, kernel_initializer=tf.initializers.GlorotUniform(), units = 2, activation=None
      )


      # predicted Q value.
      self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
      
      # The loss is the difference between the predicted Q_values and the Q_target
      # Sum(Qtarget - Q)^2
      self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
      
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)


  def getCurrentLearningRate(self):
    return self.learningRate
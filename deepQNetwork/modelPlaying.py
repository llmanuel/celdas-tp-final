import tensorflow.compat.v1 as tf
import numpy as np
import os
from game.gameWrapper import GameWrapper
from deepQNetwork.frameProcessor import FrameProcessor
from game.actions import Actions

cwd = os.getcwd()
class ModelPlaying:
  def __init__(self, dqNetwork):
    self.frameProcessor = FrameProcessor()
    self.game = GameWrapper()
    self.dqNetwork = dqNetwork

  def start(self):
    with tf.Session() as sess:
      saver = tf.train.Saver()
      # Load the model
      saver.restore(sess, f"{cwd}/models/a/3/model.ckpt")
      self.game.initGame()


      for i in range(100):
        self.game.forwardTillRevive()
        done = False
        frame = self.game.getGameFrame()

        state = self.frameProcessor.stackFrames(frame, True)
        newScore = 0      
        while not self.game.isBirdDead():
          # Take the biggest Q value (= the best action)
          Qs = sess.run(self.dqNetwork.output, feed_dict = {self.dqNetwork.inputs_: state.reshape((1, *state.shape))})
          
          # Take the biggest Q value (= the best action)
          choice = np.argmax(Qs)
          action = [Actions.HOlD_KEY, Actions.RELEASE_KEY][int(choice)]

          newScore = self.game.getScore()
          
          self.game.makeAction(action)
          done = self.game.isBirdDead()
          
          if done:
            break  
              
          else:
            nextFrame = self.game.getGameFrame()
            nextState = self.frameProcessor.stackFrames(nextFrame)
            state = nextState
                
        reward = self.game.getTotalReward()
        print('Total rewards: {:.4f}'.format(reward), 'Score: {:.4f}'.format(newScore))
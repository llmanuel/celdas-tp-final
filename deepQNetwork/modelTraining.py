import tensorflow.compat.v1 as tf
import numpy as np
import random
from deepQNetwork.frameProcessor import FrameProcessor
from game.actions import Actions
from game.gameWrapper import GameWrapper

class ModelTraining:
  TOTAL_EPISODES = 5000
  EXPLORE_START = 1.0
  EXPLORE_STOP = 0.01
  DECAY_RATE = 0.001 
  BATCH_SIZE = 64
  GAMMA = 0.95

  def __init__(self, memory, dqNetwork):
    self.memory = memory
    self.frameProcessor = FrameProcessor()
    self.game = GameWrapper()
    self.dqNetwork = dqNetwork

  def start(self):
    tf.disable_v2_behavior()
    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/home/manuel/Facultad/celdas-tp-final/tensorboard/dqn/1")
    ## Losses
    tf.summary.scalar("Loss", self.dqNetwork.loss)
    writeOp = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      decayStep = 0

      self.game.initGame()

      for episode in range(self.TOTAL_EPISODES):
        step = 0
        episodeRewards = []

        self.game.forwardTillRevive()

        frame = self.game.getGameFrame()
        state = self.frameProcessor.stackFrames(frame, True)

        justRevived = False

        while not justRevived:
          decayStep += 1

          action, exploreProbability = self.predictAction(self.EXPLORE_START, self.EXPLORE_STOP, self.DECAY_RATE, decayStep, state, sess)

          reward, isDead = self.game.makeAction(action)

          episodeRewards.append(reward)

          if isDead:
            nextState = np.zeros(state.shape)
            self.memory.add((state, action, reward, nextState, isDead))

            self.game.forwardTillRevive()
            frame = self.game.getGameFrame()
            state = self.frameProcessor.stackFrames(frame, True)

            justRevived = True

            totalReward = np.sum(episodeRewards)

            print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(totalReward),
                      'Training loss: {:.4f}'.format(loss),
                      'Explore Probability: {:.4f}'.format(exploreProbability))
          else:
            nextFrame = self.game.getGameFrame()
            nextState = self.frameProcessor.stackFrames(nextFrame)
            self.memory.add((state, action, reward, nextState, isDead))
            state = nextState

          ### LEARNING PART            
          # Obtain random mini-batch from memory
          batch = self.memory.sample(self.BATCH_SIZE)
          statesMiniBatch = np.array([each[0] for each in batch], ndmin=3)
          actionsMiniBatch = np.array([each[1] for each in batch])
          rewardsMiniBatch = np.array([each[2] for each in batch]) 
          nextStatesMiniBatch = np.array([each[3] for each in batch], ndmin=3)
          donesMiniBatch = np.array([each[4] for each in batch])

          targetQsBatch = []

          # Get Q values for next_state 
          QsNextState = sess.run(self.dqNetwork.output, feed_dict = {self.dqNetwork.inputs_: nextStatesMiniBatch})
          
          # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
          for i in range(0, len(batch)):
              terminal = donesMiniBatch[i]

              # If we are in a terminal state, only equals reward
              if terminal:
                  targetQsBatch.append(rewardsMiniBatch[i])
                  
              else:
                  target = rewardsMiniBatch[i] + self.GAMMA * np.max(QsNextState[i])
                  targetQsBatch.append(target)
                  

          targetsMiniBatch = np.array([each for each in targetQsBatch])

          loss, _ = sess.run([self.dqNetwork.loss, self.dqNetwork.optimizer],
                              feed_dict={self.dqNetwork.inputs_: statesMiniBatch,
                                          self.dqNetwork.target_Q: targetsMiniBatch,
                                          self.dqNetwork.actions_: actionsMiniBatch})

          # Write TF Summaries
          summary = sess.run(writeOp, feed_dict={self.dqNetwork.inputs_: statesMiniBatch,
                                              self.dqNetwork.target_Q: targetsMiniBatch,
                                              self.dqNetwork.actions_: actionsMiniBatch})
          writer.add_summary(summary, episode)
          writer.flush()

        # Reset while
        justRevived = True

        # Save model every 5 episodes
        if episode % 5 == 0:
          print("Model Saved")
          saver.save(sess, "/home/manuel/Facultad/celdas-tp-final/tensorboard/dqn/1/models/model.ckpt")


  def predictAction(self, exploreStart, exploreStop, decayRate, decayStep, state, sess):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    expExpTradeoff = np.random.rand()

    exploreProbability = exploreStop + (exploreStart - exploreStop) * np.exp(-decayRate * decayStep)
    
    if (exploreProbability > expExpTradeoff):
        # Make a random action (exploration)
        action = random.choice([Actions.HOlD_KEY, Actions.RELEASE_KEY])
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(self.dqNetwork.output, feed_dict = {self.dqNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = [Actions.HOlD_KEY, Actions.RELEASE_KEY][int(choice)]
                
    return action, exploreProbability
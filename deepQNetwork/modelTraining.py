import tensorflow.compat.v1 as tf
import numpy as np
import os
import random
from deepQNetwork.frameProcessor import FrameProcessor
from game.actions import Actions
from game.gameWrapper import GameWrapper

cwd = os.getcwd()

READ_FROM_META = f"{cwd}/models/e/1/model.ckpt.meta"
READ_FROM_MODEL = f"{cwd}/models/e/1/model.ckpt"
SAVE_IN_MODEL = f"{cwd}/models/e/1/model.ckpt"

class ModelTraining:
  TOTAL_EPISODES = 100000
  EXPLORE_START = 1.00
  EXPLORE_STOP = 0.0001
  DECAY_RATE = 0.0001 
  BATCH_SIZE = 64
  GAMMA = 0.95

  def __init__(self, memory, dqNetwork):
    self.memory = memory
    self.frameProcessor = FrameProcessor()
    self.game = GameWrapper()
    self.dqNetwork = dqNetwork
    self.decayStep = 0
    self.bestScore = 0
    self.trainingCycleCounter = 0

  def start(self):
    tf.disable_v2_behavior()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    self.game.initGame()

    writer = tf.summary.FileWriter(f"{cwd}/tensorboard/dqn/e/5")
    tf.summary.scalar("Loss", self.dqNetwork.loss)
    writeOp = tf.summary.merge_all()

    # If this is the first time training you need this line
    # saver = tf.train.Saver()
    # If you have a meta file use this line
    saver = tf.train.import_meta_graph(READ_FROM_META)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      sess.run(tf.global_variables_initializer())
      # If there is a model to restore from use this line
      saver.restore(sess, READ_FROM_MODEL)
      
      for episode in range(self.TOTAL_EPISODES):
        self.game.forwardTillRevive()

        frame = self.game.getGameFrame()
        state = self.frameProcessor.stackFrames(frame, True)

        justRevived = False

        exploringOfEpisode = []

        alreadySaveAt = 0

        while not justRevived:
          self.decayStep += 1

          action, exploreProbability = self.predictAction(self.EXPLORE_START, self.EXPLORE_STOP, self.DECAY_RATE, state, sess)

          newScore = self.game.getScore()

          reward, isDead = self.game.makeAction(action)

          exploringOfEpisode.append(exploreProbability)

          if newScore >= 30  and newScore != alreadySaveAt and newScore % 5 == 0:
            print("Model Saved")
            saver.save(sess, SAVE_IN_MODEL)
            alreadySaveAt = newScore

          if isDead:
            nextState = np.zeros(state.shape)
            self.memory.add((state, action, reward, nextState, isDead))

            self.game.forwardTillRevive()
            frame = self.game.getGameFrame()
            state = self.frameProcessor.stackFrames(frame, True)

            justRevived = True

            if newScore > self.bestScore:
              self.bestScore =  newScore

            print(
              'Episode: {}'.format(episode),
              'Last Score: {:.4f}'.format(newScore),
              'Best Score: {:.4f}'.format(self.bestScore),
              'trainingCycleCounter: {:.4f}'.format(self.trainingCycleCounter),
              'Max explore Probability: {:.4f}'.format(max(exploringOfEpisode)),
              'Memory occupied: {:.4f} %'.format(self.memory.getMemoryOccupied())
            )
          else:
            nextFrame = self.game.getGameFrame()
            nextState = self.frameProcessor.stackFrames(nextFrame)
            self.memory.add((state, action, reward, nextState, isDead))
            state = nextState

          ### LEARNING PART            
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

            if terminal:
              targetQsBatch.append(rewardsMiniBatch[i])
                
            else:
              target = rewardsMiniBatch[i] + self.GAMMA * np.max(QsNextState[i])
              targetQsBatch.append(target)

          targetsMiniBatch = np.array([each for each in targetQsBatch])

          loss, _ = sess.run(
            [self.dqNetwork.loss, self.dqNetwork.optimizer],
            feed_dict = {
              self.dqNetwork.inputs_: statesMiniBatch,
              self.dqNetwork.target_Q: targetsMiniBatch,
              self.dqNetwork.actions_: actionsMiniBatch
            }
          )

          self.trainingCycleCounter += 1
          # # Write TF Summaries
          summary = sess.run(
            writeOp,
            feed_dict = {
              self.dqNetwork.inputs_: statesMiniBatch,
              self.dqNetwork.target_Q: targetsMiniBatch,
              self.dqNetwork.actions_: actionsMiniBatch
            }
          )
          writer.add_summary(summary, episode)
          writer.flush()

        # Reset while
        justRevived = True

        # Save model every 5 training cycles
        if episode % 5 == 0:
          print("Model Saved")
          saver.save(sess, SAVE_IN_MODEL)



  def predictAction(self, exploreStart, exploreStop, decayRate, state, sess):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    expExpTradeoff = np.random.rand()

    Qs = sess.run(self.dqNetwork.output, feed_dict = {self.dqNetwork.inputs_: state.reshape((1, *state.shape))})
    
    actionStrength = abs(np.diff(Qs))

    fastDecay = 1
    if 2 > actionStrength > 1:
      fastDecay = 200 - self.decayStep
      if self.decayStep >= 200:
        fastDecay = 100
    elif actionStrength > 2:
      self.decayStep -= 1
      fastDecay = 500

    exploreProbability = exploreStop + (exploreStart - exploreStop) * np.exp(-decayRate * self.decayStep * fastDecay)
    
    if (exploreProbability > expExpTradeoff):
      action = random.choice([Actions.HOlD_KEY, Actions.RELEASE_KEY])
        
    else:
      choice = np.argmax(Qs)
      action = [Actions.HOlD_KEY, Actions.RELEASE_KEY][int(choice)]
                
    return action, exploreProbability
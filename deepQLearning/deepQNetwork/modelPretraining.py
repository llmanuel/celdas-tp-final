import numpy as np
import random
from deepQNetwork.frameProcessor import FrameProcessor
from game.actions import Actions
from game.gameWrapper import GameWrapper

class ModelPretraining:
  PRETRAINING_LENGTH = 64

  def __init__(self, memory):
    self.memory = memory
    self.frameProcessor = FrameProcessor()
    self.game = GameWrapper()

  def start(self):
    self.game.initGame()

    for i in range(self.PRETRAINING_LENGTH):
      # If it's the first step
      if i == 0:
        # First we need a state
        frame = self.game.getGameFrame()
        state = self.frameProcessor.stackFrames(frame, True)

      # Random action
      action = random.choice([Actions.HOlD_KEY, Actions.RELEASE_KEY])

      reward, isDead = self.game.makeAction(action)

      if isDead:
        nextState = np.zeros(state.shape)
        self.memory.add((state, action, reward, nextState))

        self.game.forwardTillRevive()
        frame = self.game.getGameFrame()
        state = self.frameProcessor.stackFrames(frame, True)
      else:
        nextFrame = self.game.getGameFrame()
        nextState = self.frameProcessor.stackFrames(nextFrame)
        self.memory.add((state, action, reward, nextState))
        state = nextState

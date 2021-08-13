from game.flappybird import FlappyBird
from game.actions import Actions
from game.worldState import WorldState
from game.utilityCalculator import UtilityCalculator

class GameWrapper:

  def __init__(self):
    self.flappybird = FlappyBird()
    self.rewardCounter = 0

  def initGame(self):
    self.flappybird.initGame()
    self.flappybird.eachCycle()

  def getGameFrame(self):
    return self.flappybird.getGameImage()

  def forwardTillRevive(self):
    while self.flappybird.isDead():
      self.rewardCounter = 0
      self.flappybird.eachCycle()

  def isBirdDead(self):
    return self.flappybird.isDead()

  def getTotalReward(self):
    return self.rewardCounter

  def getScore(self):
    return self.flappybird.getScore()

  def makeAction(self, action):
    currentState = WorldState(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead())
    if action == Actions.HOlD_KEY:
      self.flappybird.holdKeyDown()
    else:
      self.flappybird.releaseKey()
    self.flappybird.eachCycle()
    resultState = WorldState(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead())

    reward = UtilityCalculator(currentState, resultState, action).getUtility()

    self.rewardCounter += reward

    return reward, self.flappybird.isDead()
  
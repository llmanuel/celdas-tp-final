from game.flappybird import FlappyBird # Fijate si no es game.flappybird
from game.actions import Actions
from game.worldState import WorldState
from game.utilityCalculator import UtilityCalculator

class GameWrapper:

  def __init__(self):
    self.flappybird = FlappyBird()

  def initGame(self):
    self.flappybird.initGame()

  def getGameFrame(self):
    return self.flappybird.getGameImage()

  def forwardTillRevive(self):
    while self.flappybird.isDead():
      self.flappybird.eachCycle()

  def isBirdDead(self):
    return self.flappybird.isDead()

  def makeAction(self, action):
    # a = input("now what")
    currentState = WorldState(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead())
    if action == Actions.HOlD_KEY:
      self.flappybird.holdKeyDown()
      # print(f"{bcolors.FAIL}Action: Hold key down{bcolors.ENDC}")
    else:
      self.flappybird.releaseKey()
      # print(f"{bcolors.WARNING}Action: Release key{bcolors.ENDC}")

    resultState = WorldState(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead())

    reward = UtilityCalculator(currentState, resultState, action).getUtility()

    return reward, self.flappybird.isDead()
  
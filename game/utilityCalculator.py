import pygame
from game.distance import Zones

class UtilityCalculator:
  def __init__(self, currentWorldState, worldResult, action):
    self.currentWorldState = currentWorldState
    self.worldResult = worldResult
    self.action = action

  def getUtility(self):
    if self.worldResult.hadDied():
      return -10
    if self.worldResult.zone == Zones.FAR:
      if self.isVelocityHelping(self.action):
        return 1
      else: 
        return -1
    elif self.worldResult.zone == Zones.MIDDLE:
      if self.isVelocityHelping(self.action):
        return 2
      else:
        return -1
    elif self.worldResult.zone == Zones.BORDER:
      if self.isVelocityHelping(self.action):
        return 3
      else:
        return -1
    elif Zones().inGapZones(self.worldResult.zone):
      if self.worldResult.zone == Zones.GAP_TOP:
        if self.isVelocityHelping(self.action):
          return 5
        else:
          return -3
      elif self.worldResult.zone == Zones.GAP_DANGER:
        if self.isVelocityHelping(self.action):
          return 7
        else:
          return -3
      else:
        return 1


  def isVelocityHelping(self, action):
    worldResultPositions = self.worldResult.currentPositions
    birdPosition = pygame.Rect(worldResultPositions[2])
    topWallPosition = pygame.Rect([birdPosition.left, *worldResultPositions[1][-3:]])
    bottomWallPosition = pygame.Rect([birdPosition.left, *worldResultPositions[0][-3:]])

    furtherZones = self.worldResult.zone == Zones.FAR or self.worldResult.zone == Zones.MIDDLE or self.worldResult.zone == Zones.BORDER

    def isReducingVelocity(currentVel, resultVel):
      if resultVel <= 0:
        return True
      elif currentVel > resultVel:
        return True
      else:
        return False
        
    if topWallPosition.colliderect(birdPosition):
      if not isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity):
        return False
      elif furtherZones and isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity):
        return True
      elif self.worldResult.velocity < 0 or (self.worldResult.velocity < self.currentWorldState.velocity and self.worldResult.velocity < 5):
        return True
      else:
        return False
    elif bottomWallPosition.colliderect(birdPosition):
      if isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity):
        return False
      if furtherZones and not isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity):
        return True
      elif self.worldResult.velocity > 0 or (self.worldResult.velocity > self.currentWorldState.velocity and self.worldResult.velocity > -5):
        return True
      else:
        return False
    elif Zones().inGapZones(self.worldResult.zone):
      if isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity) and self.worldResult.zone == Zones.GAP_TOP:
        return True
      elif not isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity) and self.worldResult.zone == Zones.GAP_BOTTOM:
        return True
      elif not isReducingVelocity(self.currentWorldState.velocity, self.worldResult.velocity) and self.worldResult.zone == Zones.GAP_DANGER:
        return True
      else:
        return False
    else:
      return False

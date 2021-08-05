import pygame

class Distance:
  SAFE = 'safe'
  CAREFUL = 'careful'
  DANGER = 'danger'

class Zones:
  FAR = 'far'
  MIDDLE = 'middle'
  BORDER = 'border'
  GAP_TOP = 'gapTop'
  GAP_MIDDLE = 'gapMiddle'
  GAP_BOTTOM = 'gapBottom'
  GAP_DANGER = 'gapDanger'

  def __init__(self):
    pass

  def getZoneAccordingToWalls(self, currentPositions):
    birdPosition = pygame.Rect(currentPositions[2])
    topWallPosition = pygame.Rect([birdPosition.left, *currentPositions[1][-3:]])
    bottomWallPosition = pygame.Rect([birdPosition.left, *currentPositions[0][-3:]])
    gapSize = bottomWallPosition.top - topWallPosition.bottom
    distance = 0
    if (bottomWallPosition.top < birdPosition.bottom):
      distance = bottomWallPosition.top - birdPosition.bottom
    elif (bottomWallPosition.top > birdPosition.bottom):
      distance = bottomWallPosition.top - birdPosition.bottom
    else:
      distance = 0

    gapLimitDanger = gapSize * 0.2
    gapLimitBottom = gapSize * 0.3
    gapLimitMiddle = gapSize * 0.5

    if distance >= 40 + gapSize or distance < -40:
      return Zones.FAR
    elif 40 + gapSize >= distance > 20 + gapSize or -40 <= distance < -20 :
      return Zones.MIDDLE
    elif gapSize < distance <= 20 + gapSize or 0 > distance >= -20:
      return Zones.BORDER
    elif gapSize >= distance > gapLimitMiddle:
      return Zones.GAP_TOP
    elif gapLimitMiddle >= distance > gapLimitBottom:
      return Zones.GAP_MIDDLE
    elif gapLimitBottom >= distance > gapLimitDanger:
      return Zones.GAP_BOTTOM
    elif gapLimitDanger >= distance >= 0:
      return Zones.GAP_DANGER

  def inGapZones(self, zone):
    return zone == Zones.GAP_TOP or zone == Zones.GAP_MIDDLE or zone == Zones.GAP_BOTTOM or zone == Zones.GAP_DANGER

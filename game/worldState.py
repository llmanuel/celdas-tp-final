import pygame
from game.distance import Zones, Distance

CRITICAL_DISTANCE = 180
DANGER_DISTANCE = 60

class WorldState:
    def __init__(self, velocity, currentPositions, isDead = None):
        self.farAwayFormWall = self.isFarAwayFormWall(currentPositions)
        self.distanceToGap = self.calculateDistanceToGap(currentPositions)
        self.zone = Zones().getZoneAccordingToWalls(currentPositions)
        self.isDead = isDead
        self.velocity = velocity
        self.currentPositions = currentPositions

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, WorldState):
            return (
                (self.distanceToGap > 0) == (other.distanceToGap > 0) and
                self.isDead == other.isDead and
                self.zone == other.zone and
                self.similarVelocity(other)
            )
        return False

    def similarVelocity(self, other):
        def first(vel):
            return vel >= 6
        def second(vel):
            return 6 > vel >= 2
        def third(vel):
            return 2 > vel >= -2
        def fourth(vel):
            return -2 > vel >= -6
        def fifth(vel):
            return -6 > vel

        return (
            first(self.velocity) == first(other.velocity) or
            second(self.velocity) == second(other.velocity) or 
            third(self.velocity) == third(other.velocity) or 
            fourth(self.velocity) == fourth(other.velocity) or 
            fifth(self.velocity) == fifth(other.velocity)
        )
    
    def isFarAwayFormWall(self, currentPositions):
        leftSideOfWall = int(currentPositions[1][0])
        rightSideOfBird = int(currentPositions[2][0] + currentPositions[2][2])

        if CRITICAL_DISTANCE < leftSideOfWall - rightSideOfBird:
            return Distance.SAFE
        elif CRITICAL_DISTANCE > leftSideOfWall - rightSideOfBird > DANGER_DISTANCE:
            return Distance.CAREFUL
        else:
            return Distance.DANGER

    def calculateDistanceToGap(self, currentPositions):
        birdPosition = pygame.Rect(currentPositions[2])
        bottomWallPosition = pygame.Rect([birdPosition.left, *currentPositions[0][-3:]])
        
        if (bottomWallPosition.top < birdPosition.bottom):
            return bottomWallPosition.top + 15 - birdPosition.bottom
        elif (bottomWallPosition.top > birdPosition.bottom):
            return bottomWallPosition.top + 15 - birdPosition.bottom
        else:
            return 0
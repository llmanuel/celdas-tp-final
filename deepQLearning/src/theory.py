import pygame
from src.worldState import WorldState
from src.utilityCalculator import UtilityCalculator
from src.distance import Distance

CRITICAL_DISTANCE = 180
DANGER_DISTANCE = 60
class Theory:
    def __init__(self, currentPositions, velocity, action, jsonTheory = None):
        if (jsonTheory is None):
            self.currentWorldState = WorldState(
                self.isFarAwayFormWall(currentPositions),
                self.calculateDistanceToGap(currentPositions),
                velocity,
                currentPositions
            )
            self.action = action
            self.expectedResult = None
            self.successCount = 0
            self.useCount = 0
            self.utility = None
        else:
            jsonCurrentWorldState = jsonTheory['currentWorldState']
            jsonExpectedResult = jsonTheory['expectedResult']
            self.currentWorldState = WorldState(
                jsonCurrentWorldState['farAwayFormWall'],
                jsonCurrentWorldState['distanceToGap'],
                jsonCurrentWorldState['velocity'],
                jsonCurrentWorldState['currentPositions'],
                jsonCurrentWorldState['isDead']
            )
            self.action = jsonTheory['action']
            self.expectedResult = WorldState(
                jsonExpectedResult['farAwayFormWall'],
                jsonExpectedResult['distanceToGap'],
                jsonExpectedResult['velocity'],
                jsonExpectedResult['currentPositions'],
                jsonExpectedResult['isDead']
            )
            self.successCount = jsonTheory['successCount']
            self.useCount = jsonTheory['useCount']
            self.utility = jsonTheory['utility']

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Theory):
            return (
                self.currentWorldState == other.currentWorldState and
                self.expectedResult == other.expectedResult and
                self.action == other.action
            )
        return False

    def similar(self, other):
        if isinstance(other, Theory):
            return (
                self.currentWorldState == other.currentWorldState and
                self.action == other.action
            )
        return False

    def getAction(self):
        return self.action

    def setAction(self, action):
        self.action = action

    def setExpectedResult(self, worldResultPositions, velocity, isDead):
        self.expectedResult = WorldState(
            self.isFarAwayFormWall(worldResultPositions),
            self.calculateDistanceToGap(worldResultPositions),
            velocity,
            worldResultPositions,
            isDead
        )
        self.utility = UtilityCalculator(self.currentWorldState, self.expectedResult, self.action).getUtility()
        if self.utility is None:
            n = input("algo raro gato2")
            print(n)

    def verifyResult(self, worldResultPositions, velocity, isDead):
        if self.action == 'holdKey' and (abs(self.calculateDistanceToGap(worldResultPositions)) - abs(self.currentWorldState.distanceToGap) > 15):
            isDead = True
        if not self.expectedResult:
            self.setExpectedResult(worldResultPositions, velocity, isDead)
            return True
        else:
            worldResult = WorldState(
                self.isFarAwayFormWall(worldResultPositions),
                self.calculateDistanceToGap(worldResultPositions),
                velocity,
                worldResultPositions,
                isDead
            )
            newUtility = UtilityCalculator(self.currentWorldState, worldResult, self.action).getUtility()
            # comparar resultados
            self.useCount += 1
            if worldResult == self.expectedResult and newUtility == self.utility:
                self.successCount += 1
                return True
            else:
                return False # crear una nueva teoria si retorna False

    def getTheoryValue(self):
        if not self.useCount or self.useCount < 5:
            return self.utility
        
        return self.utility * (self.successCount / self.useCount)
    
    def calculateDistanceToGap(self, currentPositions):
        birdPosition = pygame.Rect(currentPositions[2])
        bottomWallPosition = pygame.Rect([birdPosition.left, *currentPositions[0][-3:]])
        
        if (bottomWallPosition.top < birdPosition.bottom):
            return bottomWallPosition.top + 15 - birdPosition.bottom
        elif (bottomWallPosition.top > birdPosition.bottom):
            return bottomWallPosition.top + 15 - birdPosition.bottom
        else:
            return 0

    def isFarAwayFormWall(self, currentPositions):
        leftSideOfWall = int(currentPositions[1][0])
        rightSideOfBird = int(currentPositions[2][0] + currentPositions[2][2])

        if CRITICAL_DISTANCE < leftSideOfWall - rightSideOfBird:
            return Distance.SAFE
        elif CRITICAL_DISTANCE > leftSideOfWall - rightSideOfBird > DANGER_DISTANCE:
            return Distance.CAREFUL
        else:
            return Distance.DANGER

    def isComplete(self):
        return self.currentWorldState and self.action and self.expectedResult and self.utility is not None

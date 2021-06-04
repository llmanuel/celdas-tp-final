import json
import numpy as np
from pathlib import Path
from src.theory import Theory

class TheoryDb:
  FILE_PATH = Path(__file__).parent /'theories.json'

  def __init__(self):
    self.jsonTheories = {}
    self.jsonTheories['theories'] = []

  def theoryToJson(self, theory):
    currentWorldState = theory.currentWorldState
    expectedResult = theory.expectedResult
    currentPositionsForCurrentWorld = currentWorldState.currentPositions
    currentPositionsForExpectedResult = expectedResult.currentPositions

    if isinstance(currentPositionsForCurrentWorld, np.ndarray):
      currentPositionsForCurrentWorld = currentPositionsForCurrentWorld.tolist()
    if isinstance(currentPositionsForExpectedResult, np.ndarray):
      currentPositionsForExpectedResult = currentPositionsForExpectedResult.tolist()

    self.jsonTheories['theories'].append({
      'currentWorldState': {
        'zone': currentWorldState.zone,
        'isDead': currentWorldState.isDead,
        'velocity': currentWorldState.velocity,
        'currentPositions': currentPositionsForCurrentWorld,
        'farAwayFormWall': currentWorldState.farAwayFormWall,
        'distanceToGap': currentWorldState.distanceToGap,
      },
      'expectedResult':  {
        'zone': expectedResult.zone,
        'isDead': expectedResult.isDead,
        'velocity': expectedResult.velocity,
        'currentPositions': currentPositionsForExpectedResult,
        'farAwayFormWall': expectedResult.farAwayFormWall,
        'distanceToGap': expectedResult.distanceToGap,
      },
      'action': theory.action,
      'successCount': theory.successCount,
      'useCount': theory.useCount,
      'utility': theory.utility,
    })

  def saveTheories(self, theories):
    for i, theory in enumerate(theories):
      if theory.isComplete():
        self.theoryToJson(theory)

    with open(TheoryDb.FILE_PATH, 'w') as outfile:
      json.dump(self.jsonTheories, outfile, indent = 2)

  def fetchTheories(self):
    savedTheories = []
    with open(TheoryDb.FILE_PATH, 'r') as jsonFile:
      data = json.load(jsonFile)
      for jsonTheory in data['theories']:
        savedTheories.append(Theory(jsonTheory['currentWorldState']['currentPositions'], jsonTheory['currentWorldState']['velocity'], jsonTheory['action'], jsonTheory = jsonTheory))

    return savedTheories

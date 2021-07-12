import random
import math
import numpy as np
from pprint import pprint
from flappybird import FlappyBird
from src.theoryManager import TheoryManager
from src.actions import Actions
from deepQNetwork.frameProcessor import FrameProcessor
from deepQNetwork.model import DQNetwork

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Agent:
    def __init__(self):
        self.flappybird = FlappyBird()
        self.theoryManager = TheoryManager()
        self.frameProcessor = FrameProcessor()
        self.DQNetwork = DQNetwork()
        self.lastNTheories = []

    """
     * Method used to determine the next move to be performed by the agent.
     * now is moving random
     """
    def act(self, theory): 
        self.observeworld()
        a = input("now what")
        if theory.action == Actions.HOlD_KEY:
            self.flappybird.holdKeyDown()
            print(f"{bcolors.FAIL}Action: Hold key down{bcolors.ENDC}")
        else:
            self.flappybird.releaseKey()
            print(f"{bcolors.WARNING}Action: Release key{bcolors.ENDC}")


    def myRandom(self):
        return random.randint(0, 50)

    def observeworld(self):
        # positions = self.flappybird.getWorldPositionObjects()
        # gravity = self.flappybird.getGravity()
        # jumpSpeed = self.flappybird.getJumpSpeed()
        # birdVelocity = self.flappybird.getBirdVelocity()
        frame = self.flappybird.getGameImage()
        print(f"{bcolors.OKGREEN}Frame: {frame}{bcolors.ENDC}")
        processedFrame = self.frameProcessor.preprocessFrame(frame)
        print(f"{bcolors.OKBLUE}Processed Frame: {processedFrame}{bcolors.ENDC}")
        # print("Bottom block: ", positions[0])
        # print("Top block: ", positions[1])
        # print(f"{bcolors.OKGREEN}Bird: {positions[2]}{bcolors.ENDC}")
        # print(f"{bcolors.OKBLUE}Gravity: {gravity}{bcolors.ENDC}")
        # print(f"{bcolors.OKBLUE}Jump Speed: {jumpSpeed}{bcolors.ENDC}")
        # print(f"{bcolors.OKCYAN}Bird Velocity: {birdVelocity}{bcolors.ENDC}")
        # print("Count: ",self.flappybird.counter)
        # print("Dead: ", self.flappybird.dead)

    def printTheory(self, theory):
        print(f"{bcolors.OKGREEN}Action: {theory.action}{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}Utility: {theory.utility}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Is far away from wall: {theory.currentWorldState.farAwayFormWall}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Distance To Gap: {theory.currentWorldState.distanceToGap}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Bird Velocity: {theory.currentWorldState.velocity}{bcolors.ENDC}")
        print("successCount: ",theory.successCount)
        print("useCount: ", theory.useCount)

    def run(self):  
        self.flappybird.initGame()
        starting = True
        lastTheory = None
        turnsDeadCounter = 0
        turns = 0
        self.theoryManager.loadSavedTheories()
        while True:
            self.flappybird.eachCycle()
            if not starting and turnsDeadCounter == 0:
                self.theoryManager.verifyTheory(lastTheory, self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead(), turns)
                if self.flappybird.isDead():
                    turnsDeadCounter = 1
            else:
                starting = False

            if not self.flappybird.isDead():
                theory = self.theoryManager.getTheory(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), turns)
                # self.printTheory(theory)
                self.setLastTheory(theory)
                lastTheory = theory     
                self.act(theory)
                turnsDeadCounter = 0
                turns += 1

            # The next lines are for saving the Theories in theories.json
            # if turns % 200 == 0:
            #     print(f"{bcolors.OKBLUE}Saving theories{bcolors.ENDC}")
            #     self.theoryManager.saveTheories()
            
    def setLastTheory(self, theory):
        self.lastNTheories.append(theory)
        if (len(self.lastNTheories) > 45):
            self.lastNTheories.pop(0)



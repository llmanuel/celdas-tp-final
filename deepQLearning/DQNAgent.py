import random
import math
import numpy as np
from pprint import pprint
from game.flappybird import FlappyBird
from src.theoryManager import TheoryManager
from src.actions import Actions
from deepQNetwork.frameProcessor import FrameProcessor
from deepQNetwork.memory import Memory
from deepQNetwork.modelPretraining import ModelPretraining
from deepQNetwork.modelTraining import ModelTraining

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
  TRAIN = "Train"
  RUN = "Run"

  def __init__(self):
      self.flappybird = FlappyBird()
      self.frameProcessor = FrameProcessor()
      self.memory = Memory()

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

  def run(self):  
      option = input(f"Select between: {self.TRAIN} or {self.RUN}")

      if option == self.TRAIN:
        # Pretrain model to avoid having an empty memory
        ModelPretraining(self.memory).start()

        # Agent train
        ModelTraining(self.memory).start()
      else:
        # Agent ru
        print("Agent play")


      # while True:
      #     self.flappybird.eachCycle()
      #     if not starting and turnsDeadCounter == 0:
      #         self.theoryManager.verifyTheory(lastTheory, self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), self.flappybird.isDead(), turns)
      #         if self.flappybird.isDead():
      #             turnsDeadCounter = 1
      #     else:
      #         starting = False

      #     if not self.flappybird.isDead():
      #         theory = self.theoryManager.getTheory(self.flappybird.getWorldPositionObjects(), self.flappybird.getBirdVelocity(), turns)
      #         # self.printTheory(theory)
      #         self.setLastTheory(theory)
      #         lastTheory = theory     
      #         self.act(theory)
      #         turnsDeadCounter = 0
      #         turns += 1

      #     # The next lines are for saving the Theories in theories.json
      #     # if turns % 200 == 0:
      #     #     print(f"{bcolors.OKBLUE}Saving theories{bcolors.ENDC}")
          #     self.theoryManager.saveTheories()
          
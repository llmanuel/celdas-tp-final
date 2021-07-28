import random
import math
import numpy as np
from pprint import pprint
from game.flappybird import FlappyBird
from src.theoryManager import TheoryManager
from src.actions import Actions
from deepQNetwork.frameProcessor import FrameProcessor
from deepQNetwork.memory import Memory
from deepQNetwork.model import DQNetwork
from deepQNetwork.modelPretraining import ModelPretraining
from deepQNetwork.modelTraining import ModelTraining
from deepQNetwork.modelPlaying import ModelPlaying

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
  TRAIN = "t"
  RUN = "p"

  def __init__(self):
      self.flappybird = FlappyBird()
      self.frameProcessor = FrameProcessor()
      self.memory = Memory()
      self.dqNetwork = DQNetwork()

  # def observeworld(self):
  #     frame = self.flappybird.getGameImage()
  #     print(f"{bcolors.OKGREEN}Frame: {frame}{bcolors.ENDC}")
  #     processedFrame = self.frameProcessor.preprocessFrame(frame)
  #     print(f"{bcolors.OKBLUE}Processed Frame: {processedFrame}{bcolors.ENDC}")

  def run(self):  
      option = input(f"Select between: train {self.TRAIN} or play {self.RUN}")

      if option == self.TRAIN:
        # Pretrain model to avoid having an empty memory
        ModelPretraining(self.memory).start()

        # Agent train
        ModelTraining(self.memory, self.dqNetwork).start()
      else:
        # Agent ru
        ModelPlaying(self.dqNetwork).start()

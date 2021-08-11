from game.flappybird import FlappyBird
from deepQNetwork.frameProcessor import FrameProcessor
from deepQNetwork.memory import Memory
from deepQNetwork.model import DQNetwork
from deepQNetwork.modelPretraining import ModelPretraining
from deepQNetwork.modelTraining import ModelTraining
from deepQNetwork.modelPlaying import ModelPlaying

class Agent:
  TRAIN = "t"
  RUN = "p"

  def __init__(self):
      self.flappybird = FlappyBird()
      self.frameProcessor = FrameProcessor()
      self.memory = Memory()
      self.dqNetwork = DQNetwork()

  def run(self, mode):  
    if mode == self.TRAIN:
      # Pretrain model to avoid having an empty memory
      ModelPretraining(self.memory).start()

      # Agent train
      ModelTraining(self.memory, self.dqNetwork).start()
    else:
      # Agent ru
      ModelPlaying(self.dqNetwork).start()

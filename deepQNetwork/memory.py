from collections import deque
import numpy as np

class Memory():
  MAX_SIZE = 1000000          # Number of experiences the Memory can keep
  
  def __init__(self):
    self.buffer = deque(maxlen = self.MAX_SIZE)
  
  # experience: (state, action, reward, nextState, isDead)
  def add(self, experience):
    self.buffer.append(experience)
  
  def sample(self, batch_size):
    buffer_size = len(self.buffer)
    index = np.random.choice(np.arange(buffer_size),
                            size = batch_size,
                            replace = False)
    
    return [self.buffer[i] for i in index]
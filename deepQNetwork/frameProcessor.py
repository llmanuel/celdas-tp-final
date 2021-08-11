import cv2
import pygame
import numpy as np
from collections import deque

class FrameProcessor:
  STACK_SIZE = 4

  def __init__(self):
    self.stackedFrames = deque(
      [np.zeros((84, 84), dtype = np.int) for i in range(self.STACK_SIZE)],
      maxlen = 4
    )

  def preprocessFrame(self, frame):
    greyFrame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    normalizedFrame = greyFrame/255
    return normalizedFrame

  def stackFrames(self, currentFrame, justRevived = False):
    preprocessedframe = self.preprocessFrame(currentFrame)

    if justRevived:
      # Clear out stackedFrames
      self.stackedFrames = deque(
        [np.zeros((84, 84), dtype = np.int) for i in range(self.STACK_SIZE)],
        maxlen = 4
      )

      self.stackedFrames.append(preprocessedframe)
      self.stackedFrames.append(preprocessedframe)
      self.stackedFrames.append(preprocessedframe)
      self.stackedFrames.append(preprocessedframe)

      # Stack the frames
      stackedState = np.stack(self.stackedFrames, axis = 2)
    else:
      self.stackedFrames.append(preprocessedframe)
      stackedState = np.stack(self.stackedFrames, axis = 2)

    return stackedState




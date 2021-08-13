#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np

class FlappyBird:
  def __init__(self):
    self.screen = pygame.display.set_mode((400, 708))
    self.bird = pygame.Rect(65, 50, 50, 30)
    self.background = pygame.image.load("assets/background.png").convert()
    self.birdSprites = [
      pygame.image.load("assets/1.png").convert_alpha(),
      pygame.image.load("assets/2.png").convert_alpha(),
      pygame.image.load("assets/dead.png")
    ]
    self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
    self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
    self.gap = 130
    self.wallx = 400
    self.birdY = 350
    self.jump = 0
    self.jumpSpeed = 10
    self.gravity = 5
    self.dead = False
    self.sprite = 0
    self.counter = 0
    self.offset = random.randint(-110, 110)
    self.isKeyDown = False
    self.birdVelocity = None
    self.clockDisplay = None
    self.collided = False
        

  def updateWalls(self):
    self.wallx -= 2
    if self.wallx < -80:
      self.wallx = 400
      self.counter += 1
      self.offset = random.randint(-110, 110)

  def calculateWorldPositionObjects(self):
    self.worldPositions = np.array(
      [
        [
          self.wallx,
          360 + self.gap - self.offset + 10,
          self.wallUp.get_width() - 10,
          self.wallUp.get_height()
        ],
        [
          self.wallx,
          0 - self.gap - self.offset - 15,
          self.wallDown.get_width() - 10,
          self.wallDown.get_height()
        ],
        [
          self.bird[0],
          self.bird[1],
          self.bird[2],
          self.bird[3]
        ],
      ]
    )
    return self.worldPositions

  def getJumpSpeed(self):
    return self.jumpSpeed

  def getGravity(self):
    return self.gravity

  def getWorldPositionObjects(self):
      return self.worldPositions

  def isDead(self):
    return self.dead or (not (0 < self.bird[1] < 720))

  def getBirdVelocity(self):
    return self.birdVelocity

  def getGameImage(self):
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    return image_data

  def getScore(self):
    return self.counter

  def birdUpdate(self):
    if self.jump:
      self.jumpSpeed -= 1
      self.birdY -= self.jumpSpeed
      self.jump -= 1
    else:
      self.birdY += self.gravity
      self.gravity += 0.2
    self.birdVelocity = self.bird[1] - self.birdY
    self.bird[1] = self.birdY
    positions = self.calculateWorldPositionObjects()
    upRect = pygame.Rect(positions[0][0],positions[0][1],positions[0][2],positions[0][3])
    downRect = pygame.Rect(positions[1][0],positions[1][1],positions[1][2],positions[1][3])

    if upRect.colliderect(self.bird) or downRect.colliderect(self.bird):
      self.dead = True
      self.collided = True
    elif not 0 < self.bird[1] < 720:
      self.bird[1] = 150
      self.birdY = 150
      self.dead = True
      self.collided = False
      self.counter = 0
      self.wallx = 400
      self.offset = random.randint(-110, 110)
      self.gravity = 5
    else:
      if not self.collided:
        self.dead = False

  def holdKeyDown(self):
    self.isKeyDown = True

  def releaseKey(self):
    self.isKeyDown = False

  def doAction(self):
    self.jump = 17
    self.gravity = 0
    self.jumpSpeed = 10

  def eachCycle(self):
    clock = self.clock        
    font = self.font
    clock.tick(60)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
        self.doAction()
    
    if(self.isKeyDown):
      self.doAction()

    self.screen.fill((255, 255, 255))
    self.screen.blit(self.background, (0, 0))
    self.screen.blit(
      self.wallUp, (self.wallx, 360 + self.gap - self.offset)
    )
    self.screen.blit(
      self.wallDown, (self.wallx, 0 - self.gap - self.offset)
    )
    self.screen.blit(
      font.render(str(self.counter), -1, (255, 255, 255)), (200, 50)
    )
    if self.dead:
      self.sprite = 2
    elif self.jump:
      self.sprite = 1
    self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
    if not self.dead:
      self.sprite = 0
    self.updateWalls()        
    self.birdUpdate()
    pygame.display.update()
    self.clock = clock        
    self.font = font
    self.clockDisplay = clock

  def initGame(self):
    self.clock = pygame.time.Clock()
    pygame.font.init()
    self.font = pygame.font.SysFont("Arial", 50)
    self.clockDisplay = pygame.font.SysFont("Arial", 50)
  
  def run(self):        
    self.initGame()
    while True:
      self.eachCycle()
          

if __name__ == "__main__":
    FlappyBird().run()

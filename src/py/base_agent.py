import math
from pygame.math import *


class BaseAgent:
    """
    Базовый агент: пользователь наследует и реализует input() и step(out).
    self.world доступен внутри методов.
    """
    def __init__(self, world):
        self.world = world
        self.pos = Vector2(0, 0)
        self.angle = 0
        self.score = 0
        self.alive = True

    def reset(self):
        self.pos = Vector2(0, 0)
        self.angle = 0
        self.score = 0
        self.alive = True

    def input(self):
        raise NotImplementedError

    def step(self, out):
        raise NotImplementedError

    def kill(self):
        self.alive = False

    def forward_vector(self):
        return Vector2(math.cos(self.angle), math.sin(self.angle))

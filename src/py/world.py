# engine/world.py
import random, math
from typing import Type, Union
from pygame.math import *

from src.py.base_agent import BaseAgent
from src.py.main import GeneticMLP, GAController

class World:
    def __init__(self, ga, agent, agents_count, screen_size,
                 gen_steps):
        """
        agent: класс-наследник BaseAgent (constructor: agent(world))
        ga: GeneticMLP instance or GAController
        """
        self.agent = agent
        self.ctrl = ga if isinstance(ga, GAController) else GAController(ga)
        self.N = agents_count
        self.w = screen_size[0]; self.h = screen_size[1]
        self.gen_steps = gen_steps

        self.gen = 0
        self.step_count = 0
        self.mouse = Vector2(self.w/2, self.h/2)

        self.agents = []
        self._init_agents()

    def random_pos(self):
        return Vector2(random.uniform(0, self.w), random.uniform(0, self.h))

    def _init_agents(self):
        self.agents = []
        for i in range(self.N):
            a = self.agent(self)
            a.reset()
            self.agents.append(a)

    def reset_agents(self):
        for a in self.agents:
            a.reset()

    def step_all(self):
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            inputs = a.input()
            out = self.ctrl.predict_agent(i, inputs)
            r = a.step(out)
            a.score += float(r)

        self.step_count += 1
        if self.step_count >= self.gen_steps:
            self._evolve()
            return True
        return False

    def _evolve(self, crossover=0.8, mutate=0.02, strength=0.1):
        scores = [a.score for a in self.agents]
        self.ctrl.evaluate_and_step(scores, crossover=crossover, mutate=mutate, strength=strength)
        self.gen += 1
        self.step_count = 0
        self.reset_agents()

    def get_agent_states(self):
        return [(a.pos.x, a.pos.y, a.angle, a.score, a.alive) for a in self.agents]

    def get_best_index(self):
        return self.ctrl.get_best_index()
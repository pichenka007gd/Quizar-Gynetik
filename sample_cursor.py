# sample_cursor.py — very short runnable example
import random, math, numpy as np
from pygame.math import Vector2

from src.py import GeneticMLP, World, BaseAgent, run, draw_circle_agent

class FollowMouseAgent(BaseAgent):
    def reset(self):
        self.pos = Vector2(self.world.w/2, self.world.h/2)
        self.angle = random.uniform(0, 2*math.pi)

    def input(self):
        to = self.world.mouse - self.pos
        d = to.length()
        return [d / math.hypot(self.world.w, self.world.h),
                self.forward_vector().angle_to(to) / 180.0]

    def step(self, out):
        f, t = np.tanh(out[0]), np.tanh(out[1])
        self.angle += t * 0.6
        self.pos += self.forward_vector() * (f * 4.0)
        # clamp
        self.pos.x = max(0, min(self.world.w, self.pos.x))
        self.pos.y = max(0, min(self.world.h, self.pos.y))
        # reward
        to = self.world.mouse - self.pos
        d = to.length()
        fv = self.forward_vector()
        to_n = to.normalize() if d > 0 else Vector2(0,0)
        return 10*(1.0/(d + 1.0)) + 0.01 * max(0.0, fv.dot(to_n))

# setup & run (compact)
W,H = 800,600
AGENTS = 80
ga = GeneticMLP(layers=[2,8,8,2], agents=AGENTS, best_agents=4)
world = World(ga=ga, agent=FollowMouseAgent, agents_count=AGENTS,
              screen_size=(W,H), gen_steps=300)
run(world, screen_size=(W,H), fps=60, agent_draw_fn=draw_circle_agent)

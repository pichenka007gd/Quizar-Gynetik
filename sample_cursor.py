import sys, math, random
import numpy as np
import pygame
from pygame.math import *
from src.py.main import GeneticMLP, GAController
from src.py.base_agent import BaseAgent
from src.py.world import World


class FollowMouseAgent(BaseAgent):
	def __init__(self, world):
		super().__init__(world)

	def reset(self):
		self.pos = Vector2(self.world.w/2, self.world.h/2)
		self.angle = random.uniform(0, 2*math.pi)
		self.score = 0.0
		self.alive = True

	def input(self):
		to_target = self.world.mouse - self.pos
		dist = to_target.length()
		norm_dist = dist / math.hypot(self.world.w, self.world.h)
		norm_angle = self.forward_vector().angle_to(to_target) / 180.0
		return [norm_dist, norm_angle]

	def step(self, out):
		f, t = np.tanh(out[0]), np.tanh(out[1])
		self.angle += t
		self.pos += self.forward_vector() * (f * 5)
		self.pos.x = max(0, min(self.world.w, self.pos.x))
		self.pos.y = max(0, min(self.world.h, self.pos.y))

		to_target = self.world.mouse - self.pos
		d = to_target.length()
		fv = self.forward_vector()
		to_target_norm = to_target.normalize() if d>0 else Vector2(0,0)

		return 1/(d + 1) + 0.1 * max(0, fv.dot(to_target_norm))


def main():
	pygame.init()
	W, H = 800, 600
	screen = pygame.display.set_mode((W, H))
	clock = pygame.time.Clock()
	FPS = 200

	AGENTS = 200

	ga = GeneticMLP(layers=[2, 8, 8, 2], agents=AGENTS, best_agents=4)

	world = World(ga=ga, agent=FollowMouseAgent, agents_count=AGENTS,
				  screen_size=(W, H), gen_steps=300)

	running = True
	while running:
		for ev in pygame.event.get():
			if ev.type == pygame.QUIT:
				running = False

		mouse = pygame.mouse.get_pos()
		world.mouse = Vector2(mouse)
		world.step_all()

		screen.fill((28, 28, 35))

		pygame.draw.circle(screen, (255, 80, 80), (int(mouse[0]), int(mouse[1])), 6)
		best_idx = world.get_best_index()
		for i, (x, y, angle, score, alive) in enumerate(world.get_agent_states()):
			col = (140, 220, 140) if i != best_idx else (255, 220, 90)

			if not alive:
				col = (80, 80, 80)
			pygame.draw.circle(screen, col, (int(x), int(y)), 4)
			ex = x + 10*math.cos(angle)
			ey = y + 10*math.sin(angle)
			pygame.draw.line(screen, col, (int(x), int(y)), (int(ex), int(ey)), 1)

		font = pygame.font.SysFont(None, 20)
		txt = font.render(f"Gen: {world.gen}  Step: {world.step_count}/{world.gen_steps}", True, (200,200,200))
		screen.blit(txt, (8, 8))
		if world.step_count % 2 == 0:
			pygame.display.flip()
			clock.tick(FPS)

	pygame.quit()
	sys.exit()

if __name__ == "__main__":
	main()

import math
import pygame
from pygame.math import Vector2

def draw_circle_agent(agent, surface, color=(140,220,140), radius=6, dir_len=12, dir_color=None):
    """Простая отрисовка агента: круг + направляющая линия."""
    x, y = int(agent.pos.x), int(agent.pos.y)
    if dir_color is None:
        dir_color = color
    pygame.draw.circle(surface, color, (x, y), radius)
    ex = x + int(math.cos(agent.angle) * dir_len)
    ey = y + int(math.sin(agent.angle) * dir_len)
    pygame.draw.line(surface, dir_color, (x, y), (ex, ey), 2)

def draw_circle(world, surface, pos, radius=6, color=(255,80,80)):
    """Нарисовать окружность в координатах pos (Vector2 или tuple)."""
    if hasattr(pos, "x"):
        x, y = int(pos.x), int(pos.y)
    else:
        x, y = int(pos[0]), int(pos[1])
    pygame.draw.circle(surface, color, (x, y), radius)

def attach_drawable_api(world):
    """
    Добавляет в экземпляр world методы:
      - add_circle(pos, radius=6, color=(..))
      - clear_drawables()
      - draw(surface)  # рисует примитивы, если они есть
    Хранит примитивы в world._drawables.
    """
    if hasattr(world, "_drawables"):
        return  # уже прикреплено

    world._drawables = []

    def add_circle(pos, radius=6, color=(255,80,80)):
        p = pos if hasattr(pos, "x") else Vector2(pos[0], pos[1])
        world._drawables.append(("circle", p, radius, color))

    def clear_drawables():
        world._drawables.clear()

    def draw(surface):
        # отрисовать примитивы мира
        for prim in world._drawables:
            kind = prim[0]
            if kind == "circle":
                _, p, r, col = prim
                pygame.draw.circle(surface, col, (int(p.x), int(p.y)), r)
            # можно расширить: rect, line, text ...
        # также можно прорисовать дополнительные элементы мира, если есть world.custom_draw
        if hasattr(world, "custom_draw") and callable(world.custom_draw):
            try:
                world.custom_draw(surface)
            except Exception:
                pass

    # привязываем методы
    world.add_circle = add_circle
    world.clear_drawables = clear_drawables
    world.draw = draw

    return world

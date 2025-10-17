import sys
import pygame
from pygame.math import Vector2
from .draw import draw_circle_agent, attach_drawable_api

def run(world,
        screen_size=(800,600),
        fps=60,
        bg_color=(28,28,35),
        agent_draw_fn=None,
        show_mouse=True,
        auto_attach_draw_api=True):
    """
    Запускает Pygame-цикл для world.
    - world: экземпляр класса World (должен иметь step_all(), get_agent_states(), get_best_index(), reset_agents() и поле mouse)
    - agent_draw_fn(agent, surface) - функция отрисовки агента, по умолчанию draw_circle_agent
    - Если у агента реализован метод draw(self, surface), он будет вызван вместо agent_draw_fn.
    """
    if agent_draw_fn is None:
        agent_draw_fn = draw_circle_agent

    if auto_attach_draw_api:
        attach_drawable_api(world)

    pygame.init()
    W, H = screen_size
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # обновляем позицию мыши в world (если используется)
        mouse = pygame.mouse.get_pos()
        # поддерживаем Vector2 или tuple
        world.mouse = Vector2(mouse[0], mouse[1])

        # шаг симуляции (world должен реализовать step_all())
        world.step_all()

        # отрисовка
        screen.fill(bg_color)

        # рисуем примитивы мира (если прикреплены)
        if hasattr(world, "draw"):
            try:
                world.draw(screen)
            except Exception:
                pass

        # рисуем мышь (опционально)
        if show_mouse:
            pygame.draw.circle(screen, (255,80,80), (int(mouse[0]), int(mouse[1])), 6)

        # отрисовка агентов: либо agent.draw(surface) если есть, либо общий agent_draw_fn
        best_idx = None
        try:
            best_idx = world.get_best_index()
        except Exception:
            pass

        for i, agent_state in enumerate(world.get_agent_states()):
            x, y, angle, score, alive = agent_state
            # найти объект агента по индексу, если world хранит их
            agent_obj = None
            if hasattr(world, "agents"):
                try:
                    agent_obj = world.agents[i]
                except Exception:
                    agent_obj = None

            # prefer agent-defined draw
            if agent_obj is not None and hasattr(agent_obj, "draw") and callable(agent_obj.draw):
                try:
                    agent_obj.draw(screen)
                except Exception:
                    # fallback to default
                    agent_draw_fn(agent_obj, screen)
            else:
                # если нет объекта — нарисуем простую метку
                # но если agent_obj есть — используем его для параметров цвета/радиуса
                if agent_obj is not None:
                    agent_draw_fn(agent_obj, screen,
                                  color=(255,220,90) if i==best_idx else (140,220,140))
                else:
                    # draw from state only
                    col = (255,220,90) if i==best_idx else (140,220,140)
                    pygame.draw.circle(screen, col, (int(x), int(y)), 4)
                    ex = x + 10 * pygame.math.cos(angle)
                    ey = y + 10 * pygame.math.sin(angle)
                    pygame.draw.line(screen, col, (int(x), int(y)), (int(ex), int(ey)), 1)

        # HUD (если world имеет gen / step_count)
        if hasattr(world, "gen") and hasattr(world, "step_count") and hasattr(world, "steps_per_gen"):
            font = pygame.font.SysFont(None, 20)
            txt = font.render(f"Gen: {world.gen}  Step: {world.step_count}/{world.steps_per_gen}", True, (200,200,200))
            screen.blit(txt, (8, 8))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()

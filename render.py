import pygame
import numpy as np
from collections import deque
from data_models import World

class RenderSystem:
    def __init__(self, cell_size: int = 5, sidebar_width: int = 250):
        pygame.init()
        self.cell_size = cell_size
        self.sidebar_width = sidebar_width
        self.screen = None
        self.font = pygame.font.SysFont("Consolas", 14)
        
        # История данных для графиков (храним последние 200 тиков)
        self.history_len = 200
        self.pop_history = deque([0] * self.history_len, maxlen=self.history_len)
        self.pred_history = deque([0] * self.history_len, maxlen=self.history_len)
        self.tank_history = deque([0] * self.history_len, maxlen=self.history_len)

    def _init_display(self, world: World):
        self.world_w = world.width * self.cell_size
        self.world_h = world.height * self.cell_size
        self.total_w = self.world_w + self.sidebar_width
        
        self.screen = pygame.display.set_mode((self.total_w, self.world_h))
        pygame.display.set_caption("Onigafuti: Lab Analytics")

    def _draw_graph(self, data, color, rect, label):
        """Отрисовка одного графика в заданном прямоугольнике.
        FIX #5: Vectorized point computation — no more Python for-loop over deque.
        """
        if not data: return

        pygame.draw.rect(self.screen, (20, 20, 20), rect)

        arr     = np.asarray(data, dtype=np.float32)
        max_val = float(max(arr.max(), 1.0))

        # Vectorized: compute all (x, y) pairs in one shot
        indices = np.arange(len(arr), dtype=np.float32)
        xs = rect.x + (indices / self.history_len) * rect.width
        ys = rect.y + rect.height - (arr / max_val) * rect.height
        points = np.column_stack([xs, ys]).tolist()

        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

        text = self.font.render(f"{label}: {int(arr[-1])}", True, color)
        self.screen.blit(text, (rect.x, rect.y - 18))

    def draw(self, world: World, max_veg: float = 20.0, max_corpse: float = 50.0):
        if self.screen is None:
            self._init_display(world)

        # 1. СТАТИСТИКА (Собираем данные до отрисовки)
        alive_mask = world.agents.is_alive
        pop_count = np.count_nonzero(alive_mask)
        
        if pop_count > 0:
            # Хищники: укус > 10.0, Танки: броня > 0.5
            pred_count = np.count_nonzero(world.agents.bite_force[alive_mask] > 10.0)
            tank_count = np.count_nonzero(world.agents.defense[alive_mask] > 0.5)
        else:
            pred_count = tank_count = 0

        self.pop_history.append(pop_count)
        self.pred_history.append(pred_count)
        self.tank_history.append(tank_count)

        # 2. МИР (Рендерим симуляцию на отдельную поверхность)
        bg_rgb = np.zeros((world.width, world.height, 3), dtype=np.uint8)
        bg_rgb[:, :, 1] = np.clip((world.vegetation / max_veg) * 255, 0, 255).astype(np.uint8)
        bg_rgb[:, :, 2] = np.clip((world.corpse_map / max_corpse) * 255, 0, 255).astype(np.uint8)

        if pop_count > 0:
            x, y = world.agents.x[alive_mask], world.agents.y[alive_mask]
            r = (100 + (world.agents.bite_force[alive_mask] / 20.0) * 155).astype(np.uint8)
            b = ((world.agents.defense[alive_mask] / 0.9) * 255).astype(np.uint8)
            bg_rgb[x, y, 0] = r
            bg_rgb[x, y, 2] = b

        world_surf = pygame.surfarray.make_surface(bg_rgb)
        if self.cell_size > 1:
            world_surf = pygame.transform.scale(world_surf, (self.world_w, self.world_h))
        
        # Выводим мир на экран
        self.screen.fill((10, 10, 10)) # Очистка фона
        self.screen.blit(world_surf, (0, 0))

        # 3. ИНТЕРФЕЙС (Sidebar)
        # Рисуем три графика друг под другом
        graph_h = 80
        padding = 40
        start_x = self.world_w + 20
        
        # График Популяции (Белый)
        self._draw_graph(self.pop_history, (200, 200, 200), 
                         pygame.Rect(start_x, 30, self.sidebar_width - 40, graph_h), "Population")
        
        # График Хищников (Красный)
        self._draw_graph(self.pred_history, (255, 50, 50), 
                         pygame.Rect(start_x, 30 + graph_h + padding, self.sidebar_width - 40, graph_h), "Predators")
        
        # График Танков (Синий)
        self._draw_graph(self.tank_history, (50, 100, 255), 
                         pygame.Rect(start_x, 30 + (graph_h + padding) * 2, self.sidebar_width - 40, graph_h), "Tanks")

        pygame.display.flip()

    def quit(self):
        pygame.quit()
import pygame
import random
import numpy as np
from data_models import World, Agent
from environment import BioSystem, EnergySystem
from entities import AgentSystem
from render import RenderSystem

def main():
    # 1. Инициализация Мира
    width, height = 100, 100
    world = World(width, height)
    
    # Настраиваем маятник солнца (амплитуда чуть меньше границ, чтобы не билось о стены)
    world.sun_params = {
        'amplitude_x': (width / 2) - 10,
        'amplitude_y': (height / 2) - 10,
        'angular_velocity': 0.02,
        'drip_rate': 25.0  # Увеличили порцию энергии
    }
    
    # 2. "Шумный старт" — Заселяем планету первичной зеленью
    # Это заменит твои ручные world.vegetation += 0.1
    world.seed_life(vegetation_chance=0.08, nutrient_range=(0.2, 0.6))
    
    # Даем огромный стартовый запас энергии для Солнца (Leak Buffer)
    world.leak_buffer = 15000.0 

    # 3. Спавним Первопроходцев
    # Сделаем их чуть выносливее (metabolism 0.1 вместо 0.5), чтобы они успели найти еду
    for i in range(60):
        a = Agent(
            id=i, 
            x=random.randint(0, width-1), 
            y=random.randint(0, height-1), 
            energy=60.0, 
            genome={
                'bite_force': 4.0,   # Увеличили КПД укуса
                'metabolism': 0.15,  # Снизили расход на жизнь
                'randomness': 0.2    # Шанс забить на запах и пойти куда глаза глядят
            }
        )
        # Проверяем, не занята ли клетка другим агентом
        if world.agent_map[a.x, a.y] == -1:
            world.agents.append(a)
            world.agent_map[a.x, a.y] = a.id

    # 4. Инициализация Систем
    bio_sys = BioSystem()
    energy_sys = EnergySystem()
    agent_sys = AgentSystem()
    render_sys = RenderSystem(cell_size=7) # Чуть крупнее для красоты

    clock = pygame.time.Clock()
    running = True
    
    print("--- Симуляция Онигафути запущена ---")
    
    while running:
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- ЦИКЛ СИМУЛЯЦИИ ---
        
        # А) Среда: Диффузия, рост травы и гниение трупов
        bio_sys.update(world) 
        
        # Б) Энергия: Движение маятника солнца и впрыск нутриентов
        energy_sys.update(world)
        
        # В) Логика: Принюхивание, движение, питание и размножение
        agent_sys.update(world)

        # Г) Визуал: Черный фон, зеленые луга, синие трупы, красные герои
        render_sys.draw(world)

        # Мониторинг популяции в консоли (раз в секунду)
        if pygame.time.get_ticks() % 1000 < 20:
            print(f"Population: {len(world.agents)} | Leak Buffer: {int(world.leak_buffer)}")
        
        if pygame.time.get_ticks() % 100 == 0:
            world.leak_buffer += 5.0

        clock.tick(60)

    render_sys.quit()

if __name__ == "__main__":
    main()
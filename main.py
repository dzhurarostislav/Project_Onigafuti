import pygame
import numpy as np
import random

from data_models import World
from environment import BioSystem, EnergySystem, ScentSystem
from entities import AgentSystem
from render import RenderSystem

def main():
    # 1. Инициализация Мира (сразу резервируем память под 10 000 агентов)
    width, height = 60, 60
    world = World(width, height, max_agents=10000, n_inputs=10, n_outputs=5)
    
    world.sun_params = {
        'amplitude_x': (width / 2) - 10,
        'amplitude_y': (height / 2) - 10,
        'angular_velocity': 0.01,    # БЫЛО 0.02. Замедляем движение в 2 раза.
        'base_drip': 15.0,
        'bonus_drip_max': 20.0
    }
    
    # 2. Шумный старт
    world.seed_life(vegetation_chance=0.08, nutrient_range=(0.2, 0.6))
    world.leak_buffer = 15000.0 

    # 3. Спавним Первопроходцев (Адам и Ева x30)
    # Настраиваем размерности нашей простейшей нейросети
    N_INPUTS = 10
    N_OUTPUTS = 5
    
    initial_population = 60
    for _ in range(initial_population):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # Если клетка занята, пропускаем (естественный отбор начинается до рождения)
        if world.agent_map[x, y] != -1:
            continue
            
        # ГЕНЕЗИС: Создаем случайный мозг (матрицу весов)
        # Нормальное распределение вокруг нуля, чтобы агенты не дергались слишком резко в начале
        brain_weights = np.random.normal(0, 0.5, size=(N_INPUTS, N_OUTPUTS)).astype(np.float32)
        
        # Регистрируем агента в SoA-структуре за О(1)
        try:
            idx = world.agents.spawn(
                x=x, 
                y=y, 
                energy=200.0,  # Даем больше энергии на старт, пока они тупые
                weights=brain_weights,
                metabolism=0.15,
                bite_force=4.0,
                max_age=500
            )
            # Записываем ID (индекс) в карту мира для системы коллизий
            world.agent_map[x, y] = idx
        except RuntimeError as e:
            print(f"Ошибка спавна: {e}")
            break

    # 4. Инициализация Систем (Stateless процессоры)
    bio_sys = BioSystem()
    energy_sys = EnergySystem()
    scent_sys = ScentSystem()  # ВОССТАНОВЛЕНО: Система феромонов
    agent_sys = AgentSystem()
    render_sys = RenderSystem(cell_size=7) 

    clock = pygame.time.Clock()
    running = True
    ticks = 0
    
    print("--- Симуляция Onigafuti: Digital Evolution запущена ---")
    
    while running:
        # Обработка событий окна
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- ЦИКЛ СИМУЛЯЦИИ (Строгий порядок важен) ---
        
        # А) Среда: Диффузия нутриентов, рост растительности, гниение трупов
        bio_sys.update(world, decay_rate=0.01) 
        
        # Б) Энергия: Движение солнца и впрыск новой энергии на субстрат
        energy_sys.update(world)
        
        # В) Феромоны: Испарение запаха от растительности и его диффузия
        scent_sys.update(world)
        
        # Г) Агенты: Метаболизм -> Смерть -> Питание -> Нейронное движение -> Размножение
        agent_sys.update(world)

        # Д) Визуал: Отрисовка матриц через Pygame.surfarray
        render_sys.draw(world)

        # Мониторинг (Без циклов! Используем векторизованный подсчет живых)
        if ticks % 60 == 0:
            alive_count = np.count_nonzero(world.agents.is_alive)
            print(f"Tick: {ticks} | Population: {alive_count} | Leak Buffer: {int(world.leak_buffer)}")
            
            if alive_count == 0:
                print("Вымирание. Эксперимент провален.")
                running = False
        
        if ticks > 0 and ticks % 500 == 0:
            alive_indices = np.where(world.agents.is_alive)[0]
            if len(alive_indices) > 0:
                ages = world.agents.age[alive_indices]
                kills = world.agents.kills[alive_indices]
                
                # Находим индексы чемпионов через argmax
                oldest_idx = alive_indices[np.argmax(ages)]
                killer_idx = alive_indices[np.argmax(kills)]
                
                print(f"\n{'='*40}")
                print(f"ОТЧЕТ ЭВОЛЮЦИИ (Tick {ticks})")
                print(f"{'='*40}")
                
                print(f"СТАРЕЙШИНА [ID {oldest_idx}]:")
                print(f"  Возраст: {world.agents.age[oldest_idx]} | Фраги: {world.agents.kills[oldest_idx]} | Энергия: {int(world.agents.energy[oldest_idx])}")
                
                print(f"\nАЛЬФА-ХИЩНИК [ID {killer_idx}]:")
                print(f"  Фраги: {world.agents.kills[killer_idx]} | Возраст: {world.agents.age[killer_idx]} | Энергия: {int(world.agents.energy[killer_idx])}")
                
                # Читаем матрицу весов (Геном) Альфа-хищника
                brain = world.agents.brain_weights[killer_idx]
                
                # Нас интересуют веса реакции на соседей (входы 4, 5, 6, 7)
                # Выходы нейронки: 0:Стоять, 1:Вверх, 2:Вниз, 3:Влево, 4:Вправо
                # Если вес положительный — агент хочет идти в занятую клетку (атака)
                # Если отрицательный — агент избегает столкновения (трусость)
                print("\nГЕНЕТИЧЕСКИЙ АНАЛИЗ АЛЬФЫ (Рефлексы Атаки):")
                print(f"  Видит сверху -> Удар вверх: {brain[4, 1]:.2f}")
                print(f"  Видит снизу  -> Удар вниз:  {brain[5, 2]:.2f}")
                print(f"  Видит слева  -> Удар влево: {brain[6, 3]:.2f}")
                print(f"  Видит справа -> Удар вправо: {brain[7, 4]:.2f}")
                print(f"  Сила Укуса   -> {world.agents.bite_force[killer_idx]}")
                print(f"  Броня        -> {world.agents.defense[killer_idx]}")
                print(f"{'='*40}\n")
        ticks += 1
        clock.tick(60) # Ограничение FPS

    render_sys.quit()

if __name__ == "__main__":
    main()
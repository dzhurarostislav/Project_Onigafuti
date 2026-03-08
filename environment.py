import numpy as np

class BioSystem:
    """Handles biological grid operations: diffusion and vegetation growth."""
    
    def update(self, world, diffusion_rate=0.1, threshold=0.01, growth_rate=0.05, max_veg=20000.0, decay_rate=0.005):
        # --- 1. Vectorized Diffusion (Clean Hard Borders) ---
        spread_mask = world.nutrients > threshold
        energy_to_diffuse = np.zeros_like(world.nutrients)
        energy_to_diffuse[spread_mask] = world.nutrients[spread_mask] * diffusion_rate
        
        world.nutrients -= energy_to_diffuse
        per_neighbor = energy_to_diffuse / 4.0
        
        # Сдвигаем во все 4 стороны (Von Neumann neighborhood)
        rolled_right = np.roll(per_neighbor, 1, axis=1)
        rolled_left = np.roll(per_neighbor, -1, axis=1)
        rolled_down = np.roll(per_neighbor, 1, axis=0)
        rolled_up = np.roll(per_neighbor, -1, axis=0)
        
        # Обнуляем телепортировавшуюся через края энергию
        rolled_right[:, 0] = 0
        rolled_left[:, -1] = 0
        rolled_down[0, :] = 0
        rolled_up[-1, :] = 0
        
        # Считаем то, что ушло за края (в leak_buffer)
        leak = np.sum(per_neighbor[:, -1]) + np.sum(per_neighbor[:, 0]) + \
               np.sum(per_neighbor[-1, :]) + np.sum(per_neighbor[0, :])
        world.leak_buffer += leak
        
        # Применяем диффузию
        world.nutrients += (rolled_right + rolled_left + rolled_down + rolled_up)
        
        # --- 2. Vegetation Growth ---
        growth = world.nutrients * growth_rate
        actual_growth = np.minimum(growth, max_veg - world.vegetation)
        world.vegetation += actual_growth
        world.nutrients -= actual_growth

        # --- 3. Corpse Decay ---
        decay = world.corpse_map * decay_rate
        world.corpse_map -= decay
        world.nutrients += decay


class EnergySystem:
    """Handles global energy dynamics: sun movement and energy dripping."""
    
    def update(self, world, radius=2):
        params = world.sun_params
        world.sun_angle += params['angular_velocity'] + np.random.normal(0, 0.002)
        
        # --- Случайное блуждание с ПРУЖИНОЙ (Mean Reversion) ---
        base_amp_x = world.width * 0.4
        base_amp_y = world.height * 0.4
        
        params['amplitude_x'] += np.random.uniform(-0.5, 0.5)
        params['amplitude_y'] += np.random.uniform(-0.5, 0.5)
        
        # Пружина тянет амплитуду обратно к базе на 2% за тик, чтобы не улетела
        params['amplitude_x'] += (base_amp_x - params['amplitude_x']) * 0.02
        params['amplitude_y'] += (base_amp_y - params['amplitude_y']) * 0.02
        
        cx, cy = world.width / 2.0, world.height / 2.0
        world.sun_pos[0] = cx + params['amplitude_x'] * np.cos(world.sun_angle)
        world.sun_pos[1] = cy + params['amplitude_y'] * np.sin(world.sun_angle * 1.1)
        
        world.sun_pos = np.clip(world.sun_pos, radius, [world.width - radius, world.height - radius])
        
        # --- Energy Drip ---
        bonus_drip = min(world.leak_buffer, params.get('bonus_drip_max', 20.0))
        drip_amount = params.get('base_drip', 5.0) + bonus_drip
        
        if drip_amount > 0:
            world.leak_buffer -= bonus_drip
            icx, icy = int(world.sun_pos[0]), int(world.sun_pos[1])
            x_start, x_end = max(0, icx - radius), min(world.width, icx + radius + 1)
            y_start, y_end = max(0, icy - radius), min(world.height, icy + radius + 1)
            
            area = (x_end - x_start) * (y_end - y_start)
            if area > 0:
                world.nutrients[x_start:x_end, y_start:y_end] += drip_amount / area


class ScentSystem:
    """Handles environmental marking: creates a cumulative scent gradient."""

    def update(self, world, scent_decay=0.05, diffusion_rate=0.2):
        # 1. Еда выделяет новый запах каждый тик
        world.scent_map += world.vegetation * 0.1
        # FIX #4: Clamp scent_map to prevent overflow.
        # Without this, values grow to astronomical floats and the neural scent input saturates,
        # making it indistinguishable across all cells (all agents see the "same" amount of food).
        np.clip(world.scent_map, 0.0, 50.0, out=world.scent_map)

        # Теперь агенты сами наносят феромоны в AgentSystem!

        # 2. Быстрая диффузия запаха еды (Von Neumann)
        scent = world.scent_map
        diffused = (
            np.roll(scent, 1, axis=0) + np.roll(scent, -1, axis=0) +
            np.roll(scent, 1, axis=1) + np.roll(scent, -1, axis=1)
        ) / 4.0
        world.scent_map = scent * (1.0 - diffusion_rate) + diffused * diffusion_rate

        # 3. Диффузия ФЕРОМОНОВ (растекание следа толпы)
        pheromones = world.pheromone_map
        diffused_pheromones = (
            np.roll(pheromones, 1, axis=0) + np.roll(pheromones, -1, axis=0) +
            np.roll(pheromones, 1, axis=1) + np.roll(pheromones, -1, axis=1)
        ) / 4.0
        world.pheromone_map = pheromones * (1.0 - diffusion_rate) + diffused_pheromones * diffusion_rate

        # 4. Выветривание, чтобы карта не переполнилась цифрами
        world.scent_map *= (1.0 - scent_decay)
        
        # Феромоны выветриваются чуть быстрее (например, 10% за тик), 
        # чтобы старые тропы исчезали, и микробы не бегали по кругу
        world.pheromone_map *= 0.9

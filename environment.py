import numpy as np

class BioSystem:
    """Handles biological grid operations: diffusion and vegetation growth."""
    
    def update(self, world, diffusion_rate=0.1, threshold=0.01, growth_rate=0.05, max_veg=20.0, decay_rate=0.1):
        """
        Updates the world nutrients via diffusion, grows vegetation, and decays corpses.
        """
        # --- 1. Vectorized Diffusion ---
        spread_mask = world.nutrients > threshold
        energy_to_diffuse = np.zeros_like(world.nutrients)
        energy_to_diffuse[spread_mask] = world.nutrients[spread_mask] * diffusion_rate
        
        world.nutrients -= energy_to_diffuse
        per_neighbor = energy_to_diffuse / 4.0
        
        leak_right = np.sum(per_neighbor[:, -1])
        world.nutrients += np.roll(per_neighbor, 1, axis=1)
        world.nutrients[:, 0] -= per_neighbor[:, -1]
        
        leak_left = np.sum(per_neighbor[:, 0])
        world.nutrients += np.roll(per_neighbor, -1, axis=1)
        world.nutrients[:, -1] -= per_neighbor[:, 0]
        
        leak_down = np.sum(per_neighbor[-1, :])
        world.nutrients += np.roll(per_neighbor, 1, axis=0)
        world.nutrients[0, :] -= per_neighbor[-1, :]
        
        leak_up = np.sum(per_neighbor[0, :])
        world.nutrients += np.roll(per_neighbor, -1, axis=0)
        world.nutrients[-1, :] -= per_neighbor[0, :]
        
        world.leak_buffer += (leak_right + leak_left + leak_down + leak_up)
        
        # --- 2. Vegetation Growth (Buffed) ---
        growth = world.nutrients * growth_rate
        actual_growth = np.minimum(growth, np.maximum(0, max_veg - world.vegetation))
        world.vegetation += actual_growth
        world.nutrients -= actual_growth

        # --- 3. Corpse Decay (New) ---
        decay = world.corpse_map * decay_rate
        world.corpse_map -= decay
        world.nutrients += decay

class EnergySystem:
    """Handles global energy dynamics: sun movement and energy dripping."""
    
    def update(self, world, radius=4):
        """
        Updates the sun position and drips energy into the environment.
        """
        # --- 1. Sun Movement (Random Pendulum) ---
        params = world.sun_params
        noise = np.random.normal(0, 0.002)
        world.sun_angle += params['angular_velocity'] + noise
        
        params['amplitude_x'] += np.random.uniform(-0.1, 0.1)
        params['amplitude_y'] += np.random.uniform(-0.1, 0.1)
        
        cx, cy = world.width / 2.0, world.height / 2.0
        world.sun_pos[0] = cx + params['amplitude_x'] * np.cos(world.sun_angle)
        world.sun_pos[1] = cy + params['amplitude_y'] * np.sin(world.sun_angle * 1.1)
        
        world.sun_pos = np.clip(world.sun_pos, radius, [world.width - radius, world.height - radius])
        
        # --- 2. Energy Drip (Balanced) ---
        # base_drip is constant, bonus comes from leak_buffer
        bonus_drip = min(world.leak_buffer, params.get('bonus_drip_max', 20.0))
        drip_amount = params.get('base_drip', 5.0) + bonus_drip
        
        if drip_amount > 0:
            world.leak_buffer -= bonus_drip
            
            icx, icy = int(world.sun_pos[0]), int(world.sun_pos[1])
            x_start = max(0, icx - radius)
            x_end = min(world.width, icx + radius + 1)
            y_start = max(0, icy - radius)
            y_end = min(world.height, icy + radius + 1)
            
            area = (x_end - x_start) * (y_end - y_start)
            if area > 0:
                world.nutrients[x_start:x_end, y_start:y_end] += drip_amount / area

class ScentSystem:
    """Handles environmental marking: creates a scent gradient based on vegetation."""

    def update(self, world):
        """
        Updates the world scent_map by blurring the vegetation grid.
        
        Logic:
        1. Apply 3x3 box filter (averaging) using np.roll.
        2. Normalize/clamp the result to prevent float32 accumulation issues.
        """
        # Start with its own value
        veg = world.vegetation.astype(np.float32)
        scent = veg.copy()
        
        # 4 Cardinal directions
        scent += np.roll(veg, 1, axis=0)
        scent += np.roll(veg, -1, axis=0)
        scent += np.roll(veg, 1, axis=1)
        scent += np.roll(veg, -1, axis=1)
        
        # 4 Diagonal directions
        scent += np.roll(np.roll(veg, 1, axis=0), 1, axis=1)
        scent += np.roll(np.roll(veg, 1, axis=0), -1, axis=1)
        scent += np.roll(np.roll(veg, -1, axis=0), 1, axis=1)
        scent += np.roll(np.roll(veg, -1, axis=0), -1, axis=1)
        
        # Average (Box Filter)
        scent /= 9.0
        
        # Write back to world
        world.scent_map[:] = scent

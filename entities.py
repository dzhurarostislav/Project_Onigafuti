import numpy as np
import random
from data_models import World

class AgentSystem:
    """Handles vectorized logic for agent life cycles, neural movement, and feeding."""

    def __init__(self):
        # Pre-allocated input buffer — avoids a per-tick np.zeros allocation (FIX #2)
        self._inputs_buf: np.ndarray | None = None

    def update(self, world: World):
        agents = world.agents

        # 0. Проверка популяции
        alive_mask = agents.is_alive
        if not np.any(alive_mask):
            return

        alive_indices = np.where(alive_mask)[0]
        N = len(alive_indices)

        # Захватываем текущие координаты живых
        x_alive = agents.x[alive_indices]
        y_alive = agents.y[alive_indices]

        # Нанесение феромонов
        world.pheromone_map[x_alive, y_alive] += 1.0

        # ==========================================
        # 1. NEURAL THINKING (Сначала решаем, что делать)
        # ==========================================
        up_y    = (y_alive - 1) % world.height
        down_y  = (y_alive + 1) % world.height
        left_x  = (x_alive - 1) % world.width
        right_x = (x_alive + 1) % world.width

        vis_up    = (world.agent_map[x_alive, up_y]    != -1).astype(np.float32)
        vis_down  = (world.agent_map[x_alive, down_y]  != -1).astype(np.float32)
        vis_left  = (world.agent_map[left_x, y_alive]  != -1).astype(np.float32)
        vis_right = (world.agent_map[right_x, y_alive] != -1).astype(np.float32)

        # --- Lazily grow the input buffer to avoid a per-tick allocation ---
        if self._inputs_buf is None or self._inputs_buf.shape[0] < N:
            self._inputs_buf = np.zeros((max(N + 128, 512), 10), dtype=np.float32)
        inputs = self._inputs_buf[:N]
        inputs[:] = 0.0  # fast in-place zero (single memset)

        inputs[:, 0] = 1.0                                       # Bias
        inputs[:, 1] = agents.energy[alive_indices] / 1000.0    # Энергия
        inputs[:, 2] = world.scent_map[x_alive, y_alive]        # Запах травы
        # FIX #9: Deterministic internal clock instead of pure noise.
        # Allows stable behavioral policies to evolve; eliminates jitter that drowns learned strategies.
        inputs[:, 3] = np.sin(agents.age[alive_indices] * 0.1)  # Внутренние часы
        inputs[:, 4] = vis_up
        inputs[:, 5] = vis_down
        inputs[:, 6] = vis_left
        inputs[:, 7] = vis_right
        inputs[:, 8] = world.pheromone_map[x_alive, y_alive]    # Феромоны
        inputs[:, 9] = agents.defense[alive_indices]             # Броня

        # Einsum: вычисляем решения (N, 5)
        decisions  = np.einsum('ni,nio->no', inputs, agents.brain_weights[alive_indices])
        directions = np.argmax(decisions, axis=1)

        # Уменьшаем таймер переваривания
        in_cooldown = agents.cooldown[alive_indices] > 0
        agents.cooldown[alive_indices[in_cooldown]] -= 1

        # Те, кто переваривает, не могут двигаться и атаковать
        directions[in_cooldown] = 0

        # ==========================================
        # 2. METABOLISM & AGEING
        # ==========================================
        cost       = agents.metabolism[alive_indices] + (agents.defense[alive_indices] * 0.2)
        multiplier = np.ones(N, dtype=np.float32)
        multiplier[directions == 0] = 0.5

        agents.energy[alive_indices] -= cost * multiplier
        agents.age[alive_indices]    += 1

        # ==========================================
        # 3. DEATH (Зачистка мертвых)
        # ==========================================
        starved_mask = (
            (agents.energy[alive_indices] <= 0) |
            (agents.age[alive_indices] > agents.max_age[alive_indices])
        )

        if np.any(starved_mask):
            dead_idx = alive_indices[starved_mask]
            world.corpse_map[agents.x[dead_idx], agents.y[dead_idx]] += 50.0
            world.agent_map[agents.x[dead_idx],  agents.y[dead_idx]] = -1
            for idx in dead_idx:
                agents.kill(idx)

        # Recompute survivors; filter directions to match
        alive_mask = agents.is_alive
        if not np.any(alive_mask):
            return
        alive_indices = np.where(alive_mask)[0]
        N             = len(alive_indices)
        directions    = directions[~starved_mask]

        # ==========================================
        # 4. EAT (Растительная диета)
        # ==========================================
        x_now, y_now = agents.x[alive_indices], agents.y[alive_indices]
        veg_at_feet  = world.vegetation[x_now, y_now]

        BASE_BITE_SIZE = 5.0
        actual_bites   = np.minimum(veg_at_feet, BASE_BITE_SIZE)
        world.vegetation[x_now, y_now] -= actual_bites

        herb_efficiency = np.maximum(0.0, 1.0 - (agents.bite_force[alive_indices] / 20.0))
        agents.energy[alive_indices] += actual_bites * 5.0 * herb_efficiency

        # ==========================================
        # 5. MOVE & PREDATION
        # FIX #1: Vectorized batch free-moves; small loop only for the rare attack case.
        # ==========================================
        dx = np.zeros(N, dtype=np.int32)
        dy = np.zeros(N, dtype=np.int32)
        dx[directions == 3] = -1
        dx[directions == 4] =  1
        dy[directions == 1] = -1
        dy[directions == 2] =  1

        desired_x = (agents.x[alive_indices] + dx) % world.width
        desired_y = (agents.y[alive_indices] + dy) % world.height

        moving_local = np.where(directions != 0)[0]  # local indices into alive_indices

        if len(moving_local) > 0:
            mg       = alive_indices[moving_local]   # global IDs for movers
            nx_all   = desired_x[moving_local]
            ny_all   = desired_y[moving_local]
            target_ids = world.agent_map[nx_all, ny_all]

            # --- FREE MOVES (target cell is empty) — fully vectorized ---
            free_local_mask = (target_ids == -1)
            free_local      = np.where(free_local_mask)[0]

            if len(free_local) > 0:
                # Conflict resolution: if two agents want the same empty cell,
                # only the first one by index order actually moves.
                keys       = nx_all[free_local].astype(np.int64) * world.height + ny_all[free_local]
                _, first_occ = np.unique(keys, return_index=True)

                fg  = mg[free_local[first_occ]]
                fnx = nx_all[free_local[first_occ]]
                fny = ny_all[free_local[first_occ]]

                # Batch-update agent_map and position arrays
                world.agent_map[agents.x[fg], agents.y[fg]] = -1
                world.agent_map[fnx, fny]                   = fg
                agents.x[fg] = fnx
                agents.y[fg] = fny

            # --- ATTACKS (target cell occupied — small loop, justified) ---
            attack_local = np.where((~free_local_mask) & (target_ids != mg))[0]

            for k in attack_local:
                idx       = int(mg[k])
                target_id = int(target_ids[k])
                tnx, tny  = int(nx_all[k]), int(ny_all[k])

                if not agents.is_alive[target_id]:
                    continue

                raw_dmg    = agents.bite_force[idx] * 10.0
                final_dmg  = raw_dmg * (1.0 - agents.defense[target_id])
                actual_dmg = min(float(agents.energy[target_id]), final_dmg)

                agents.energy[target_id] -= actual_dmg
                agents.energy[idx]       += actual_dmg * 0.9

                if agents.energy[target_id] <= 0:
                    agents.kills[idx]      += 1
                    agents.bite_force[idx]  = min(agents.bite_force[idx] + 0.5, 20.0)
                    agents.cooldown[idx]    = 10

                    # FIX #8: Properly clear map slot and recycle the agent slot into the free pool.
                    # Previously dead targets permanently blocked their cell on agent_map.
                    world.corpse_map[tnx, tny] += 50.0
                    world.agent_map[tnx, tny]   = -1
                    agents.kill(target_id)

        # ==========================================
        # 6. REPRODUCE (Sexual Crossover)
        # ==========================================
        ready_indices = alive_indices[agents.energy[alive_indices] > 1000.0]

        for idx in ready_indices:
            px, py = agents.x[idx], agents.y[idx]

            neighbor_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            random.shuffle(neighbor_dirs)

            partner_idx = -1
            spawn_pos   = None

            for odx, ody in neighbor_dirs:
                nx, ny = (px + odx) % world.width, (py + ody) % world.height
                target = world.agent_map[nx, ny]

                if target != -1 and target != idx:
                    if agents.is_alive[target] and agents.energy[target] > 500:
                        partner_idx = target
                elif target == -1:
                    spawn_pos = (nx, ny)

            if partner_idx != -1 and spawn_pos:
                ox, oy = spawn_pos

                offspring_energy = 400.0
                agents.energy[idx]         -= 250.0
                agents.energy[partner_idx] -= 250.0

                mask = np.random.rand(*agents.brain_weights[idx].shape) > 0.5
                child_weights  = np.where(mask,
                                          agents.brain_weights[idx],
                                          agents.brain_weights[partner_idx])
                child_weights += np.random.normal(0, 0.02, size=child_weights.shape)

                c_metab = (agents.metabolism[idx] + agents.metabolism[partner_idx]) / 2 + random.gauss(0, 0.01)
                c_bite  = (agents.bite_force[idx]  + agents.bite_force[partner_idx])  / 2 + random.gauss(0, 0.05)
                c_def   = (agents.defense[idx]      + agents.defense[partner_idx])      / 2 + random.gauss(0, 0.02)

                try:
                    child_idx = agents.spawn(
                        ox, oy, offspring_energy, child_weights,
                        max(0.01, c_metab), max(0.1, c_bite),
                        500, max(0.0, min(0.9, c_def))
                    )
                    world.agent_map[ox, oy] = child_idx
                except RuntimeError:
                    # FIX #7: Pool full — break cleanly (removed dead 'pass' statement).
                    break
import numpy as np

class AgentData:
    """
    SoA (Structure of Arrays) контейнер для всех агентов.
    Заменяет список объектов Agent на плоские C-массивы.
    """
    def __init__(self, max_agents: int, n_inputs: int, n_outputs: int):
        self.max_agents = max_agents
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Маска жизни. True = агент активен, False = слот свободен/мертв
        self.is_alive = np.zeros(max_agents, dtype=np.bool_)

        # Пространственные и физические данные
        self.x = np.zeros(max_agents, dtype=np.int32)
        self.y = np.zeros(max_agents, dtype=np.int32)
        self.energy = np.zeros(max_agents, dtype=np.float32)
        self.age = np.zeros(max_agents, dtype=np.int32)
        self.defense = np.zeros(max_agents, dtype=np.float32)

        # Фенотип (характеристики, которые могут мутировать)
        self.metabolism = np.zeros(max_agents, dtype=np.float32)
        self.bite_force = np.zeros(max_agents, dtype=np.float32)
        self.max_age = np.zeros(max_agents, dtype=np.int32)
        self.cooldown = np.zeros(max_agents, dtype=np.int32)

        self.kills = np.zeros(max_agents, dtype=np.int32)
        

        # Геном: Тензор весов нейросети (MAX_AGENTS x INPUTS x OUTPUTS)
        self.brain_weights = np.zeros((max_agents, n_inputs, n_outputs), dtype=np.float32)

        # Пул свободных индексов (LIFO стек). Изначально доступны все индексы от 0 до max_agents-1
        self.free_indices = list(range(max_agents - 1, -1, -1))

    def spawn(
            self,
            x: int,
            y: int,
            energy: float,
            weights: np.ndarray,
            metabolism: float = 0.1,
            bite_force: float = 1.0,
            max_age: int = 1500,
            defense: float = 0.1
        ) -> int:
        """Регистрирует нового агента за О(1). Возвращает его индекс (ID)."""
        if not self.free_indices:
            raise RuntimeError("Maximum agent capacity reached. Increase max_agents.")

        # Берем свободный индекс
        idx = self.free_indices.pop()
        
        # Перезаписываем данные в массивах по этому индексу
        self.is_alive[idx] = True
        self.x[idx] = x
        self.y[idx] = y
        self.energy[idx] = energy
        self.metabolism[idx] = metabolism
        self.bite_force[idx] = bite_force
        self.defense[idx] = defense
        self.brain_weights[idx] = weights
        self.kills[idx] = 0

        self.age[idx] = 0           # Обнуляем возраст при рождении
        self.max_age[idx] = max_age
        
        return idx

    def kill(self, idx: int):
        """Убивает агента за О(1), возвращая индекс в пул."""
        if self.is_alive[idx]:
            self.is_alive[idx] = False
            self.free_indices.append(idx)


class World:
    """Pure data container for the simulation world state."""
    def __init__(self, width: int, height: int, max_agents: int = 10000, n_inputs: int = 5, n_outputs: int = 4):
        self.width = width
        self.height = height
        
        # Energy and spatial tracking
        self.nutrients = np.zeros((width, height), dtype=np.float32)
        self.vegetation = np.zeros((width, height), dtype=np.float32)
        self.corpse_map = np.zeros((width, height), dtype=np.float32)
        self.agent_map = np.full((width, height), -1, dtype=np.int32)
        self.scent_map = np.zeros((width, height), dtype=np.float32)
        self.pheromone_map = np.zeros((width, height), dtype=np.float32)
        
        # Global state
        self.leak_buffer: float = 0.0
        self.sun_pos = np.array([width / 2.0, height / 2.0], dtype=np.float64)
        self.sun_angle: float = 0.0
        self.sun_params: dict = {
            'angular_velocity': 0.05,
            'amplitude_x': width * 0.4,
            'amplitude_y': height * 0.4,
            'base_drip': 5.0,
            'bonus_drip_max': 20.0
        }
        
        # Entity storage теперь SoA
        self.agents = AgentData(max_agents, n_inputs, n_outputs)

        # NOTE: seed_life() is intentionally NOT called here.
        # The caller (main.py) seeds explicitly with the desired parameters.

    def seed_life(self, vegetation_chance: float = 0.08, nutrient_range: tuple[float, float] = (0.1, 0.5)):
        # Generate random mask for vegetation
        mask = np.random.rand(self.width, self.height) < vegetation_chance
        
        # Populate vegetation in masked cells
        self.vegetation[mask] = np.random.uniform(nutrient_range[0], nutrient_range[1], size=np.count_nonzero(mask)).astype(np.float32)
        self.nutrients[mask] = 0.5
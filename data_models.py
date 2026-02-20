from dataclasses import dataclass, field
import numpy as np

@dataclass
class Agent:
    """Pure data container for an individual agent."""
    id: int
    x: int
    y: int
    energy: float
    genome: dict = field(default_factory=lambda: {'bite_force': 1.0, 'speed': 1.0, 'metabolism': 0.1})

class World:
    """Pure data container for the simulation world state."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Energy and spatial tracking
        self.nutrients = np.zeros((width, height), dtype=np.float32)
        self.vegetation = np.zeros((width, height), dtype=np.float32)
        self.corpse_map = np.zeros((width, height), dtype=np.float32)
        self.agent_map = np.full((width, height), -1, dtype=np.int32)
        self.scent_map = np.zeros((width, height), dtype=np.float32)
        
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
        
        # Entity storage
        self.agents: list[Agent] = []

        # Initial seeding
        self.seed_life()

    def seed_life(self, vegetation_chance: float = 0.08, nutrient_range: tuple[float, float] = (0.1, 0.5)):
        """
        Noisy Start: Vectorized initialization of vegetation and nutrients.
        """
        # Generate random mask for vegetation
        mask = np.random.rand(self.width, self.height) < vegetation_chance
        
        # Populate vegetation in masked cells
        self.vegetation[mask] = np.random.uniform(nutrient_range[0], nutrient_range[1], size=np.count_nonzero(mask)).astype(np.float32)
        
        # Ensure initial soil quality in masked cells
        self.nutrients[mask] = 0.5

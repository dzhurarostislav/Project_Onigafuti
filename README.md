# Project Onigafuti

Onigafuti is a simplified evolutionary simulator. It focuses on the core mechanics of natural selection, mutation, and adaptation within a discrete 2D substrate. Agents are driven by evolved neural networks, competing for vegetation, hunting each other, and reproducing via sexual crossover. A digital laboratory for raw evolutionary logic.

## Tech Stack

- **Python 3.10+**
- **NumPy** — vectorized grid math (SoA data layout)
- **Pygame** — 2D visualization with sidebar analytics
- **Ruff** — linting

## Project Structure

```
Project_Onigafuti/
├── main.py               # Entry point, simulation loop, initial population seeding
├── data_models.py        # AgentData (SoA) and World data containers
├── environment.py        # BioSystem, EnergySystem, ScentSystem
├── entities.py           # AgentSystem (neural brain, predation, reproduction)
├── render.py             # RenderSystem (Pygame output + sidebar graphs)
├── requirements.txt
├── manifest.md           # Architecture manifest
├── global_constraints.md # Allowed/forbidden libraries
├── global_dict.md        # World state contract
└── README.md
```

## Architecture

The project follows **ECS-lite** and **data-oriented design**. All entity data is stored in a flat **Structure of Arrays (SoA)** layout for cache-friendly vectorized processing.

| Layer | Role |
|-------|------|
| **Data** (`data_models.py`) | `World` — grid state; `AgentData` — SoA container for all agents |
| **Systems** | Stateless processors that read/write `World` |
| **Rendering** | Read-only; never mutates simulation state |

### AgentData (SoA Container)

Replaces per-agent objects with flat NumPy arrays indexed by agent slot ID. Spawn/kill operations run in O(1) via a LIFO free-index pool.

| Array | Type | Description |
|-------|------|-------------|
| `is_alive` | bool (N) | Active/dead mask |
| `x`, `y` | int32 (N) | Grid coordinates |
| `energy` | float32 (N) | Current energy level |
| `age` | int32 (N) | Ticks alive |
| `defense` | float32 (N) | Damage reduction (0.0–0.9) |
| `metabolism` | float32 (N) | Energy cost per tick |
| `bite_force` | float32 (N) | Attack/eating power |
| `max_age` | int32 (N) | Lifespan limit |
| `cooldown` | int32 (N) | Post-kill digestion timer |
| `kills` | int32 (N) | Lifetime kill count |
| `brain_weights` | float32 (N×I×O) | Neural network genome |

### World State (`World`)

| Field | Type | Description |
|-------|------|-------------|
| `nutrients` | float32 (W×H) | Soil energy |
| `vegetation` | float32 (W×H) | Edible plant energy |
| `corpse_map` | float32 (W×H) | Decaying corpse energy |
| `agent_map` | int32 (W×H) | Agent slot IDs (-1 = empty) |
| `scent_map` | float32 (W×H) | Food scent gradient |
| `pheromone_map` | float32 (W×H) | Agent trail pheromones |
| `leak_buffer` | float | Energy lost at boundaries |
| `sun_pos` | float[2] | Center of energy drip |
| `sun_angle` | float | Current pendulum angle |
| `sun_params` | dict | Angular velocity, amplitude, drip rates |
| `agents` | AgentData | SoA entity storage |

### Simulation Loop (per tick)

1. **BioSystem** — Nutrient diffusion, vegetation growth, corpse decay
2. **EnergySystem** — Sun pendulum movement (mean-reverting Brownian) and energy drip
3. **ScentSystem** — Food scent emission/diffusion + pheromone trail diffusion/decay
4. **AgentSystem** — Neural thinking → Metabolism → Death → Eat → Move/Predation → Reproduce

### Agent Neural Brain

Each agent carries a weight matrix (`n_inputs × n_outputs`) as its genome. Decisions are computed via `np.einsum` every tick.

**Inputs (10):**

| # | Signal |
|---|--------|
| 0 | Bias (constant 1.0) |
| 1 | Energy (normalized) |
| 2 | Food scent at current cell |
| 3 | Internal clock (`sin(age × 0.1)`) |
| 4–7 | Vision: agent detected up/down/left/right |
| 8 | Pheromone intensity at current cell |
| 9 | Own defense value |

**Outputs (5):** Stay, Move Up, Move Down, Move Left, Move Right

### Predation & Combat

When an agent moves into an occupied cell, it attacks the occupant. Damage scales with `bite_force` and is reduced by target's `defense`. Killing an opponent grants 90% of its energy, increments `kills`, increases `bite_force` (capped at 20), and triggers a 10-tick cooldown during which the predator cannot move or attack.

### Reproduction (Sexual Crossover)

Agents with energy > 1000 can reproduce with an adjacent partner (energy > 500). Offspring receive:
- Brain weights via random mask crossover from both parents + Gaussian mutation
- Averaged phenotype traits (`metabolism`, `bite_force`, `defense`) with small noise
- 400 energy; each parent pays 250

## Running the Simulation

```bash
# Create and activate virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Display

### World View (left panel)

| Color Channel | Encodes |
|---------------|---------|
| **Green** | Vegetation density |
| **Blue** | Corpse energy |
| **Red intensity** (agents) | `bite_force` (brighter = stronger predator) |
| **Blue intensity** (agents) | `defense` (brighter = tankier) |

### Sidebar (right panel)

Three real-time line graphs tracking the last 200 ticks:
- **Population** (white) — total alive agents
- **Predators** (red) — agents with `bite_force > 10.0`
- **Tanks** (blue) — agents with `defense > 0.5`

### Console Reports

Every 500 ticks the simulation prints an evolution report highlighting the oldest agent and the top predator, including a genetic analysis of the alpha's attack reflexes (neural weights for neighbor-vision → movement outputs).

## Constraints (see `global_constraints.md`)

- Allowed: `numpy`, `pygame`, `random`, `math`
- Forbidden: `scipy` (unless requested), `pandas`, `multiprocessing`
- Single-threaded; no logging in the main loop

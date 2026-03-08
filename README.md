# Project Onigafuti

Onigafuti is a simplified evolutionary simulator. It focuses on the core mechanics of natural selection, mutation, and adaptation within a discrete 2D substrate. By eliminating visual noise, the project analyzes emergent behaviors and genetic survival in resource-limited environments. A digital laboratory for raw evolutionary logic.

## Tech Stack

- **Python 3.10+**
- **NumPy** — vectorized grid math
- **Pygame** — 2D visualization
- **Ruff** — linting

## Project Structure

```
Project_Onigafuti/
├── main.py           # Entry point, simulation loop
├── data_models.py    # World and Agent dataclasses
├── environment.py    # BioSystem, EnergySystem, ScentSystem
├── entities.py       # AgentSystem (lifecycle, movement, feeding)
├── render.py         # RenderSystem (Pygame output)
├── requirements.txt
├── manifest.md       # Architecture document
├── global_constraints.md
├── global_dict.md
└── README.md
```

## Architecture

The project follows **ECS-lite** and **data-oriented design**:

| Layer | Role |
|-------|------|
| **Data** (`data_models.py`) | `World` — grid state; `Agent` — individual entity with genome |
| **Systems** | Stateless processors that read/write `World` and `agents` |
| **Rendering** | Read-only; never mutates simulation state |

### World State (`World`)

| Field | Type | Description |
|-------|------|-------------|
| `nutrients` | float32 (W×H) | Soil energy |
| `vegetation` | float32 (W×H) | Edible energy |
| `corpse_map` | float32 (W×H) | Decaying corpses |
| `agent_map` | int32 (W×H) | Agent IDs (-1 = empty) |
| `scent_map` | float32 (W×H) | Navigation gradient |
| `leak_buffer` | float | Global energy from boundary losses |
| `sun_pos` | float[2] | Center of energy drip |
| `agents` | list[Agent] | Living entities |

### Simulation Loop (per tick)

1. **BioSystem** — Nutrient diffusion, vegetation growth, corpse decay  
2. **EnergySystem** — Sun pendulum movement and energy drip into the grid  
3. **AgentSystem** — Metabolism → death → eat → move → reproduce  
4. **RenderSystem** — Draw world and agents  

### Agent Genome

- `bite_force` — Eating efficiency
- `metabolism` — Energy spent per tick
- `randomness` — Chance of random movement vs scent-following

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

### Display Legend

- **Green** — Vegetation (food)
- **Blue** — Corpses (decaying nutrients)
- **Red** — Agents

## Constraints (see `global_constraints.md`)

- Allowed: `numpy`, `pygame`, `random`, `math`
- Forbidden: `scipy` (unless requested), `pandas`, `multiprocessing`
- Single-threaded; no logging in the main loop

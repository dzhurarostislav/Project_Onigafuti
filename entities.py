import random
import numpy as np
from data_models import World, Agent

class AgentSystem:
    """Handles logic for agent life cycles, movement, and feeding."""
    
    def update(self, world: World):
        """
        Updates each agent in the world state.
        Execution Order: SHUFFLE -> METABOLISM -> DEATH -> EAT -> MOVE
        """
        # 1. SHUFFLE
        random.shuffle(world.agents)
        
        dead_agent_ids = set()
        new_agents = []
        
        # Track max ID for new offspring
        current_max_id = max([a.id for a in world.agents] + [0])
        
        for agent in world.agents:
            # 2. METABOLISM
            metabolism = agent.genome.get('metabolism', 0.1)
            agent.energy -= metabolism
            
            # 3. DEATH (Starvation Only)
            if agent.energy <= 0:
                # Deposit body energy to corpse_map
                world.corpse_map[agent.x, agent.y] += 50.0
                # Clear spot immediately
                if world.agent_map[agent.x, agent.y] == agent.id:
                    world.agent_map[agent.x, agent.y] = -1
                dead_agent_ids.add(agent.id)
                continue
            
            # 4. EAT (Eat at current location before moving)
            veg_at_feet = world.vegetation[agent.x, agent.y]
            if veg_at_feet > 0.05:
                # Consume according to bite_force
                bite_force = agent.genome.get('bite_force', 1.0)
                amount_to_eat = min(veg_at_feet, bite_force)
                
                world.vegetation[agent.x, agent.y] -= amount_to_eat
                # Nutrient efficiency: energy gain is 10x the amount consumed
                agent.energy += amount_to_eat * 10.0
            
            # 5. MOVE (Calculate next position)
            moved = False
            
            # Consistency Guard: Ensure agent is registered correctly in the map
            if world.agent_map[agent.x, agent.y] != agent.id:
                if world.agent_map[agent.x, agent.y] == -1:
                    world.agent_map[agent.x, agent.y] = agent.id
            
            # randomness comes from genome or defaults to 15%
            random_chance = agent.genome.get('randomness', 0.15)
            
            if random.random() < random_chance:
                # Random exploration
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
                new_x = (agent.x + dx) % world.width
                new_y = (agent.y + dy) % world.height
                
                if world.agent_map[new_x, new_y] == -1:
                    if world.agent_map[agent.x, agent.y] == agent.id:
                        world.agent_map[agent.x, agent.y] = -1
                    agent.x, agent.y = new_x, new_y
                    world.agent_map[agent.x, agent.y] = agent.id
                    moved = True
            
            if not moved:
                # Scent-based movement
                neighbors = []
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx = (agent.x + dx) % world.width
                        ny = (agent.y + dy) % world.height
                        
                        if world.agent_map[nx, ny] == -1 or (nx == agent.x and ny == agent.y):
                            s_val = world.scent_map[nx, ny]
                            neighbors.append(((nx, ny), s_val))
                
                if neighbors:
                    max_s = max(n[1] for n in neighbors)
                    if max_s > 0:
                        candidates = [n[0] for n in neighbors if n[1] == max_s]
                        selected = random.choice(candidates)
                        
                        if selected[0] != agent.x or selected[1] != agent.y:
                            if world.agent_map[agent.x, agent.y] == agent.id:
                                world.agent_map[agent.x, agent.y] = -1
                            agent.x, agent.y = selected[0], selected[1]
                            world.agent_map[agent.x, agent.y] = agent.id
                    else:
                        # Fallback for zero scent
                        dx = random.randint(-1, 1)
                        dy = random.randint(-1, 1)
                        new_x = (agent.x + dx) % world.width
                        new_y = (agent.y + dy) % world.height
                        if world.agent_map[new_x, new_y] == -1:
                            if world.agent_map[agent.x, agent.y] == agent.id:
                                world.agent_map[agent.x, agent.y] = -1
                            agent.x, agent.y = new_x, new_y
                            world.agent_map[agent.x, agent.y] = agent.id

            # 6. REPRODUCE
            if agent.energy > 1000.0:
                # Attempt to find a free adjacent spot for offspring
                dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                random.shuffle(dirs)
                
                for odx, ody in dirs:
                    ox = (agent.x + odx) % world.width
                    oy = (agent.y + ody) % world.height
                    
                    if world.agent_map[ox, oy] == -1:
                        # Success! Create offspring
                        offspring_energy = agent.energy * 0.5
                        agent.energy -= offspring_energy
                        
                        # Mutate genome
                        new_genome = agent.genome.copy()
                        for key in new_genome:
                            if isinstance(new_genome[key], (int, float)):
                                new_genome[key] += random.gauss(0, 0.05)
                        
                        # Clamp values
                        new_genome['speed'] = max(0.1, new_genome.get('speed', 1.0))
                        new_genome['metabolism'] = max(0.01, new_genome.get('metabolism', 0.1))
                        new_genome['bite_force'] = max(0.1, new_genome.get('bite_force', 1.0))
                        new_genome['randomness'] = max(0.0, min(1.0, new_genome.get('randomness', 0.15)))
                        
                        current_max_id += 1
                        offspring = Agent(
                            id=current_max_id,
                            x=ox,
                            y=oy,
                            energy=offspring_energy,
                            genome=new_genome
                        )
                        
                        new_agents.append(offspring)
                        world.agent_map[ox, oy] = offspring.id
                        break
                
        # Clean up dead agents
        if dead_agent_ids:
            world.agents = [a for a in world.agents if a.id not in dead_agent_ids]
            
        # Add new agents to the world
        world.agents.extend(new_agents)

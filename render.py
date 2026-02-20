import pygame
import numpy as np
from data_models import World

class RenderSystem:
    """Read-only system for visualising the simulation state using Pygame."""
    
    def __init__(self, cell_size: int = 5):
        """
        Initializes the Pygame context.
        
        Args:
            cell_size: The size of each grid cell in pixels.
        """
        pygame.init()
        self.cell_size = cell_size
        self.screen = None
        self.width = 0
        self.height = 0

    def _init_display(self, world: World):
        """Initializes the display based on the world dimensions."""
        self.width = world.width * self.cell_size
        self.height = world.height * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Onigafuti Simulation")

    def draw(self, world: World):
        """
        Renders the current world state.
        
        Args:
            world: The World object to render.
        """
        if self.screen is None or world.width * self.cell_size != self.width or world.height * self.cell_size != self.height:
            self._init_display(world)

        # 1. Background Rendering
        # Vegetation -> Green channel
        # Corpses -> Blue channel
        # Nutrients -> Invisible
        
        # Create an RGB array (width, height, 3) initialized to zeros (Black base)
        bg_rgb = np.zeros((world.width, world.height, 3), dtype=np.uint8)
        
        # Green channel = Vegetation
        bg_rgb[:, :, 1] = np.clip(world.vegetation * 255, 0, 255).astype(np.uint8)
        
        # Blue channel = Corpses
        bg_rgb[:, :, 2] = np.clip(world.corpse_map * 255, 0, 255).astype(np.uint8)
        
        # Create surface from array
        bg_surface = pygame.surfarray.make_surface(bg_rgb)
        
        # Scale to match display if cell_size > 1
        if self.cell_size > 1:
            bg_surface = pygame.transform.scale(bg_surface, (self.width, self.height))
            
        self.screen.blit(bg_surface, (0, 0))

        # 2. Agent Rendering
        for agent in world.agents:
            rect = pygame.Rect(
                agent.x * self.cell_size,
                agent.y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (255, 0, 0), rect)

        pygame.display.flip()

    def quit(self):
        """Shuts down Pygame."""
        pygame.quit()

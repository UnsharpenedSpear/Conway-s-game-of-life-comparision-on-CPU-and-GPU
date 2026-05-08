#!/usr/bin/env python3
"""
interactive_gol.py — Interactive Conway's Game of Life Viewer

Features:
  - Visual grid display with live cell rendering
  - Pattern selection (glider, blinker, block, toad, beacon, pentomino, random, empty)
  - Generation counter with display
  - Adjustable tick speed (via keyboard or UI)
  - Real-time power/CPU usage graph
  - Pause/Resume functionality
  - Clear grid option

Controls:
  SPACE     → Play/Pause
  R         → Random pattern
  C         → Clear grid
  G         → Add glider
  B         → Add blinker
  K         → Add block
  T         → Add toad
  +/-       → Speed up/Slow down
  ESC       → Quit
"""

import sys
import os
import time
import psutil
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from threading import Thread
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GRID_WIDTH = 256
GRID_HEIGHT = 256
CELL_SIZE = 3  # pixels per cell
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + 400  # Extra space for graphs and UI
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 50

ALIVE_COLOR = (0, 255, 0)    # green
DEAD_COLOR = (20, 20, 20)    # dark gray
GRID_COLOR = (40, 40, 40)    # slightly lighter gray
UI_COLOR = (100, 100, 100)
TEXT_COLOR = (255, 255, 255)

DEFAULT_TICK_RATE = 10  # ticks per second (max)
MAX_HISTORY = 600  # store 600 samples (10 seconds at 60 fps)


class Pattern:
    """Static patterns for Game of Life initialization."""
    
    @staticmethod
    def glider(grid, x, y):
        """Add a glider pattern at (x, y)."""
        pattern = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for dx, dy in pattern:
            nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
            grid[ny, nx] = 1
    
    @staticmethod
    def blinker(grid, x, y):
        """Add a blinker pattern at (x, y)."""
        for i in range(3):
            nx, ny = (x + i) % GRID_WIDTH, y % GRID_HEIGHT
            grid[ny, nx] = 1
    
    @staticmethod
    def block(grid, x, y):
        """Add a 2x2 block pattern at (x, y)."""
        for dx in range(2):
            for dy in range(2):
                nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
                grid[ny, nx] = 1
    
    @staticmethod
    def toad(grid, x, y):
        """Add a toad pattern at (x, y)."""
        pattern = [(0, 1), (1, 1), (2, 1), (1, 0), (2, 0), (3, 0)]
        for dx, dy in pattern:
            nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
            grid[ny, nx] = 1
    
    @staticmethod
    def beacon(grid, x, y):
        """Add a beacon pattern at (x, y)."""
        pattern = [(0, 0), (1, 0), (0, 1), (2, 2), (1, 3), (2, 3)]
        for dx, dy in pattern:
            nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
            grid[ny, nx] = 1
    
    @staticmethod
    def pentomino(grid, x, y):
        """Add a pentomino (F pentomino) pattern at (x, y)."""
        pattern = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        for dx, dy in pattern:
            nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
            grid[ny, nx] = 1
    
    @staticmethod
    def random(grid, density=0.3):
        """Fill grid with random pattern."""
        grid[:] = np.random.binomial(1, density, size=(GRID_HEIGHT, GRID_WIDTH))


class GameOfLife:
    """Conway's Game of Life simulator."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.generation = 0
        self.paused = True
    
    def step(self):
        """Execute one generation."""
        if self.paused:
            return
        
        # Count live neighbours for each cell using convolution
        neighbours = np.zeros_like(self.grid)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbours += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)
        
        # Apply rules: alive if (alive and 2-3 neighbours) or (dead and 3 neighbours)
        self.grid = np.uint8(
            (self.grid & ((neighbours == 2) | (neighbours == 3))) |
            (~self.grid & (neighbours == 3))
        )
        self.generation += 1
    
    def clear(self):
        """Clear the grid."""
        self.grid.fill(0)
        self.generation = 0
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
    
    def get_population(self):
        """Get number of alive cells."""
        return int(np.sum(self.grid))


class UsageMonitor:
    """Monitor CPU and memory usage over time."""
    
    def __init__(self, max_history=MAX_HISTORY):
        self.max_history = max_history
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.start_time = time.time()
        self.process = psutil.Process()
    
    def update(self):
        """Record current usage."""
        current_time = time.time() - self.start_time
        cpu_percent = self.process.cpu_percent(interval=0.01)
        mem_percent = self.process.memory_percent()
        
        self.timestamps.append(current_time)
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(mem_percent)
    
    def get_graph_texture(self, width=350, height=120):
        """Generate a matplotlib graph and return as pygame texture."""
        try:
            fig = Figure(figsize=(width/100, height/100), dpi=100)
            ax = fig.add_subplot(111)
            
            if len(self.timestamps) > 1:
                ax.plot(list(self.timestamps), list(self.cpu_history), 
                       'c-', linewidth=1.5, label='CPU %')
                ax.plot(list(self.timestamps), list(self.memory_history), 
                       'y-', linewidth=1.5, label='Memory %')
                ax.set_ylim([0, 100])
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Usage %', fontsize=8)
                ax.legend(fontsize=7, loc='upper left')
                ax.grid(True, alpha=0.3)
            
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white', labelsize=7)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            
            size = canvas.get_width_height()
            return pygame.image.fromstring(raw_data, size, "RGB")
        except Exception as e:
            print(f"Warning: Could not generate usage graph: {e}", file=sys.stderr)
            return None


class GameOfLifeApp:
    """Main application for interactive Game of Life."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Conway's Game of Life - Interactive Viewer")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 16)
        self.font_large = pygame.font.Font(None, 24)
        
        self.gol = GameOfLife(GRID_WIDTH, GRID_HEIGHT)
        self.monitor = UsageMonitor()
        self.running = True
        self.tick_rate = DEFAULT_TICK_RATE  # ticks per second
        self.next_step_time = 0
        self.graph_texture = None
        self.last_graph_update = 0
        
        # Initialize with a random pattern
        self.gol.clear()
        Pattern.random(self.gol.grid, density=0.25)
        self.gol.paused = True
    
    def handle_input(self):
        """Handle keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.gol.toggle_pause()
                elif event.key == pygame.K_r:
                    self.gol.clear()
                    Pattern.random(self.gol.grid, density=0.3)
                elif event.key == pygame.K_c:
                    self.gol.clear()
                elif event.key == pygame.K_g:
                    self.gol.paused = True
                    x, y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                    Pattern.glider(self.gol.grid, x, y)
                elif event.key == pygame.K_b:
                    self.gol.paused = True
                    x, y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                    Pattern.blinker(self.gol.grid, x, y)
                elif event.key == pygame.K_k:
                    self.gol.paused = True
                    x, y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                    Pattern.block(self.gol.grid, x, y)
                elif event.key == pygame.K_t:
                    self.gol.paused = True
                    x, y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                    Pattern.toad(self.gol.grid, x, y)
                elif event.key == pygame.K_p:
                    self.gol.paused = True
                    x, y = GRID_WIDTH // 2, GRID_HEIGHT // 2
                    Pattern.pentomino(self.gol.grid, x, y)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.tick_rate = min(60, self.tick_rate + 1)
                elif event.key == pygame.K_MINUS:
                    self.tick_rate = max(1, self.tick_rate - 1)
    
    def update(self):
        """Update simulation and monitoring."""
        # Update game state
        current_time = time.time()
        if current_time >= self.next_step_time:
            self.gol.step()
            self.next_step_time = current_time + (1.0 / self.tick_rate)
        
        # Update usage monitor
        self.monitor.update()
        
        # Update graph every 0.5 seconds
        if current_time - self.last_graph_update > 0.5:
            self.graph_texture = self.monitor.get_graph_texture()
            self.last_graph_update = current_time
    
    def draw(self):
        """Draw the game state and UI."""
        self.screen.fill((10, 10, 10))
        
        # Draw grid background
        grid_surface = pygame.Surface((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
        grid_surface.fill(DEAD_COLOR)
        
        # Draw alive cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.gol.grid[y, x]:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(grid_surface, ALIVE_COLOR, rect)
        
        self.screen.blit(grid_surface, (0, 0))
        
        # Draw grid lines
        for x in range(0, GRID_WIDTH * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, GRID_HEIGHT * CELL_SIZE), 1)
        for y in range(0, GRID_HEIGHT * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (GRID_WIDTH * CELL_SIZE, y), 1)
        
        # Draw UI panel on the right
        ui_x = GRID_WIDTH * CELL_SIZE + 10
        ui_y = 10
        
        # Draw stats
        status = "PAUSED" if self.gol.paused else "RUNNING"
        status_color = (255, 100, 100) if self.gol.paused else (100, 255, 100)
        
        stats = [
            f"Generation: {self.gol.generation}",
            f"Population: {self.gol.get_population()}",
            f"Speed: {self.tick_rate} ticks/s",
            f"Status: {status}",
            "",
            "--- CONTROLS ---",
            "SPACE: Play/Pause",
            "R: Random",
            "C: Clear",
            "G: Glider",
            "B: Blinker",
            "K: Block",
            "T: Toad",
            "P: Pentomino",
            "+/-: Speed",
            "ESC: Quit",
        ]
        
        for i, text in enumerate(stats):
            if text == "Status: " + status:
                surf = self.font_small.render(text, True, status_color)
            elif text == "":
                continue
            else:
                surf = self.font_small.render(text, True, TEXT_COLOR)
            self.screen.blit(surf, (ui_x, ui_y + i * 18))
        
        # Draw usage graph
        if self.graph_texture:
            graph_y = ui_y + len(stats) * 18 + 20
            self.screen.blit(self.graph_texture, (ui_x - 5, graph_y))
        
        # Draw bottom info bar
        info_text = f"FPS: {self.clock.get_fps():.1f} | Grid: {GRID_WIDTH}x{GRID_HEIGHT}"
        surf = self.font_small.render(info_text, True, (150, 150, 150))
        self.screen.blit(surf, (10, GRID_HEIGHT * CELL_SIZE + 15))
        
        pygame.display.flip()
    
    def run(self):
        """Main application loop."""
        while self.running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)  # 60 FPS display refresh
        
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        app = GameOfLifeApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

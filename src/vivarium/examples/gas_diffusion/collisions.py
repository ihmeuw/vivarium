from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class Collisions(Component):
    ##############
    # Properties #
    ##############
    configuration_defaults = {
        "gas_collisions": {
            "enabled": True,
            "restitution": 1.0,  # Perfect elastic collisions
        }
    }

    columns_required = ["x", "y", "vx", "vy", "radius"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.gas_collisions
        self.neighbors = builder.value.get_value("neighbors")

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        if not self.config.enabled:
            return
            
        pop = self.population_view.get(event.index)
        neighbors = self.neighbors(event.index)
        
        # Process collisions
        updated_pop = self._handle_collisions(pop, neighbors)
        self.population_view.update(updated_pop)

    ##################
    # Helper methods #
    ##################

    def _handle_collisions(self, pop: pd.DataFrame, neighbors: pd.Series) -> pd.DataFrame:
        """Handle elastic collisions between particles."""
        processed_pairs = set()
        
        for idx, neighbor_list in neighbors.items():
            if not neighbor_list:
                continue
                
            particle = pop.loc[idx]
            
            for neighbor_idx in neighbor_list:
                # Skip if we've already processed this pair
                pair = tuple(sorted([idx, neighbor_idx]))
                if pair in processed_pairs:
                    continue
                    
                neighbor = pop.loc[neighbor_idx]
                
                # Calculate distance between particles
                dx = neighbor.x - particle.x
                dy = neighbor.y - particle.y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check if particles are actually overlapping/colliding
                min_distance = particle.radius + neighbor.radius
                if distance <= min_distance and distance > 0:
                    # Perform elastic collision
                    self._elastic_collision(pop, idx, neighbor_idx, dx, dy, distance)
                    processed_pairs.add(pair)
        
        return pop

    def _elastic_collision(self, pop: pd.DataFrame, idx1: int, idx2: int, 
                          dx: float, dy: float, distance: float) -> None:
        """Perform elastic collision between two particles."""
        # Get particle data
        p1 = pop.loc[idx1]
        p2 = pop.loc[idx2]
        
        # Assume equal mass for simplicity (can be made configurable)
        m1 = m2 = 1.0
        
        # Unit vector from particle 1 to particle 2
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        dvx = p2.vx - p1.vx
        dvy = p2.vy - p1.vy
        
        # Relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Do not resolve if velocities are separating
        if dvn > 0:
            return
            
        # Collision impulse
        impulse = 2 * dvn / (m1 + m2) * self.config.restitution
        
        # Update velocities
        pop.loc[idx1, "vx"] += impulse * m2 * nx
        pop.loc[idx1, "vy"] += impulse * m2 * ny
        pop.loc[idx2, "vx"] -= impulse * m1 * nx  
        pop.loc[idx2, "vy"] -= impulse * m1 * ny
        
        # Separate overlapping particles
        overlap = (p1.radius + p2.radius) - distance
        if overlap > 0:
            separation = overlap / 2
            pop.loc[idx1, "x"] -= separation * nx
            pop.loc[idx1, "y"] -= separation * ny
            pop.loc[idx2, "x"] += separation * nx
            pop.loc[idx2, "y"] += separation * ny

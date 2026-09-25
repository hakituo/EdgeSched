import logging
import time
from typing import Dict, Optional, Any
from mvp_core.services.nvram_manager import get_nvram_manager
from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine

logger = logging.getLogger(__name__)

class BiologicalSystem:
    """
    Bridge to C++ BiologicalSystem.
    Syncs C++ state to NVRAM (Software-Defined NVRAM).
    The actual biological logic (decay, circadian rhythm, etc.) resides in C++.
    """
    def __init__(self):
        self.nvram = get_nvram_manager()
        self.cpp_bio = None
        self._last_connect_attempt = 0
        self._connect_retry_interval = 5.0

    def _ensure_cpp_connection(self):
        """Try to connect to the C++ Biological System instance"""
        if self.cpp_bio:
            return

        now = time.time()
        if now - self._last_connect_attempt < self._connect_retry_interval:
            return

        self._last_connect_attempt = now
        
        # Access the scheduler from the engine singleton
        if cpp_scheduler_engine.enabled and cpp_scheduler_engine.scheduler:
             try:
                 # The method name in binding is getBiologicalSystem
                 self.cpp_bio = cpp_scheduler_engine.scheduler.getBiologicalSystem()
                 if self.cpp_bio:
                     logger.info("Connected to C++ Biological System")
                     # Initial sync
                     self.update()
             except Exception as e:
                 logger.debug(f"Waiting for C++ Biological System: {e}")

    def update(self):
        """
        Sync state from C++ to NVRAM.
        C++ handles the actual decay/update logic internally via NativeExecutor.
        This method is called periodically by Python TaskScheduler to persist state.
        """
        self._ensure_cpp_connection()
        
        if self.cpp_bio:
            try:
                # Read from C++
                neuro = self.cpp_bio.getNeurotransmitters()
                energy = self.cpp_bio.getEnergy()
                
                # Mood calculation (simplification for visualization)
                # In C++, mood might be derived or just serotonin/dopamine mix
                mood = (neuro.serotonin + neuro.dopamine) / 2.0
                
                # Write to NVRAM (Software-Defined NVRAM persistence)
                self.nvram.update_bio_state(
                    dopamine=neuro.dopamine,
                    energy=energy,
                    mood=mood
                )
            except Exception as e:
                logger.error(f"Error syncing Bio state from C++: {e}")
                # Invalidate connection on error
                self.cpp_bio = None

    def get_state(self) -> Dict[str, float]:
        """
        Get the current state from NVRAM (as the single source of truth for external readers)
        """
        return self.nvram.get_bio_state()

    def stimulate(self, dopamine_hit: float = 0.0, energy_boost: float = 0.0, mood_lift: float = 0.0):
        """
        Apply stimulus to the Biological System (via C++)
        """
        self._ensure_cpp_connection()
        
        if self.cpp_bio:
            try:
                if dopamine_hit != 0:
                    self.cpp_bio.adjustNeurotransmitter("dopamine", dopamine_hit)
                if mood_lift != 0:
                    # Map mood lift to serotonin/dopamine
                    self.cpp_bio.adjustNeurotransmitter("serotonin", mood_lift * 0.7)
                    self.cpp_bio.adjustNeurotransmitter("dopamine", mood_lift * 0.3)
                if energy_boost != 0:
                    if energy_boost > 0:
                        self.cpp_bio.recoverEnergy(energy_boost)
                    else:
                        self.cpp_bio.consumeEnergy(-energy_boost)
                
                # Trigger immediate sync so NVRAM reflects the change
                self.update()
            except Exception as e:
                logger.error(f"Failed to stimulate Bio System: {e}")

# Global Singleton
_bio_system = None
def get_biological_system():
    global _bio_system
    if _bio_system is None:
        _bio_system = BiologicalSystem()
    return _bio_system

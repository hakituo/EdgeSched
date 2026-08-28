import os
import sys
import time
import struct
import unittest
import multiprocessing

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mvp_core.services.nvram_manager import NVRAMManager

def writer_process(file_path):
    print(f"[Writer] Opening NVRAM at {file_path}")
    nvram = NVRAMManager(file_path)
    print("[Writer] Updating Bio State")
    nvram.update_bio_state(0.8, 0.9, 0.5)  # Dopamine, Energy, Mood
    
    print("[Writer] Registering Task")
    nvram.register_task("task_test_123", 1, 2) # ID, Priority, Status=Running
    
    # Wait a bit to ensure Reader sees it
    time.sleep(1)
    
    print("[Writer] Updating Task to Completed")
    nvram.update_task_status("task_test_123", 3) # Completed
    nvram.close()

def reader_process(file_path):
    # Give writer a moment
    time.sleep(0.5)
    print(f"[Reader] Opening NVRAM at {file_path}")
    nvram = NVRAMManager(file_path)
    
    bio = nvram.get_bio_state()
    print(f"[Reader] Bio State: {bio}")
    
    tasks = nvram.get_active_tasks()
    print(f"[Reader] Initial Tasks: {tasks}")
    
    # Wait for update
    time.sleep(1.0)
    tasks_after = nvram.get_active_tasks()
    print(f"[Reader] Final Tasks: {tasks_after}")
    nvram.close()

class TestNVRAM(unittest.TestCase):
    def test_simulation(self):
        # Use a temporary file
        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_nvram.dat"))
        if os.path.exists(test_file):
            os.remove(test_file)
            
        p1 = multiprocessing.Process(target=writer_process, args=(test_file,))
        p2 = multiprocessing.Process(target=reader_process, args=(test_file,))
        
        p1.start()
        p2.start()
        
        p1.join()
        p2.join()
        
        # Verify final state in main process
        nvram = NVRAMManager(test_file)
        tasks = nvram.get_active_tasks()
        print(f"[Main] Final Check: {tasks}")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['task_id'], "task_test_123")
        self.assertEqual(tasks[0]['status'], 3) # Completed
        nvram.close()

if __name__ == "__main__":
    unittest.main()

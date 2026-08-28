import os
import sys
import time
import unittest
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mvp_core.services.state_backplane import StateBackplane


def writer_process(file_path):
    print(f"[Writer] Opening Backplane at {file_path}")
    backplane = StateBackplane(file_path)

    print("[Writer] Registering Task")
    backplane.register_task("task_test_123", 1, 2)

    time.sleep(1)

    print("[Writer] Updating Task to Completed")
    backplane.update_task_status("task_test_123", 3)
    backplane.close()


def reader_process(file_path):
    time.sleep(0.5)
    print(f"[Reader] Opening Backplane at {file_path}")
    backplane = StateBackplane(file_path)

    tasks = backplane.get_active_tasks()
    print(f"[Reader] Initial Tasks: {tasks}")

    time.sleep(1.0)
    tasks_after = backplane.get_active_tasks()
    print(f"[Reader] Final Tasks: {tasks_after}")
    backplane.close()


class TestStateBackplane(unittest.TestCase):
    def test_simulation(self):
        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/test_state_backplane.dat"))
        if os.path.exists(test_file):
            os.remove(test_file)

        p1 = multiprocessing.Process(target=writer_process, args=(test_file,))
        p2 = multiprocessing.Process(target=reader_process, args=(test_file,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        backplane = StateBackplane(test_file)
        tasks = backplane.get_active_tasks()
        print(f"[Main] Final Check: {tasks}")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "task_test_123")
        self.assertEqual(tasks[0]["status"], 3)
        backplane.close()


if __name__ == "__main__":
    unittest.main()

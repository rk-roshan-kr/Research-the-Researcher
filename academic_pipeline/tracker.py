import os
import json
import logging
import sys
import time

class ProcessTracker:
    """
    Registers a background process so that the Flask web application can monitor
    its execution state, command line parameters, and stream its logs.
    """
    def __init__(self, task_id, command):
        self.task_id = task_id
        self.command = command
        self.pid = os.getpid()
        self.active_json_path = os.path.abspath("outputs/active_task.json")
        self.active_log_path = os.path.abspath("outputs/active_run.log")
        self.handler = None

    def start(self):
        os.makedirs(os.path.dirname(self.active_json_path), exist_ok=True)
        
        # Write task metadata for Web App UI polling
        task_info = {
            "task_id": self.task_id,
            "status": "running",
            "command": self.command,
            "pid": self.pid,
            "start_time": time.time(),
            "logs_file": self.active_log_path
        }
        
        try:
            with open(self.active_json_path, "w", encoding="utf-8") as f:
                json.dump(task_info, f, indent=2)
        except Exception as e:
            sys.stderr.write(f"Tracker: Failed to write active task metadata: {e}\n")
            
        # Hook a FileHandler to the root logger to mirror log entries to outputs/active_run.log
        try:
            # We clear/overwrite the log file by using mode='w'
            self.handler = logging.FileHandler(self.active_log_path, mode="w", encoding="utf-8")
            self.handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(self.handler)
        except Exception as e:
            sys.stderr.write(f"Tracker: Failed to configure log mirroring: {e}\n")

    def stop(self):
        # Remove the file handler from the logger
        if self.handler:
            try:
                logging.getLogger().removeHandler(self.handler)
                self.handler.close()
            except Exception as e:
                sys.stderr.write(f"Tracker: Failed to detach log handler: {e}\n")
                
        # Clean up task state file
        if os.path.exists(self.active_json_path):
            try:
                os.remove(self.active_json_path)
            except Exception as e:
                sys.stderr.write(f"Tracker: Failed to clean up metadata file: {e}\n")

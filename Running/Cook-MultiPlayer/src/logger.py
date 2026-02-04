import sys
import os
from datetime import datetime


class Logger:
    def __init__(self, log_file_path=None):
        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"logs/log_{timestamp}.log"
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.log_file = open(self.log_file_path, "a+", encoding="utf-8")

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        os.fsync(self.log_file.fileno())

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self  
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.close()
        print(f"\nAll terminal output has been saved to log file: {self.log_file_path}")

import threading
import time
import logging
from functools import wraps

# Thread-local storage to hold the stack of ScopeTimer instances
_scope_timer_stack = threading.local()

# Global variable for the logger; it will be set in the init function.
timer_logger = None


def scopetimer_init(log_file="scope_timer_log.txt",
                    mode="w",
                    level=logging.INFO):
    """
    Initialize the ScopeTimer logger.
    """
    global timer_logger
    timer_logger = logging.getLogger("ScopeTimer")
    # Clear existing handlers
    if timer_logger.hasHandlers():
        timer_logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(
        logging.Formatter('%(asctime)s [PID: %(process)d] - %(message)s'))
    timer_logger.addHandler(handler)
    timer_logger.setLevel(level)
    timer_logger.propagate = False  # Prevent log propagation to the root logger
    _scope_timer_stack.stack = []


class ScopeTimer:
    unique_id_counter = 0
    lock = threading.Lock()

    def __init__(self, msg, parent=None, log_to_screen=True):
        self.msg = msg
        self.stopped = False
        self.id = None
        self.parent = parent
        self.full_name = ""
        self.start_time = None
        self.log_to_screen = log_to_screen
        if not hasattr(_scope_timer_stack, 'stack'):
            raise RuntimeError(
                "ScopeTimer must be initialized with scopetimer_init()")

    def __enter__(self):
        with self.lock:
            self.id = ScopeTimer.unique_id_counter
            ScopeTimer.unique_id_counter += 1

        if self.parent:
            self.parent = self.parent
        elif _scope_timer_stack.stack:
            self.parent = _scope_timer_stack.stack[-1]
        else:
            self.parent = None

        _scope_timer_stack.stack.append(self)

        if self.parent:
            self.full_name = f"{self.parent.full_name}.{self.msg}"
        else:
            self.full_name = self.msg

        self.start_time = time.perf_counter()

        timer_logger.info(f"<<Timer>> {self.full_name}: start, id-{self.id}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def stop(self):
        if not self.stopped:
            end_time = time.perf_counter()
            total_time = end_time - self.start_time
            timer_logger.info(
                f"<<Timer>> {self.full_name}: {total_time:.3f}s, id-{self.id}")
            _scope_timer_stack.stack.pop()
            indent_size = len(_scope_timer_stack.stack)
            self.stopped = True
            if self.log_to_screen:
                print(
                    f'{"  " * indent_size}>> {self.msg}: {total_time*1000:.3f}ms'
                )
            return total_time
        return 0.0


def scopetimed(msg, log_to_screen=True):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            with ScopeTimer(msg, log_to_screen=log_to_screen):
                return func(*args, **kwargs)

        return wrapper

    return decorator

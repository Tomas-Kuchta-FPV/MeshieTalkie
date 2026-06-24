import select
import sys
import threading
import time
import termios


class TerminalRawInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        self.raw_settings = termios.tcgetattr(self.fd)
        self.raw_settings[3] = self.raw_settings[3] & ~(termios.ECHO | termios.ICANON)
        self.raw_settings[6][termios.VMIN] = 1
        self.raw_settings[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.raw_settings)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_settings)


class KeyboardPttController:
    def __init__(self, hold_timeout=0.7, restart_cooldown=1.0, on_record_start=None, on_record_stop=None):
        self.hold_timeout = hold_timeout
        self.restart_cooldown = restart_cooldown
        self.on_record_start = on_record_start
        self.on_record_stop = on_record_stop

        self._recording_active = threading.Event()
        self._last_t_time = None
        self._cooldown_until = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording_active.is_set()

    def run(self):
        with TerminalRawInput():
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    char = sys.stdin.read(1)
                    if char == "\x03":
                        raise KeyboardInterrupt
                    if char.lower() == "t":
                        now = time.monotonic()
                        if now < self._cooldown_until:
                            continue

                        if not self._recording_active.is_set():
                            self._recording_active.set()
                            if self.on_record_start is not None:
                                self.on_record_start()
                        self._last_t_time = now

                if self._recording_active.is_set() and self._last_t_time is not None:
                    now = time.monotonic()
                    if now - self._last_t_time >= self.hold_timeout:
                        self._recording_active.clear()
                        if self.on_record_stop is not None:
                            self.on_record_stop()
                        self._cooldown_until = now + self.restart_cooldown
                        self._last_t_time = None

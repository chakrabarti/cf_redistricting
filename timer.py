import time
from collections import Counter


class Timer:

    total_time_tracker = Counter()
    call_tracker = Counter()
    start_time = time.time()
    disabled = False

    def __init__(self, name=None):
        self.start = time.time()
        self.name = name

    @classmethod
    def TimerClassReset(cls):
        cls.total_time_tracker = Counter()
        cls.call_tracker = Counter()
        cls.start_time = time.time()
        cls.disabled = False

    def Reset(self):
        self.start = time.time()

    def TimeElapsed(self):
        return time.time() - self.start

    def Accumulate(self):
        if self.name is not None and not Timer.disabled:
            Timer.total_time_tracker[self.name] += self.TimeElapsed()
            Timer.call_tracker[self.name] += 1

    @classmethod
    def Disable(cls):
        cls.disabled = True

    @classmethod
    def Enable(cls):
        cls.disabled = False

    def PrintTimeElapsed(self):
        if self.name is None:
            print(f"Time elapsed: {self.TimeElapsed():.2e}")
        else:
            print(f"{self.name} time elapsed: {self.TimeElapsed():.2e}")

    @staticmethod
    def PrintAccumulated():
        all_total_time = time.time() - Timer.start_time

        print_matrix = []
        header_row = [
            "Name",
            "total time",
            "total percent",
            "num calls",
            "avg call time",
        ]
        print_matrix.append(header_row)

        for key in Timer.total_time_tracker:
            key_total_time = Timer.total_time_tracker[key]
            calls = Timer.call_tracker[key]
            avg_time = key_total_time / calls
            perc_time = 100 * key_total_time / all_total_time
            print_row = [
                key,
                f"{key_total_time:.5e}s",
                f"{perc_time:.5f}%",
                calls,
                f"{avg_time:.5e}s",
            ]
            print_matrix.append(print_row)

        s = [[str(e) for e in row] for row in print_matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = "  ".join("{{:{}}}".format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        s = "\n".join(table)
        s += f"\n Total Time: {all_total_time:.5f}"
        print(s)
        return s

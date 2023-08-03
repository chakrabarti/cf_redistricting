from enum import Enum


class State(Enum):
    PA = 1
    NC = 2
    MD = 3
    OH = 4

    @property
    def num_districts(self):
        if self.name == "PA":
            return 18
        elif self.name == "NC":
            return 13
        elif self.name == "MD":
            return 8
        elif self.name == "OH":
            return 16

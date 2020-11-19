import time


class Timer:
    def __init__(self, config):
        self.measure = config["Timer"]["measure"]
        self.times = []

    def start(self):
        if self.measure:
            self.times.append(("", time.perf_counter()))

    def log(self, msg):

        if self.measure:
            self.times.append((msg, time.perf_counter()))

    def print(self):
        if self.measure and len(self.times) > 1:

            txt = ""

            for i in range(len(self.times)-1):
                txt += "\n" + self.times[i+1][0] + " : " + \
                    str((self.times[i+1][1]-self.times[i][1])*1000.0)

            txt += "\ntotal " + str((self.times[-1][1]-self.times[0][1]) * 1000.0)

            print(txt)

        self.times = []

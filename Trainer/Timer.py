# MIT License

# Copyright (c) 2020 Patrik Persson and Linn Öström

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

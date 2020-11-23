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

from DataLoaderTask.DataExample import DataExample
from DataLoaderTask.DataLoaderTask import DataLoaderTask
import queue
import numpy as np
import os


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.q = queue.Queue(500)
        self.terminate = False

        self.datafolder = config['rgbd_dataset']['root_dir']

        if os.path.exists(self.datafolder):

            self.loaders = [DataLoaderTask(config, self.q) for i in range(12)]

        else:
            
            self.loaders = []
            self.terminate = True

        self.dataBuffer = []
        self.dataBufferRecycled = []
        self.i = 0
        self.max_buffer = 1000
        self.batch_size = config['model']['batch_size']

    def getIds(self):
        return self.loaders[0].getIds()

    def getNbrSamples(self):
        return self.loaders[0].getNbrSamples()

    def setExit(self):
        for loader in self.loaders:
            loader.setExit()

    def start(self):
        for loader in self.loaders:
            loader.start()

    def getItem(self):
        while True:
            try:
                item = self.q.get(block=True, timeout=1)
                return item, True

            except queue.Empty:
                if self.terminate:
                    return None, False

    def next(self):

        if self.terminate:
            return None, False

        while (len(self.dataBuffer) + len(self.dataBufferRecycled)) < self.batch_size and not self.terminate:

            item, flag = self.getItem()

            if flag:
                self.dataBuffer.append(item)

        if self.terminate:
            return None, False

        batch = []

        for i in range(self.batch_size):

            if len(self.dataBuffer) > 0:
                item = self.dataBuffer.pop(0)

                if len(self.dataBufferRecycled) > self.max_buffer:
                    self.dataBufferRecycled.pop(0)

                self.dataBufferRecycled.append(item)

                batch.append(item)

            elif len(self.dataBufferRecycled) > 0:
                index = np.random.randint(0, len(self.dataBufferRecycled))

                batch.append(self.dataBufferRecycled[index])

        if len(self.dataBuffer) < self.max_buffer and not self.q.empty():
            try:
                item = self.q.get(block=False)
                self.dataBuffer.append(item)

            except queue.Empty:
                if self.terminate:
                    return None, False

        return batch, len(batch) == self.batch_size

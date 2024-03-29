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

import tensorflow as tf
import time

class VoxelMap:

    def __init__(self):
        self.cell_width = tf.constant(0.1, dtype=tf.float32)
        self.primes = tf.reshape(tf.constant(
            [3259, 1487, 1093], dtype=tf.float32), shape=[1, 1, 3])
        self.voxels = dict()

    def update(self, X, id, g):

        t1 = time.perf_counter()

        voxel = g.voxelize2(X, self.cell_width, self.primes)

        shape = tf.shape(voxel)

        np_voxel = voxel.numpy()

        t2 = time.perf_counter()

        for i in range(shape[0]):
            voxel_visited = dict()

            for e in np_voxel[i]:

                if not self.update_visited_voxel(e, voxel_visited):
                    continue

                self.update_voxel(e, id[i])

        t3 = time.perf_counter()

        #print("{0}, {1}, {2}".format(1000.0*(t2-t1), 1000.0*(t3-t2), 1000.0*(t3-t1)))

    def fetch(self, X, g):

        voxel = g.voxelize2(X, self.cell_width, self.primes)

        shape = tf.shape(voxel)

        np_voxel = voxel.numpy()

        total_ids = dict()

        for i in range(shape[0]):
            voxel_visited = dict()

            for e in np_voxel[i]:

                if not self.update_visited_voxel(e, voxel_visited):
                    continue

                self.get_ids(e, total_ids)

        return total_ids

    def get_ids(self, e, total_ids):
        index = e[3]

        if index in self.voxels:
            val = self.voxels[index]

            ids = []

            for vox_ind in val:
                if (vox_ind[0] == e[0:3]).all():

                    for el in vox_ind[1]:
                        if el in total_ids:
                            total_ids[el] += 1
                        else:
                            total_ids.update({el: 0})

            return True

        return False

    def update_voxel(self, e, id):
        index = e[3]

        if index in self.voxels:
            val = self.voxels[index]

            for vox_ind in val:
                if (vox_ind[0] == e[0:3]).all():
                    vox_ind[1].append(id)

                    return True

            val.append((e[0:3], [id]))

            return True

        self.voxels.update({index: [(e[0:3], [id])]})

        return True

    def update_visited_voxel(self, e, voxel_visited):
        index = e[3]

        if index in voxel_visited:
            val = voxel_visited[index]

            for vox_ind in val:
                if (vox_ind == e[0:3]).all():

                    return False

            val.append(e[0:3])

        else:
            voxel_visited.update({index: [e[0:3]]})

        return True

    def contains(self, e, voxel_visited):
        index = e[3]

        if index in voxel_visited:
            val = voxel_visited[index]

            for vox_ind in val:
                if (vox_ind == e[0:3]).all():

                    return True

        return False

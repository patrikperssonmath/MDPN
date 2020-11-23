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

docker pull colmap/colmap:latest

docker run -u $(id -u):$(id -g) --gpus all -w /script -v $(pwd)/utility:/script -v $1:/working --rm -it colmap/colmap:latest /script/colmap_s.sh /working/ 0;

path_name=`dirname $1`

filename=`basename $1`

echo $path_name
echo $filename

docker run -u $(id -u):$(id -g) --rm --gpus all -it \
 -v $(pwd)/demo/data/:/data:rw \
 -v $(pwd)/demo/database/:/database:rw \
 -v $(pwd):/project:rw \
 -v $path_name:/working \
 -w /project \
  shapeestimator \
  python Converters/sfm_colmap_converter.py config/config.yml --i=/working --d=$filename
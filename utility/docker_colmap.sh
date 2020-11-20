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
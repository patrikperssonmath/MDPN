docker run -u $(id -u):$(id -g) --rm --gpus all -it \
 -v $(pwd)/demo/data/:/data:rw \
 -v $(pwd)/demo/database/:/database:rw \
 -v $(pwd):/project:rw \
 -w /project \
  shapeestimator \
  python main.py config/config.yml --d=$1

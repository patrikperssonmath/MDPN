docker run -u $(id -u):$(id -g) --rm --gpus all -it \
 -v /data/:/data:rw \
 -v /database/shape_info:/database:rw \
 -v $(pwd):/project:rw \
 -w /project \
  shapeestimator

# ShapeEstimator

## Installation
    1. Make sure you have an updated nvidia grapich card driver
    2. Install Docker Community
    3. Install NVIDIA Container Toolkit

# Usage
    1. Build container by running: ./demo/docker_build.sh
    2. Perform shape estimation by running: ./demo/docker_run_demo.sh <dataset>
    eg: ./demo/docker_run_demo.sh fountain
    3. The reconstructions are stored in ./demo/data/predictions/unsupervised/0/

# Add Your Own Images
    1. Put images in a folder /path/name/images/
    2. Run ./utility/docker_colmap.sh /path/name/
    3. To Reconstruct 3D structure: Run ./demo/docker_run_demo.sh name

# Optional
     Install vscode with docker plugin for easy in docker container development (optional)
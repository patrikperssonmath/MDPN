# ShapeEstimator

### Installation
1. Make sure you have an updated nvidia grapich card driver
2. Install Docker Community
3. Install NVIDIA Container Toolkit

### Download network and demo images

| Network |  Demo data |
|---|---|
|[Google Drive](https://drive.google.com/file/d/1T0cyOM50Cj5evPGqWKCBNINpHXg4d763/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1z7kX1gmeyTf3kAHFRsiaEdeVBbi71ZcL/view?usp=sharing)

 Unzip network in ./demo/data/models/,
```
./demo/data/models/
```
 Afterwards it should be 

```
./demo/data/models/ShapeNetwork
```
 Unzip demo in 
```
./demo/database/
```
Afterwards it should be 
```
./demo/database/sfm
```

### Usage

Build container by running: 
```
./demo/docker_build.sh
```
Perform shape estimation by running: 
```
./demo/docker_run_demo.sh "dataset"
```
As example:
```
./demo/docker_run_demo.sh fountain
```
The reconstructions are stored in 
```
./demo/data/predictions/unsupervised/0/
```

### Add Your Own Images
Put images in a folder 
```
/path/name/images/
```

Run 
```
./utility/docker_colmap.sh /path/name/
```
to to find a sparse sfm solution using [colmap](https://colmap.github.io/) and convert it to the expected format. The solutions are saved in

```
./demo/database/sfm/processed/
```

Run 
```
./demo/docker_run_demo.sh name
```
to reconstruct the 3D structure, 

### Optional
Install vscode with docker plugin for easy in docker container development (optional)

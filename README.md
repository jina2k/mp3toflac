# mp3toflac
Made for osu! purposes specifically, converting .mp3 or .ogg files to their .flac equivalent with a base .flac file


This script is intended for osu! purposes, to play songs at a higher quality as beatmaps are normally run on .mp3. It requires a FLAC version of the song as well, which normally is different (and longer) from the .mp3 version which is normally based on the Youtube music videos.

Basically, we run custom chunks (default set to 3.3s) of the .mp3 provided in the music folder, then we compare that to chunks in the .flac. It records a confidence score for each chunk, and uses ffmpeg to concatenate all chunks together to produce the end result. This results in an unranked version of the song, so leaderboards would be disabled if the music file is replaced. Chunks where confidence is below a certain rate (default 33%) is replaced with silence. The resulting file is named after the original mp3 song but with a .flac extension. This needs to manually replace the .mp3 file (make sure to keep a backup) in order to use it in osu!.

This file is generally intended to run on CUDA to produce faster runtime. However, it can also run on CPU as a fallback.

To install CUDA toolkit properly on WSL2 (Ubuntu):
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-wsl

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2

export PATH=/usr/local/cuda-13.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-13.2/lib64

sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev

nvcc --version

pip install cupy-cuda13x
```

An example python script to test if CUDA works properly is as follows below:
```
import cupy as cp

x = cp.array([1, 2, 3])
print(x)
print("Device:", cp.cuda.Device())
```

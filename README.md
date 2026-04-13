# mp3toflac
Made for osu! purposes specifically, converting .mp3 or .ogg files to their .flac equivalent with a base .flac file

### Disclaimer

This project was made entirely by vibecoding with ChatGPT (with some tweaks). After testing in-game with this, the experience is slightly "off" as sometimes the beats are not matching by the millisecond. This was mostly made just for fun, to see how it would sound, especially since I did not want to search around the music to cut when exactly certain parts happen. This can also technically be used to compare the Youtube music videos to the actual studio versions that are sold as .flacs. Lastly, this does not work with a longer .mp3 file vs a shorter .flac. The script assumes the .mp3 file is typically shorter than the .flac, if not the same length. This is typically the case with most osu! songs.

### Details (+setup)

This script is intended for osu! purposes, to play songs at a higher quality as beatmaps are normally run on .mp3. It requires a FLAC version of the song as well, which normally is different (and longer) from the .mp3 version which is normally based on the Youtube music videos.

Basically, we run custom chunks (default set to 3.3s) of the .mp3 provided in the music folder, then we compare that to chunks in the .flac. It records a confidence score for each chunk, and uses ffmpeg to concatenate all chunks together to produce the end result. This results in an unranked version of the song, so leaderboards would be disabled if the music file is replaced. Chunks where confidence is below a certain rate (default 50%) is replaced with silence. The resulting file is named after the original mp3 song but with a .flac extension. All files with an extension .osu in the same folder as the .mp3 then replace the AudioFilename with the .flac instead.

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

### Anecdotes

Personally, I have tested this on a few songs in particular, ie; https://osu.ppy.sh/beatmapsets/71561 (The Beginning) and https://osu.ppy.sh/beatmapsets/1400282 (On & On). The output would look like the below (for On & On):
<img width="601" height="946" alt="image" src="https://github.com/user-attachments/assets/64053fb2-2f79-4bd9-b46d-2f3ccbaeb945" />

Note the skip from Chunk 27. I also do not recommend skipping in-game as it does not work properly with the .flac files. The overall pacing does not match if you skip for whatever reason. 

## This Project has Moved!

You can find the new project page on [Codeberg](https://codeberg.org/slembcke/lifft).

At least for now, I will be supporting the project on both sites, but eventually I plan to eventually drop GitHub entirely.

To migrate an existing checkout, you can use the following command, or make a new checkout of the project from the new location.

```
git remote set-url origin https://codeberg.org/slembcke/lifft
git fetch origin master
git reset origin/master
```

You may additionally need to run `git checkout .` to cleanup the deleted files. *This will also discard your local changes if you have made any.*

To read more about why this project moved, you can read about it in [PROJECT_HAS_MOVED.md](PROJECT_HAS_MOVED.md).

# 🎛️ LiFFT
The little FFT library.

A simple FFT and DCT (II/III/IV) library in a header only format that makes it easy to drop into a project and use. Measuring in at ~150 sloc for the FFT functions, and another ~70 for DCTs it's main goal is just to be small. Even though performance isn't the main concern, it's reasonably comparable to [FFTPACK](https://www.netlib.org/fftpack/) or [FFTW](https://fftw.org/).

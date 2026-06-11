Fast tilt-series alignment, including methods for:
- Excluding tilt images based on tilt-series image statistics.
- Finding the specimen orientation (rotation, tilt, pitch) using image cross-correlation and common-lines.
- Aligning images (XY translation) using projection matching.
- Refining tilt-axis angle using projection matching.
- Finding the specimen thickness by analyzing the signal in the tomogram.
- Fitting the CTF of the tilt-series.

The CTF fitting can be run independently of the tilt-series alignment and can fit:
- per-image defoci.
- tilt-dependent astigmatisms (up to per-image).
- time-dependent phase-shifts (up to per-image).
- specimen orientation (rotation, tilt, pitch).
- specimen thickness.


## `Build`

```shell
git clone https://github.com/thomasfrosio/quinoa.git
cd quinoa
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install -DQN_ENABLE_CUDA=ON # enable GPU support
cmake --build ./build --parallel
cmake --install ./build
```

Additional configure options can be passed, especially to make sure the correct compilers are used. These are often used:

```shell
...
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install \
    -DQN_ENABLE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/path/to/specific/dependency \
    -DCMAKE_C_COMPILER=/path/to/c \
    -DCMAKE_CXX_COMPILER=/path/to/c++ \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CUDA_COMPILER=/path/to/cuda-toolkit
...
```

When compiling a single binary to run on GPU with different architectures, use `all-major` or `all`. See [`DCMAKE_CUDA_ARCHITECTURES`](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES) for more details.


## `Run`

Copy the `settings.toml` from the repository. Feel free

```shell
quinoa --help

quinoa --settings=quinoa.toml
quinoa --settings=quinoa.toml --mdocs=mdocs/*

```

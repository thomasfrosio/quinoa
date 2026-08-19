## `Work In Progress`

This repository is a work in progress. The CTF estimation is the only completed part (and the only part built by default). The tilt-series alignment and tomogram reconstruction are mostly done, but the code in this repository is incomplete and not ready for use. The full version should be available by the end of August 2026.


## `Quinoa`

### `CTF estimate`
Fast CTF fitting, running independently of the tilt-series alignment and including fit of:
  - per-image defoci.
  - tilt-dependent astigmatisms (up to per-image).
  - time-dependent phase-shifts (up to per-image).
  - specimen orientation (tilt, pitch, and refine rotation).
  - specimen thickness.

### `Tilt-series alignment`
Fast tilt-series alignment, including methods for:
- Excluding tilt images based on tilt-series image statistics.
- Finding the specimen orientation (rotation, tilt, pitch) using image cross-correlation and common-lines.
- Aligning images (XY translation) using projection matching.
- Refining tilt-axis angle using projection matching.
- Finding the specimen thickness by analyzing the signal in the tomogram.


## `Dependencies`

- `CMake >=3.23`.
- `clang++ >=21` or `g++ >=14.2`
- If `CUDA` is enabled, use a toolkit version `>=12.8`, including `>=13` if your GPU supports it.
- For TIFF file support, `libtiff` is required. It should be already installed on most systems and automatically found by CMake. If it is not found, TIFF files will not be supported.
- Other dependencies are automatically downloaded by CMake, built, and statically linked to the application. This includes `noa`, `spdlog`, `nlopt`, `cxxopts`, `tomlplusplus`, `Eigen`, `glob-cpp`.


## `Build`

```shell
git clone git@github.com:thomasfrosio/quinoa.git
cd quinoa
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install -DQN_ENABLE_CUDA=ON # enable GPU support
cmake --build ./build --parallel
cmake --install ./build
```

Additional configure options can be passed, especially to make sure the correct compilers are used. CUDA can be quite finicky about these things, so these are recommended:

```shell
...
cmake -B ./build -DCMAKE_INSTALL_PREFIX=./install \
    -DQN_ENABLE_CUDA=ON \
    -DCMAKE_PREFIX_PATH=/path/to/specific/dependency \
    -DCMAKE_C_COMPILER=/path/to/c \
    -DCMAKE_CXX_COMPILER=/path/to/c++ \
    -DCMAKE_CUDA_ARCHITECTURES=all \
    -DCMAKE_CUDA_COMPILER=/path/to/cuda-toolkit
...
```

When compiling a single binary to run on GPU with different architectures, use `all-major` or `all`. See [`DCMAKE_CUDA_ARCHITECTURES`](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES) for more details. By default, we build for with `native`, specifically targeting the GPU architecture on the system used for the build.


## `Run`

```shell
# Examples:
quinoa --help
quinoa --settings=share/settings_ctf.toml
quinoa --mdocs=*.mdoc --stacks=*.mrc --tilt-axis=175
```
See setting files in the installation directory (or in [share](share/)) for more information.

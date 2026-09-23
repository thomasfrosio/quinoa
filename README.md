## `Quinoa`

### `CTF estimate`
Fast CTF fitting, possibly running independently of the tilt-series alignment and including fit of:
  - per-image defoci.
  - tilt-dependent astigmatisms (up to per-image).
  - time-dependent phase-shifts (up to per-image).
  - specimen orientation (tilt, pitch, and refine rotation).
  - specimen thickness.

See this [preprint](https://www.biorxiv.org/content/10.64898/2026.07.15.738674v1) to learn more about how it works.

### `Tilt-series alignment`
Fast tilt-series alignment, including methods for:
- Excluding tilt images based on tilt-series image statistics.
- Finding the specimen orientation (rotation, tilt, pitch) using image cross-correlation and common-lines.
- Aligning images (XY translation) using projection matching.
- Finding the specimen thickness by analyzing the signal in the tomogram.

These methods work together to optimize for a (rigid-body) tomogram containing at its center the leveled specimen.

This project was focused on implementing fast methods to project tomograms from its central slices. These projections are used to align the tilt images using a projection matching algorithm similar to [AreTomo](https://github.com/czimaginginstitute/AreTomo3). However, using projection matching as an optimization metric to align tilt-series is [inherently limited](https://www.biorxiv.org/content/10.64898/2026.04.29.721716v1) and may produce mediocre results. As such, and similar to AreTomo, it should only be used for initial alignment and should be refined with other methods down the line. The advantage of our method is speed, as we can easily align a tilt-series in a few seconds if a GPU is available.

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
# (WIP): quinoa --mdocs=*.mdoc --stacks=*.mrc --tilt-axis=175
```
See setting files in the installation directory (or in [share](share/)) for more information.

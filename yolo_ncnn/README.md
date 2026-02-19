# Requirements

1. **[emsdk](https://github.com/emscripten-core/emsdk)** downloaded and installed.
2. **[ncnn simd](https://github.com/Tencent/ncnn)** built for **wasm**. (Can be built locally or downloaded.)

---

### Setup

To set up the environment, use the following steps:

1. **Build:**

   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
   emmake make
   ```

Once built, ensure that you grab the yolo.data, yolo.js, yolo.wasm

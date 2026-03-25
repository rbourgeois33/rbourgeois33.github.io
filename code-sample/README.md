# Instruction:
Load `cuda` or `rocm`. Then in build e.g. `cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/cuda.ada.cmake`, you can easily make a custom toolchain file

To run and generate a `ncu` report:
```
ncu --target-processes all --import-source yes --set full -f -o sample-1.ncu-rep ./sample-1
```

for a `rocprof-compute` report:

```
rocprof-compute profile --name sample-1 -- ./sample-1
```


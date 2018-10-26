# Troubleshooting

Here is a compilation if common issues that you might face
while compiling / running this code:

## Compilation errors when compiling the library
If you encounter build errors like the following:
```
/usr/include/c++/6/type_traits:1558:8: note: provided for ‘template<class _From, class _To> struct std::is_convertible’
     struct is_convertible
        ^~~~~~~~~~~~~~
/usr/include/c++/6/tuple:502:1: error: body of constexpr function ‘static constexpr bool std::_TC<<anonymous>, _Elements>::_NonNestedTuple() [with _SrcTuple = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>&&; bool <anonymous> = true; _Elements = {at::Tensor, at::Tensor, at::Tensor, at::Tensor}]’ not a return-statement
     }
 ^
error: command '/usr/local/cuda/bin/nvcc' failed with exit status 1
```
check your CUDA version and your `gcc` version.
```
nvcc --version
gcc --version
```
If you are using CUDA 9.0 and gcc 6.4.0, then refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/25,
which has a summary of the solution. Basically, CUDA 9.0 is not compatible with gcc 6.4.0.

## ImportError: No module named maskrcnn_benchmark.config when running webcam.py

This means that `maskrcnn-benchmark` has not been properly installed.
Refer to https://github.com/facebookresearch/maskrcnn-benchmark/issues/22 for a few possible issues.
Note that we now support Python 2 as well.

RTX2080Ti needs CUDA10 to work properly.
Execute:
```
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=10.0

cd maskrcnn-benchmark
rm -rf build/
python setup.py build develop
```
instead of:
```
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0
```
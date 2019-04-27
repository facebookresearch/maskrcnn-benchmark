### Reference 
1 [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/pdf/1811.11168.pdf)  
2 third-party: [mmdetection](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn)  

### Performance
|      case                   | bbox AP | mask AP |
|----------------------------:|--------:|:-------:|
| R-50-FPN-dcn (implement)    |  39.8   |  -      |
| R-50-FPN-dcn (mmdetection)  |  40.0   |  -      |
| R-50-FPN-mdcn (implement)   |  40.0   |  -      |
| R-50-FPN-mdcn (mmdetection) |  40.3   |  -      |
| R-50-FPN-dcn (implement)    |  40.8   |  36.8   |
| R-50-FPN-dcn (mmdetection)  |  41.1   |  37.2   |
| R-50-FPN-dcn (implement)    |  40.7   |  36.7   |
| R-50-FPN-dcn (mmdetection)  |  41.4   |  37.4   |


### Note
see [dcn-v2](https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md#deformable-convolution-v2) in `mmdetection` for more details.  


### Usage
add these three lines
```
MODEL:
	RESNETS:
		# corresponding to C2,C3,C4,C5
		STAGE_WITH_DCN: (False, True, True, True)
		WITH_MODULATED_DCN: True
		DEFORMABLE_GROUPS: 1
```
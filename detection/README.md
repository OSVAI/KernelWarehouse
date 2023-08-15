# MS-COCO Object Detection with KernelWarehouse 

We use the popular [MMDetection](https://github.com/open-mmlab/mmdetection) toolbox for experiments on the MS-COCO dataset with the pre-trained ResNet50, MobileNetV2 (1.0×) and ConvNeXt-Tiny models as the backbones for the detector. We select the mainstream Faster RCNN and Mask R-CNN detectors with Feature Pyramid Networks as the necks to build the basic object detection systems.


## Training

Please follow [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) on how to prepare the environment and the dataset. Then attach our code to the origin project and modify the config files according to your own path to the pre-trained models and directories to save logs and models.

To train a detector with pre-trained models as backbone:

```shell
bash tools/dist_train.sh {path to config file} {number of gpus}
```

## Evaluation

To evaluate a fine-tuned model:
```shell
bash tools/dist_test.sh {path to config file} {path to fine-tuned model} {number of gpus} --eval bbox segm --show
```


## Results and Models

| Backbones          | Detectors | box AP | mask AP | Config | Google Drive | Baidu Drive |
|:------------|:-------:|:------:|:-------:|:-------------:|:-------------:|:-------------:|
| ResNet50           | Mask R-CNN |  39.6  |  36.4   | [config](configs/kernelwarehouse/mask_rcnn_resnet50_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1j6wSJLett-JeVDTh7CW7CHhC4jQHDzad/view?usp=sharing) | [model](https://pan.baidu.com/s/1U7q2U0jYXjDCAVxqUMWmHw?pwd=4wih) |
| + KW (1×)          | Mask R-CNN |  41.8  |  38.4   | [config](configs/kernelwarehouse/resnet50/mask_rcnn_kw1x_resnet50_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1XBXKF8TU0iMFVBt-IF048hAmYTL9-spk/view?usp=sharing) | [model](https://pan.baidu.com/s/1AI01STe9v0KzAKVVPMUhog?pwd=a7ce) |
| + KW (4×)          | Mask R-CNN |  42.4  |  38.9   | [config](configs/kernelwarehouse/resnet50/mask_rcnn_kw4x_resnet50_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1GUDEr2jNT0Il8A04g_f8sRQ1WFAycOO8/view?usp=sharing) | [model](https://pan.baidu.com/s/1ZSJkfVy8xr5IB_OfubXzRw?pwd=xig5) |
| MobileNetV2 (1.0×) | Mask R-CNN |  33.8  |  31.7   | [config](configs/kernelwarehouse/mobilenetv2/mask_rcnn_mobilenetv2_100_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1UJifIqx56cOOm2hx-D3DSHh4MWgFzOVB/view?usp=sharing) | [model](https://pan.baidu.com/s/1S7vo59mzEVL_8ai9Sg1iUQ?pwd=4sh8) |
| + KW (1×)      | Mask R-CNN |  36.4  |  33.7   | [config](configs/kernelwarehouse/mobilenetv2/mask_rcnn_kw1x_mobilenetv2_100_adamw_1x_coco.py) |  [model](https://drive.google.com/file/d/1wdzs-Ry6LefgG4Nc9RWUlrDrsyGOWhL5/view?usp=sharing) | [model](https://pan.baidu.com/s/1q3U4Euw2qNCWXipPCn4vtQ?pwd=8g38) |
| + KW (4×)      | Mask R-CNN |  38.0  |  34.9   | [config](configs/kernelwarehouse/mobilenetv2/mask_rcnn_kw4x_mobilenetv2_100_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/14nfWpHUHgH0mA4gbEPX3F_3UqOXPIGK7/view?usp=sharing) | [model](https://pan.baidu.com/s/1HidKe3MgnIEERvvKgdYMHg?pwd=n5uu) |
| ConvNeXt-Tiny      | Mask R-CNN |  43.4  |  39.7   | [config](configs/kernelwarehouse/convnext_tiny/mask_rcnn_convnext_tiny_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1xarty4OTZOKGo1ltAUcTJCoKHCIOipC6/view?usp=sharing) | [model](https://pan.baidu.com/s/1bouC_aK9C1czPrIYkkS3Ug?pwd=79f4) |
| + KW (4×)      | Mask R-CNN |  44.7  |  40.6   | [config](configs/kernelwarehouse/convnext_tiny/mask_rcnn_kw1x_convnext_tiny_adamw_1x_coco.py) | [model](https://drive.google.com/file/d/1simtPisVzZo__iSXZwrynWi6TlUwPG3b/view?usp=sharing) | [model](https://pan.baidu.com/s/1iBD4lCrvSTX0Wu7e2I0BKg?pwd=am2w) |
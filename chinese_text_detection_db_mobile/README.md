## 概述

Differentiable Binarization（简称DB）是一种基于分割的文本检测算法。在各种文本检测算法中，基于分割的检测算法可以更好地处理弯曲等不规则形状文本，因此往往能取得更好的检测效果。但分割法后处理步骤中将分割结果转化为检测框的流程复杂，耗时严重。DB将二值化阈值加入训练中学习，可以获得更准确的检测边界，从而简化后处理流程。该Module是一个超轻量级文本检测模型，支持直接预测。

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/image/ocr/db_algo.png" hspace='10'/> <br />
</p>

更多详情参考[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)


## 命令行预测

```shell
$ hub run chinese_text_detection_db_mobile --input_path "/PATH/TO/IMAGE"
```

**该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper。**

## API

## API

### \_\_init\_\_(enable_mkldnn=False)

构造ChineseTextDetectionDB对象

**参数**

* enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。


```python
def detect_text(paths=[],
                images=[],
                use_gpu=False,
                output_dir='detection_result',
                box_thresh=0.5,
                visualization=False)
```

预测API，检测输入图片中的所有中文文本的位置。

**参数**

* paths (list\[str\]): 图片的路径；
* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
* box\_thresh (float): 检测文本框置信度的阈值；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 detection\_result；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    * data (list): 检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
    * save_path (str): 识别结果的保存路径, 如不保存图片则save_path为''

### 代码示例

```python
import paddlehub as hub
import cv2

text_detector = hub.Module(name="chinese_text_detection_db_mobile", enable_mkldnn=True)
result = text_detector.detect_text(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result =text_detector.detect_text(paths=['/PATH/TO/IMAGE'])
```




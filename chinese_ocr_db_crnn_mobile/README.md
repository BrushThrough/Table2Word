## 概述

chinese_ocr_db_crnn_mobile Module用于识别图片当中的汉字。其基于[chinese_text_detection_db_mobile Module](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db_mobile&en_category=TextRecognition)检测得到的文本框，继续识别文本框中的中文文字。之后对检测文本框进行角度分类。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个超轻量级中文OCR模型，支持直接预测。


<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/image/ocr/rcnn.png" hspace='10'/> <br />
</p>

更多详情参考[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)

## 命令行预测

```shell
$ hub run chinese_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE"
```

**该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper。**

## API

### \_\_init\_\_(text_detector_module=None, enable_mkldnn=False)

构造ChineseOCRDBCRNN对象

**参数**

* text_detector_module(str): 文字检测PaddleHub Module名字，如设置为None，则默认使用[chinese_text_detection_db_mobile Module](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db_mobile&en_category=TextRecognition)。其作用为检测图片当中的文本。
* enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。


```python
def recognize_text(images=[],
                    paths=[],
                    use_gpu=False,
                    output_dir='ocr_result',
                    visualization=False,
                    box_thresh=0.5,
                    text_thresh=0.5,
                    angle_classification_thresh=0.9)
```

预测API，检测输入图片中的所有中文文本的位置。

**参数**

* paths (list\[str\]): 图片的路径；
* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
* box\_thresh (float): 检测文本框置信度的阈值；
* text\_thresh (float): 识别中文文本置信度的阈值；
* angle_classification_thresh(float): 文本角度分类置信度的阈值
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，默认设为 ocr\_result；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
    * data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
        * text(str): 识别得到的文本
        * confidence(float): 识别文本结果置信度
        * text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
      如果无识别结果则data为\[\]
    * save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为''

### 代码示例

```python
import paddlehub as hub
import cv2

ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
```

* 样例结果示例

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/image/ocr/ocr_res.jpg" hspace='10'/> <br />
</p>

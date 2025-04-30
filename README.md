
# OpenCV 学习与实践项目集

本项目集包含多个使用OpenCV实现的计算机视觉应用，用于学习和实践目的。

## 项目列表

### 1. 文档扫描仪 (document scanner)
- 功能：将任意角度的文档图片转换为规整的扫描件
- 核心技术：边缘检测、透视变换、二值化处理
- 使用方法：
  ```bash
  python scan.py -i img/test.jpeg
  ```

### 2. 人脸检测系统 (FaceDetector)
- 功能：检测图片或视频中的人脸
- 核心技术：Caffe深度学习模型、DNN模块
- 使用方法：
  ```bash
  python detect_faces.py -i test.jpeg -p deploy.prototxt -m res10_300x300_ssd_iter_140000.caffemodel
  ```

### 3. 人脸识别系统 (Facerecognition)
- 完整工作流程：
  1. 图片采集：使用search_serapi.py从网络下载人脸图片到dataset
  2. 特征编码：使用encode_faces.py生成人脸特征编码
  3. 图片识别：使用recognize_faces_image.py识别静态图片中的人脸
  4. 视频识别：使用recognize_faces_video.py实时识别视频流中的人脸

- 核心技术：face_recognition库、CNN/HOG特征提取、SerpAPI图片搜索

- 使用方法：
  ```bash
  # 1. 采集图片 (需要SerpAPI key)
  python search_serapi.py
  
  # 2. 生成特征编码
  python encode_faces.py --dataset dataset --encodings encodings.pickle
  
  # 3. 静态图片识别
  python recognize_faces_image.py -i examples/example01.png -e encodings.pickle
  
  # 4. 实时视频识别
  python recognize_faces_video.py -e encodings.pickle -o output/output.mp4

  # 5. 视频识别
  python python recognize_faces_video_file.py --encodings encodings.pickle \
	--input videos/output_fixed.mp4 --output output/ output.mp4 \
	--display 0
  ```

### 4. 实时物体检测 (ObjectDetection)
- 功能：实时检测摄像头画面中的常见物体
- 核心技术：MobileNet SSD模型
- 使用方法：
  ```bash
  python real_time_object_detection.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel
  ```

### 5. 显著性检测系统 (SaliencyDetection)
- 功能：检测图像中最吸引人注意的区域
- 包含三种检测方法：
  1. 静态显著性检测：识别图像中静态的显著区域
  2. 运动显著性检测：识别视频中的运动显著区域
  3. 物体显著性检测：基于物体特征的显著性检测
- 核心技术：OpenCV显著性检测算法
- 使用方法：
  ```bash
  # 静态显著性检测
  python static_saliency.py -i images/barcelona.jpg
  
  # 运动显著性检测
  python motion_saliency.py
  
  # 物体显著性检测
  python objectness_saliency.py
  ```

### 6. 物体追踪系统 (CamShiftTrackObject)
- 功能：基于CamShift算法追踪视频中的移动物体
- 核心技术：颜色直方图分析、MeanShift算法
- 使用方法：
  ```bash
  python track.py
  ```

### 7. 人脸聚类系统 (FaceClustering)
- 功能：对未知人脸数据集进行自动聚类分组
- 核心技术：CNN特征提取、层次聚类算法
- 使用方法：
  ```bash
  # 1. 提取人脸特征
  python encode_faces.py --dataset dataset --encodings encodings.pickle
  
  # 2. 执行聚类
  python cluster_faces.py --encodings encodings.pickle --jobs -1
  ```

### 8. 人数统计系统 (PeopleCounter)
- 功能：实时统计视频中经过的人数
- 核心技术：MobileNet SSD检测、质心追踪算法
- 使用方法：
  ```bash
  python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
    --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
    --input videos/example01.mp4
  ```

### 9. 语义分割系统 (SemanticSegmentation)
- 功能：对图像/视频进行像素级语义分割
- 核心技术：ENet深度学习模型
- 使用方法：
  ```bash
  # 图片分割
  python segment.py -i images/example01.jpg
  
  # 视频分割
  python segment_video.py -v videos/citywalk.mp4
  ```

### 10. 文本检测系统 (TextDetection)
- 功能：检测图像中的文本区域
- 核心技术：EAST文本检测模型
- 使用方法：
  ```bash
  # 图片文本检测
  python text_detection.py --image images/ex01.jpg --east frozen_east_text_detection.pb
  
  # 视频文本检测
  python text_detection_video.py --east frozen_east_text_detection.pb
  ```

## 环境要求
- Python 3.6+
- OpenCV 4.x
- 其他依赖库：
  ```
  numpy
  imutils
  face_recognition
  scikit-image
  ```

## 项目结构
```
.
├── CamShiftTrackObject/  # 物体追踪项目
├── DocumentScanner/      # 文档扫描项目
├── FaceClustering/       # 人脸聚类项目
├── FaceDetector/         # 人脸检测项目
├── Facerecognition/      # 人脸识别项目
├── ObjectDetection/      # 物体检测项目
├── PeopleCounter/        # 人数统计项目
├── SaliencyDetection/    # 显著性检测项目
├── SemanticSegmentation/ # 语义分割项目
└── TextDetection/        # 文本检测项目
```


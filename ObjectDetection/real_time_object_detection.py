from imutils.video import VideoStream  # 用來讀取攝影機影像
from imutils.video import FPS  # 用來計算每秒幀數（FPS）
import numpy as np  # 匯入數學函式庫，常用於矩陣與數值運算
import argparse     # 用來處理命令列參數
import cv2          # OpenCV 函式庫，進行影像處理與模型推論
import time         # 用來計算時間
import imutils


# 建立命令列參數解析器
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")  # Caffe 模型的 prototxt 結構檔案
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model file (.caffemodel)")  # 模型訓練好的權重檔案
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")  # 最低信心值門檻
args = vars(ap.parse_args())  # 解析參數

# 預定義的分類類別（來自 MobileNet SSD 模型訓練的標籤）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# 為每個類別指定一個隨機顏色（用於畫框）
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 載入模型
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()  # 開啟攝影機
time.sleep(2.0)  # 等待攝影機暖機
fps = FPS().start()  # 開始計算 

while True:
    # 讀取影格
    frame = vs.read()
    frame = imutils.resize(frame, width=400)  # 調整影格大小
    
    (h, w) = frame.shape[:2]  # 取得影格的高度與寬度
    # 建立 blob：將影格縮放、轉換格式供模型輸入
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # 進行物件偵測
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    # 顯示結果影格
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF  # 等待鍵盤輸入
    if key == ord("q"):  # 如果按下 'q' 鍵，就退出迴圈
        break
    fps.update()  # 更新 FPS 計算

# 停止計算 FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))  # 顯示每秒幀數
# 關閉攝影機與視窗
cv2.destroyAllWindows()
vs.stop()

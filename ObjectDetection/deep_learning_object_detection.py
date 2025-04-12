import numpy as np  # 匯入數學函式庫，常用於矩陣與數值運算
import argparse     # 用來處理命令列參數
import cv2          # OpenCV 函式庫，進行影像處理與模型推論

# 建立命令列參數解析器
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")  # 輸入圖片路徑
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

# 讀取圖片並取得其尺寸
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# 建立 blob：將圖片縮放、轉換格式供模型輸入
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             0.007843, (300, 300), 127.5)

# 進行物件偵測
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# 對所有偵測結果進行處理
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]  # 取得信心值

    # 如果信心值超過指定門檻，就視為有效偵測
    if confidence > args["confidence"]:
        idx = int(detections[0, 0, i, 1])  # 取得物件類別的索引
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 計算實際框的位置
        (startX, startY, endX, endY) = box.astype("int")

        # 建立標籤文字（類別 + 百分比）
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))

        # 繪製偵測框與文字標籤
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# 顯示結果圖片
cv2.imshow("Output", image)
cv2.waitKey(0)  # 等待任意鍵後關閉視窗

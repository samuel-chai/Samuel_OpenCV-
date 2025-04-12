from imutils.video import VideoStream  # 使用 imutils 封裝的簡化版攝影機擷取方式
import numpy as np                    # 數值處理用的 numpy 函式庫
import argparse                       # 處理命令列參數
import imutils                        # 包含影像處理常用工具（resize 等）
import time                           # 用於等待攝影機初始化
import cv2                            # OpenCV 函式庫（核心）

# 建立命令列參數解析器
ap = argparse.ArgumentParser()
# 指定 prototxt 網路架構檔案（必填）
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
# 指定模型權重檔案（.caffemodel）（必填）
ap.add_argument("-m", "--model", required=True, help="path to Caffe 'deploy' prototxt file")
# 設定信心值門檻，低於這個值的不顯示（可選，預設為 0.5）
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())  # 解析參數後轉成 dictionary

# 載入訓練好的 Caffe 模型
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 啟動攝影機並等待感測器啟動（2 秒）
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()  # src=0 表示預設攝影機
time.sleep(2.0)

# 持續讀取攝影機畫面直到按下 'q'
while True:
    frame = vs.read()  # 讀取一幀影像
    frame = imutils.resize(frame, width=400)  # 調整影像大小為寬度 400（等比例縮放）

    # 擷取影像尺寸（高、寬）
    (h, w) = frame.shape[:2]

    # 將影像轉為 blob 格式，供模型輸入：
    # 1. 調整為 300x300 尺寸
    # 2. 使用預設均值減除 (104, 177, 123)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # 將影像 blob 設定為模型輸入
    net.setInput(blob)

    # 進行推論，取得所有偵測結果
    detections = net.forward()

    # 遍歷所有偵測結果
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # 取得每個框的信心值

        # 忽略信心值太低的預測
        if confidence < args["confidence"]:
            continue

        # 轉換偵測框座標為實際像素位置
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")  # 轉為整數

        # 計算文字標籤內容（顯示信心百分比）
        text = "{:.2f}%".format(confidence * 100)
        # 決定文字顯示位置，避免超出畫面
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # 在畫面上畫出框與信心值文字（紅色）
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)

    # 顯示目前這一幀的影像
    cv2.imshow("Frame", frame)

    # 等待鍵盤輸入（1ms），按下 q 就離開
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 關閉所有視窗並釋放攝影機資源
cv2.destroyAllWindows()
vs.stop()
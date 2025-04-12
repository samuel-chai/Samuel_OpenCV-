import numpy as np  # 匯入數學函式庫，常用於矩陣與數值運算
import argparse     # 用來處理命令列參數
import cv2          # OpenCV 函式庫，進行影像處理與模型推論

# 建立命令列參數解析器，允許使用者從外部傳入檔案路徑等設定
ap = argparse.ArgumentParser()

# 加入參數：輸入圖片路徑（必填）
ap.add_argument("-i", "--image", required=True, help="path to input image")

# 加入參數：Caffe 架構檔（.prototxt）（必填）
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")

# 加入參數：訓練好的模型權重（.caffemodel）（必填）
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model file (.caffemodel)")

# 加入參數：設定最低信心門檻（可選，預設為0.5）
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

# 解析命令列參數
args = vars(ap.parse_args())

# 顯示提示訊息：正在載入模型
print("[INFO] loading model...")

# 使用 OpenCV 的 DNN 模組載入 Caffe 模型（網路架構與權重）
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 讀入輸入圖片
image = cv2.imread(args["image"])

# 取得圖片高度與寬度（後續繪製框線時會用到）
(h, w) = image.shape[:2]

# 建立 blob（模型的標準輸入格式）：
# 1. 將圖片縮放至 300x300
# 2. 設定 scale 為 1.0
# 3. 減去平均值進行 normalize（使用訓練時的預設值：104, 177, 123）
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             1.0, (300, 300),
                             (104.0, 177.0, 123.0))

# 將 blob 輸入模型
net.setInput(blob)

# 進行 forward 推論，取得預測結果
print("[INFO] computing object detections...")
detections = net.forward()

# 逐一檢查每個預測結果（每個偵測框）
for i in range(0, detections.shape[2]):
    # 取得該預測框的信心值（範圍在 0~1 之間）
    confidence = detections[0, 0, i, 2]

    # 若信心值高於使用者設定的門檻，就認為是有效偵測
    if confidence > args["confidence"]:
        # 取得該預測框的座標（是比例值，所以要乘上圖片寬高）
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")  # 轉換成整數型別

        # 建立顯示文字（顯示信心百分比）
        text = "{:.2f}%".format(confidence * 100)

        # 決定文字顯示的 Y 座標位置（避免超出圖片邊界）
        y = startY - 15 if startY - 15 > 15 else startY + 15

        # 在圖片上畫出偵測框（綠色框線）
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # 顯示信心文字（白色字體）
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

# 顯示結果圖片（開啟一個視窗）
cv2.imshow("Output", image)

# 等待按鍵輸入後關閉視窗（按任意鍵）
cv2.waitKey(0)
import os
import requests
import cv2
from imutils import paths

# ==================== 設定區 ====================
API_KEY = "508f636e5dc5465eaf5fb2a9ad213fd390189599343a8d3538e5912e9fbc1692"  # ← 換成你自己的金鑰
SEARCH_TERM = "Owen Grady"       # ← 替換為你想要的搜尋關鍵字
NUM_IMAGES = 35               # ← 想要下載的圖片數量
OUTPUT_FOLDER = f"dataset/{SEARCH_TERM}"  # 儲存的資料夾
# ==============================================

# 建立儲存資料夾
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 查詢圖片資料
print(f"[INFO] 正在使用 SerpAPI 搜尋圖片：{SEARCH_TERM}")
params = {
    "engine": "google",
    "q": SEARCH_TERM,
    "tbm": "isch",
    "num": NUM_IMAGES,
    "api_key": API_KEY
}
search_url = "https://serpapi.com/search.json"
response = requests.get(search_url, params=params)

# 解析結果
results = response.json()
image_results = results.get("images_results", [])
print(f"[INFO] 找到 {len(image_results)} 張圖片")

# 下載每一張圖
for idx, image in enumerate(image_results[:NUM_IMAGES]):
    url = image["original"]
    try:
        image_data = requests.get(url, timeout=5).content
        file_path = os.path.join(OUTPUT_FOLDER, f"{str(idx).zfill(4)}.jpg")
        with open(file_path, "wb") as f:
            f.write(image_data)
        print(f"[INFO] 已儲存圖片: {file_path}")
    except Exception as e:
        print(f"[WARNING] 無法下載圖片：{url}，錯誤：{e}")

# 檢查無法讀取的檔案（壞圖）
for imagePath in paths.list_images(OUTPUT_FOLDER):
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[INFO] 刪除壞圖: {imagePath}")
        os.remove(imagePath)

print("[DONE] 圖片搜尋與下載完成 ✅")
# Glass Defect Detection

本專案為一個基於顏色差異的玻璃瑕疵自動檢測系統，使用 Python 與 OpenCV 實作，能夠自動偵測影像中與背景顏色不同的區域，並進行分類與視覺化。

## 目錄結構

```
data.html         # 測試數據與說明網頁
main.py           # 主程式，瑕疵檢測核心
path.txt          # 影像路徑與程式說明
test.py           # 測試用程式
test.txt          # 測試計畫與需求說明
```

## 主要功能

- 自動估算影像背景顏色
- 建立顏色差異圖
- 動態或固定閾值偵測異常區域
- 瑕疵分類（白色、黑色、灰色 defect）
- 瑕疵資訊統計與視覺化顯示

## 依賴套件

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

安裝方式：
```sh
pip install opencv-python numpy matplotlib
```

## 使用方式

1. 修改 `main.py` 中的 `image_path` 變數，指定要檢測的影像路徑。
2. 執行主程式：
   ```sh
   python main.py
   ```
3. 程式會顯示瑕疵統計資訊，並彈出視窗顯示原圖、灰階圖、顏色差異圖、異常遮罩與瑕疵標註結果。

## 參數調整

- `sensitivity`：調整檢測靈敏度（數值越小越敏感，預設 1.5）
- `min_area`：最小瑕疵面積過濾（預設 15）

可於 `main.py` 的 `detector.detect_all_defects()` 參數中調整。


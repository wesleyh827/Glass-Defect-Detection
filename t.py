import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import os
import glob
import pandas as pd

# 定義defect資訊結構
DefectInfo = namedtuple('DefectInfo', ['contour', 'area', 'center', 'bbox', 'type', 'color_diff'])

class ColorBasedDefectDetector:
    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.defects = []
        self.background_color = None
        
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"無法載入影像: {image_path}")
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return True   
    
    def estimate_background_color(self, sample_size=0.1):
        """估算背景顏色 - 使用影像邊緣區域的統計"""
        h, w = self.gray_image.shape
        
        # 取影像四個角落的區域作為背景樣本 
        corner_size = int(min(h, w) * sample_size)
        
        corners = [
            self.gray_image[0:corner_size, 0:corner_size],            # 左上
            self.gray_image[0:corner_size, w-corner_size:w],          # 右上
            self.gray_image[h-corner_size:h, 0:corner_size],          # 左下
            self.gray_image[h-corner_size:h, w-corner_size:w]         # 右下
        ]
        
        # 計算四個角落的中位數，取最穩定的值作為背景
        corner_medians = [np.median(corner) for corner in corners]
        self.background_color = np.median(corner_medians)
        
        print(f"估算的背景顏色值: {self.background_color:.1f}")
        return self.background_color

    def create_color_difference_map(self, blur_kernal=5):
        if self.background_color is None:
            self.estimate_background_color()

        blurred = cv2.GaussianBlur(self.gray_image, (blur_kernal, blur_kernal), 0) # deblur

        diff_map = np.abs(blurred.astype(float) - self.background_color) # calculate diff with background 

        return diff_map.astype(np.uint8)
        
    def detect_color_anomalies(self, sensitivity=2.0, min_area=10):
        diff_map = self.create_color_difference_map()
        
        # 計算差異圖的統計特性 
        mean_diff = np.mean(diff_map)
        std_diff = np.std(diff_map)
        
        # 動態閾值：背景 + sensitivity * 標準差
        threshold = mean_diff + sensitivity * std_diff
        
        # threshold = 6 
        print(f"\nthreshold: {threshold:.1f}")
        # print(f"顏色差異閾值: {threshold:.1f} (平均差異: {mean_diff:.1f}, 標準差: {std_diff:.1f})")
        
        # 建立二值化遮罩
        _, anomaly_mask = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
        
        # 形態學處理去除雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel) # 開運算 先腐蝕後膨脹。用於去除小的白色雜訊點。
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel) # 閉運算 先膨脹後腐蝕。用於連接被分隔開的物體並填充小的孔洞。
        
        return anomaly_mask, diff_map
        
    def classify_defects_by_color(self, contour, diff_map):
        # 建立遮罩
        mask = np.zeros(self.gray_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # 計算defect區域的平均顏色值
        defect_pixels = self.gray_image[mask > 0]
        avg_defect_color = np.mean(defect_pixels)
        
        # 計算顏色差異
        color_diff = avg_defect_color - self.background_color
        
        # 分類
        if color_diff > 10:  # 比背景亮
            defect_type = "白色defect"
        elif color_diff < -10:  # 比背景暗
            defect_type = "黑色defect"
        else:
            defect_type = "灰色defect"  # 輕微差異
            
        return defect_type, abs(color_diff)

    def find_contours_and_classify(self, anomaly_mask, diff_map, min_area=10):
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_list = [] 
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # 計算重心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # 計算邊界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 分類defect類型
                defect_type, color_diff = self.classify_defects_by_color(contour, diff_map)
                
                defect_info = DefectInfo(
                    contour=contour,
                    area=area,
                    center=(cx, cy),
                    bbox=(x, y, w, h),
                    type=defect_type,
                    color_diff=color_diff
                )
                defect_list.append(defect_info)
        
        return defect_list
    def detect_all_defects(self, sensitivity=2.0, min_area=10):
        print(f"this is what it sounds like first you both") 
         
    def detect_all_defects(self, sensitivity=2.0, min_area=10):
        if self.gray_image is None:
            raise ValueError("請先載入影像")
            
        # 檢測顏色異常
        anomaly_mask, diff_map = self.detect_color_anomalies(sensitivity, min_area)
        
        # 找出並分類defect
        self.defects = self.find_contours_and_classify(anomaly_mask, diff_map, min_area)
        
        # 按面積大小排序 (由大到小)
        self.defects.sort(key=lambda x: x.area, reverse=True)
        
        return self.defects, anomaly_mask, diff_map
        
    def print_defect_summary(self):
        if not self.defects:
            print("未檢測到任何defect")
            return
            
        print(f"\n=== 檢測到 {len(self.defects)} 個defect ===")
        print("-" * 60)
        
        # 統計各類型數量
        white_count = sum(1 for d in self.defects if d.type == "白色defect")
        black_count = sum(1 for d in self.defects if d.type == "黑色defect")
        gray_count = sum(1 for d in self.defects if d.type == "灰色defect")
        
        print(f"白色defect數量: {white_count}")
        print(f"黑色defect數量: {black_count}")
        print(f"灰色defect數量: {gray_count}")
        print(f"背景顏色值: {self.background_color:.1f}")
        print("-" * 60)
        
        for i, defect in enumerate(self.defects):
            print(f"第{i+1}大 - {defect.type}")
            print(f"  面積: {defect.area:.2f} pixels")
            print(f"  顏色差異: {defect.color_diff:.2f}")
            print(f"  中心位置: {defect.center}")
            print(f"  邊界框: x={defect.bbox[0]}, y={defect.bbox[1]}, w={defect.bbox[2]}, h={defect.bbox[3]}")
            print()

class BatchDefectAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.results = []
        
    def get_image_files(self):
        """取得資料夾中所有的bmp檔案"""
        pattern = os.path.join(self.folder_path, "*.bmp")
        return glob.glob(pattern)
        
    def analyze_folder(self, sensitivity=1.5, min_area=15):
        """批次分析整個資料夾的影像"""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"在資料夾 {self.folder_path} 中找不到任何 .bmp 檔案")
            return
            
        print(f"找到 {len(image_files)} 個 .bmp 檔案，開始分析...")
        
        for i, image_path in enumerate(image_files):
            try:
                filename = os.path.basename(image_path)
                print(f"\n處理中 ({i+1}/{len(image_files)}): {filename}")
                
                # 初始化檢測器
                detector = ColorBasedDefectDetector()
                detector.load_image(image_path)
                
                # 檢測defect - 使用原始參數
                defects, anomaly_mask, diff_map = detector.detect_all_defects(
                    sensitivity=sensitivity,
                    min_area=min_area
                )
                
                # 計算統計資訊
                total_image_area = detector.gray_image.shape[0] * detector.gray_image.shape[1]
                total_defect_area = sum(defect.area for defect in defects)
                defect_ratio_percent = (total_defect_area / total_image_area) * 100
                
                # 統計各類型defect
                white_count = sum(1 for d in defects if d.type == "白色defect")
                black_count = sum(1 for d in defects if d.type == "黑色defect")
                gray_count = sum(1 for d in defects if d.type == "灰色defect")
                
                white_area = sum(d.area for d in defects if d.type == "白色defect")
                black_area = sum(d.area for d in defects if d.type == "黑色defect")
                gray_area = sum(d.area for d in defects if d.type == "灰色defect")
                
                # 儲存結果
                result = {
                    'filename': filename,
                    'image_path': image_path,
                    'total_image_area': total_image_area,
                    'total_defect_area': total_defect_area,
                    'defect_ratio_percent': defect_ratio_percent,
                    'defect_count': len(defects),
                    'white_count': white_count,
                    'black_count': black_count,
                    'gray_count': gray_count,
                    'white_area': white_area,
                    'black_area': black_area,
                    'gray_area': gray_area,
                    'background_color': detector.background_color
                }
                
                # 顯示這張圖片的結果
                print(f"  defect總面積: {total_defect_area:,} pixels")
                print(f"  影像總面積: {total_image_area:,} pixels") 
                print(f"  defect佔比: {defect_ratio_percent:.3f}%")
                print(f"  defect數量: {len(defects)} 個")
                
                self.results.append(result)
                
            except Exception as e:
                print(f"處理 {image_path} 時發生錯誤: {e}")
                
        print(f"\n分析完成!共處理了 {len(self.results)} 張影像")
        
    def create_summary_report(self):
        """建立統計報告""" 
        if not self.results:
            print("沒有分析結果可以顯示")
            return None
           
        # 轉換為DataFrame
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("批次分析報告")
        print("="*80)
        
        print(f"總共分析影像數量: {len(df)}")
        print(f"平均defect佔比: {df['defect_ratio_percent'].mean():.3f}%")
        print(f"最大defect佔比: {df['defect_ratio_percent'].max():.3f}%")
        print(f"最小defect佔比: {df['defect_ratio_percent'].min():.3f}%")
        print(f"defect佔比標準差: {df['defect_ratio_percent'].std():.3f}%")
        
        print(f"\n各類型defect統計:")
        print(f"白色defect總數: {df['white_count'].sum()}")
        print(f"黑色defect總數: {df['black_count'].sum()}")
        print(f"灰色defect總數: {df['gray_count'].sum()}")
        
        # 找出defect佔比最高和最低的影像
        max_idx = df['defect_ratio_percent'].idxmax()
        min_idx = df['defect_ratio_percent'].idxmin()
        
        print(f"\ndefect佔比最高的影像:")
        print(f"  檔案: {df.loc[max_idx, 'filename']}")
        print(f"  佔比: {df.loc[max_idx, 'defect_ratio_percent']:.3f}%")
        print(f"  defect面積: {df.loc[max_idx, 'total_defect_area']:,} pixels")
        print(f"  defect數量: {df.loc[max_idx, 'defect_count']}")
        
        print(f"\ndefect佔比最低的影像:")
        print(f"  檔案: {df.loc[min_idx, 'filename']}")
        print(f"  佔比: {df.loc[min_idx, 'defect_ratio_percent']:.3f}%")
        print(f"  defect面積: {df.loc[min_idx, 'total_defect_area']:,} pixels")
        print(f"  defect數量: {df.loc[min_idx, 'defect_count']}")
        
        return df
        
    def plot_distribution_analysis(self, df):
        """繪製分布分析圖表"""
        plt.figure(figsize=(15, 10))
        
        # 1. defect佔比分布直方圖 
        plt.subplot(2, 3, 1)
        plt.hist(df['defect_ratio_percent'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Defect Ratio (%)')
        plt.ylabel('Image Count')
        plt.title('Distribution of Defect Ratios')
        plt.grid(True, alpha=0.3)
        
        # 2. defect數量分布
        plt.subplot(2, 3, 2)
        plt.hist(df['defect_count'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Number of Defects')
        plt.ylabel('Image Count')
        plt.title('Distribution of Defect Counts')
        plt.grid(True, alpha=0.3)
        
        # 3. defect佔比 vs defect數量散點圖
        plt.subplot(2, 3, 3)
        plt.scatter(df['defect_count'], df['defect_ratio_percent'], alpha=0.6)
        plt.xlabel('Number of Defects')
        plt.ylabel('Defect Ratio (%)')
        plt.title('Defect Count vs Ratio')
        plt.grid(True, alpha=0.3)
        
        # 4. 各類型defect數量統計
        plt.subplot(2, 3, 4)
        defect_types = ['White', 'Black', 'Gray']
        type_counts = [df['white_count'].sum(), df['black_count'].sum(), df['gray_count'].sum()]
        plt.bar(defect_types, type_counts, color=['red', 'blue', 'gray'])
        plt.ylabel('Total Count')
        plt.title('Defect Types Distribution')
        
        # 5. defect佔比趨勢（按檔名排序）
        plt.subplot(2, 3, 5)
        plt.plot(range(len(df)), df['defect_ratio_percent'], 'o-', markersize=4)
        plt.xlabel('Image Index')
        plt.ylabel('Defect Ratio (%)')
        plt.title('Defect Ratio Trend')
        plt.grid(True, alpha=0.3)
        
        
        """
       
        # 6. 累積分布圖
        plt.subplot(2, 3, 6)
        sorted_ratios = np.sort(df['defect_ratio_percent'])
        cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
        plt.plot(sorted_ratios, cumulative, 'b-', linewidth=2)
        plt.xlabel('Defect Ratio (%)')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Distribution of Defect Ratios')
        plt.grid(True, alpha=0.3)
        """
        plt.tight_layout()
        plt.show()
        
    def save_results_to_csv(self, output_path="defect_analysis_results.csv"):
        """將結果儲存為CSV檔案"""
        if not self.results:
            print("沒有結果可以儲存")
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"結果已儲存至: {output_path}")


def main():
    # 設定資料夾路徑 - 請修改為你的實際路徑
    folder_path = r"x:\Project\ASSESSMENT\AOI_Defect_img"
    
    # 建立批次分析器
    analyzer = BatchDefectAnalyzer(folder_path)
    
    # 執行批次分析 - 使用和原始程式相同的參數 
    analyzer.analyze_folder(sensitivity=1.5, min_area=15)
    
    # 產生統計報告
    df = analyzer.create_summary_report()
    
    if df is not None:
        # 繪製分布分析圖表
        analyzer.plot_distribution_analysis(df)
        
        # 儲存結果
        analyzer.save_results_to_csv("defect_analysis_results.csv")

if __name__ == "__main__":
    main()
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
        """估算背景顏色 - 使用8個邊緣區域的統計，去除極值後取中位數"""
        h, w = self.gray_image.shape
        
        # 取樣區域大小
        corner_size = int(min(h, w) * sample_size)
        edge_size = int(min(h, w) * sample_size * 0.5)  # 邊緣中點區域稍小一些
        
        # 8個採樣點：4個角落 + 4個邊緣中點
        sample_regions = [
            # 四個角落
            self.gray_image[0:corner_size, 0:corner_size],                          # 左上角
            self.gray_image[0:corner_size, w-corner_size:w],                        # 右上角
            self.gray_image[h-corner_size:h, 0:corner_size],                        # 左下角
            self.gray_image[h-corner_size:h, w-corner_size:w],                      # 右下角
            
            # 四個邊緣中點
            self.gray_image[0:edge_size, w//2-edge_size//2:w//2+edge_size//2],      # 上邊中點
            self.gray_image[h-edge_size:h, w//2-edge_size//2:w//2+edge_size//2],    # 下邊中點
            self.gray_image[h//2-edge_size//2:h//2+edge_size//2, 0:edge_size],      # 左邊中點
            self.gray_image[h//2-edge_size//2:h//2+edge_size//2, w-edge_size:w]     # 右邊中點
        ]
        
        # 計算每個區域的中位數
        region_medians = [np.median(region) for region in sample_regions]
        
        # 排序並去除最大2個和最小2個極值
        sorted_medians = sorted(region_medians)
        # 去除最小的2個和最大的2個，保留中間4個
        filtered_medians = sorted_medians[2:-2]
        
        # 取剩餘值的中位數作為最終背景顏色
        self.background_color = np.median(filtered_medians)
        
        print(f"8個採樣點的中位數: {[f'{x:.1f}' for x in region_medians]}")
        print(f"去除極值後保留: {[f'{x:.1f}' for x in filtered_medians]}")
        print(f"最終背景顏色值: {self.background_color:.1f}")
        
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
        # threshold = mean_diff + sensitivity * std_diff
        
        threshold = 6  # 保持原始設定
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
                
                # 計算統計資訊 - 和原始程式碼相同的計算方式
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
                
        print(f"\n分析完成！共處理了 {len(self.results)} 張影像")
        
    def create_summary_report(self):
        """建立統計報告"""
        if not self.results:
            print("沒有分析結果可以顯示")
            return None
            
        # 轉換為DataFrame方便處理
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
        sorted_df = df.sort_values('filename')
        plt.plot(range(len(sorted_df)), sorted_df['defect_ratio_percent'], 'o-', markersize=4)
        plt.xlabel('Image Index (sorted by filename)')
        plt.ylabel('Defect Ratio (%)')
        plt.title('Defect Ratio Trend')
        plt.grid(True, alpha=0.3)
        
        # 6. 累積分布圖
        plt.subplot(2, 3, 6)
        sorted_ratios = np.sort(df['defect_ratio_percent'])
        cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
        plt.plot(sorted_ratios, cumulative, 'b-', linewidth=2)
        plt.xlabel('Defect Ratio (%)')
        plt.ylabel('Cumulative Percentage (%)')
        plt.title('Cumulative Distribution of Defect Ratios')
        plt.grid(True, alpha=0.3)
        
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
        
    def rename_files_by_defect_ratio(self, output_folder=None, prefix="defect_", sort_order="ascending", include_area=True):
        """根據缺陷率重新命名檔案
        
        Args:
            output_folder: 輸出資料夾路徑，如果為None則在原資料夾建立重命名對照表
            prefix: 新檔名的前綴
            sort_order: 排序方式 ("ascending": 低到高, "descending": 高到低)
            include_area: 是否在檔名中包含defect面積
        """
        if not self.results:
            print("沒有分析結果，請先執行 analyze_folder()")
            return
            
        # 轉換為DataFrame並排序
        df = pd.DataFrame(self.results)
        
        if sort_order == "ascending":
            # 缺陷率從低到高排序（品質從好到壞）
            sorted_df = df.sort_values('defect_ratio_percent', ascending=True)
            print("排序方式：缺陷率由低到高（品質由好到壞）")
        else:
            # 缺陷率從高到低排序（品質從壞到好）
            sorted_df = df.sort_values('defect_ratio_percent', ascending=False)
            print("排序方式：缺陷率由高到低（品質由壞到好）")
        
        # 計算需要的位數（例如2200張圖片需要4位數）
        total_files = len(sorted_df)
        digits = len(str(total_files))

        print(f"總共 {total_files} 個檔案，使用 {digits} 位數編號")
        
        # 建立重命名對照表
        rename_mapping = []
        
        for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
            original_filename = row['filename']
            original_path = row['image_path']
            
            # 取得原始檔案的副檔名
            file_extension = os.path.splitext(original_filename)[1]
            
            # 產生新檔名
            if include_area:
                defect_area = int(row['total_defect_area'])
                new_filename = f"{prefix}{i:0{digits}d}_area_{defect_area}{file_extension}"
            else:
                new_filename = f"{prefix}{i:0{digits}d}{file_extension}"
            
            # 如果有指定輸出資料夾，產生新路徑
            if output_folder:
                new_path = os.path.join(output_folder, new_filename)
            else:
                new_path = os.path.join(os.path.dirname(original_path), new_filename)
            
            rename_mapping.append({
                'rank': i,
                'original_filename': original_filename,
                'new_filename': new_filename,
                'original_path': original_path,
                'new_path': new_path,
                'defect_ratio_percent': row['defect_ratio_percent'],
                'defect_count': row['defect_count'],
                'total_defect_area': row['total_defect_area']
            })
        
        # 儲存重命名對照表
        mapping_df = pd.DataFrame(rename_mapping)
        mapping_csv_path = os.path.join(self.folder_path, "file_rename_mapping.csv")
        mapping_df.to_csv(mapping_csv_path, index=False, encoding='utf-8-sig')
        print(f"重命名對照表已儲存至: {mapping_csv_path}")
        
        return mapping_df
    
    def execute_file_rename(self, mapping_df, create_backup=True):
        """執行檔案重命名
        
        Args:
            mapping_df: 重命名對照表DataFrame
            create_backup: 是否建立備份資料夾
        """
        import shutil
        
        if create_backup:
            backup_folder = os.path.join(self.folder_path, "original_backup")
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)
                print(f"建立備份資料夾: {backup_folder}")
        
        print(f"\n開始重命名 {len(mapping_df)} 個檔案...")
        success_count = 0
        error_count = 0
        
        for idx, row in mapping_df.iterrows():
            try:
                original_path = row['original_path']
                new_path = row['new_path']
                
                # 檢查原始檔案是否存在
                if not os.path.exists(original_path):
                    print(f"警告：原始檔案不存在 - {original_path}")
                    error_count += 1
                    continue
                
                # 如果需要備份，先複製到備份資料夾
                if create_backup:
                    backup_path = os.path.join(backup_folder, row['original_filename'])
                    shutil.copy2(original_path, backup_path)
                
                # 執行重命名
                os.rename(original_path, new_path)
                success_count += 1
                
                if (idx + 1) % 100 == 0:  # 每100個檔案顯示進度
                    print(f"已處理 {idx + 1}/{len(mapping_df)} 個檔案...")
                    
            except Exception as e:
                print(f"重命名失敗: {row['original_filename']} -> {row['new_filename']}")
                print(f"錯誤訊息: {e}")
                error_count += 1
        
        print(f"\n重命名完成！")
        print(f"成功: {success_count} 個檔案")
        print(f"失敗: {error_count} 個檔案")
        if create_backup:
            print(f"原始檔案已備份至: {backup_folder}")
    
    def preview_rename_results(self, mapping_df, preview_count=10):
        """預覽重命名結果
        
        Args:
            mapping_df: 重命名對照表DataFrame
            preview_count: 預覽的檔案數量
        """
        print(f"\n=== 重命名預覽 (前{preview_count}個檔案) ===")
        print("-" * 120)
        print(f"{'排名':<6} {'原始檔名':<40} {'新檔名':<20} {'缺陷率(%)':<12} {'缺陷數量':<8}")
        print("-" * 120)
        
        for idx, row in mapping_df.head(preview_count).iterrows():
            print(f"{row['rank']:<6} {row['original_filename']:<40} {row['new_filename']:<20} "
                  f"{row['defect_ratio_percent']:<12.3f} {row['defect_count']:<8}")
        
        if len(mapping_df) > preview_count:
            print("...")
            print(f"總共 {len(mapping_df)} 個檔案")
        
        print("-" * 120)
        print(f"品質最好 (缺陷率最低): {mapping_df.iloc[0]['new_filename']} "
              f"({mapping_df.iloc[0]['defect_ratio_percent']:.3f}%)")
        print(f"品質最差 (缺陷率最高): {mapping_df.iloc[-1]['new_filename']} "
              f"({mapping_df.iloc[-1]['defect_ratio_percent']:.3f}%)")


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
        
        # 新增：檔案重命名功能
        print("\n" + "="*80)
        print("檔案重命名功能")
        print("="*80)
        
        # 讓使用者選擇排序方式
        print("請選擇排序方式：")
        print("1. 缺陷率由低到高 (品質由好到壞)")
        print("2. 缺陷率由高到低 (品質由壞到好)")
        
        while True:
            choice = input("請輸入選擇 (1 或 2): ").strip()
            if choice == "1":
                sort_order = "ascending"
                break
            elif choice == "2":
                sort_order = "descending"
                break
            else:
                print("請輸入 1 或 2")
        
        # 讓使用者設定檔名前綴
        prefix = input("請輸入新檔名前綴 (預設: defect_): ").strip()
        if not prefix:
            prefix = "defect_"
        
        # 讓使用者選擇是否包含defect面積
        print("是否在檔名中包含defect面積？")
        print("1. 是 (例如: defect_0001_area_1234.bmp)")
        print("2. 否 (例如: defect_0001.bmp)")
        
        while True:
            area_choice = input("請輸入選擇 (1 或 2): ").strip()
            if area_choice == "1":
                include_area = True
                break
            elif area_choice == "2":
                include_area = False
                break
            else:
                print("請輸入 1 或 2")
        
        # 產生重命名對照表
        mapping_df = analyzer.rename_files_by_defect_ratio(
            prefix=prefix, 
            sort_order=sort_order,
            include_area=include_area
        )
        
        if mapping_df is not None:
            # 預覽重命名結果
            analyzer.preview_rename_results(mapping_df, preview_count=15)
            
            # 確認是否執行重命名
            print(f"\n注意：即將重命名 {len(mapping_df)} 個檔案")
            print("原始檔案會自動備份到 'original_backup' 資料夾")
            
            confirm = input("確定要執行重命名嗎？(y/n): ").strip().lower()
            
            if confirm == 'y':
                # 執行重命名
                analyzer.execute_file_rename(mapping_df, create_backup=True)
                print("\n重命名完成！")
            else:
                print("已取消重命名。重命名對照表仍已儲存，之後可以手動執行。")

def rename_only_mode():
    """僅執行重命名功能（如果已經有分析結果）"""
    folder_path = r"x:\Project\ASSESSMENT\AOI_Defect_img"
    
    # 檢查是否有現有的分析結果
    results_file = "defect_analysis_results.csv"
    
    if os.path.exists(results_file):
        print(f"找到現有分析結果: {results_file}")
        print("載入現有結果進行重命名...")
        
        # 讀取現有結果
        df = pd.read_csv(results_file)
        
        # 重建結果格式
        analyzer = BatchDefectAnalyzer(folder_path)
        analyzer.results = df.to_dict('records')
        
        # 執行重命名流程
        print("請選擇排序方式：")
        print("1. 缺陷率由低到高 (品質由好到壞)")
        print("2. 缺陷率由高到低 (品質由壞到好)")
        
        while True:
            choice = input("請輸入選擇 (1 或 2): ").strip()
            if choice == "1":
                sort_order = "ascending"
                break
            elif choice == "2":
                sort_order = "descending"
                break
            else:
                print("請輸入 1 或 2")
        
        prefix = input("請輸入新檔名前綴 (預設: defect_): ").strip()
        if not prefix:
            prefix = "defect_"
        
        print("是否在檔名中包含defect面積？")
        print("1. 是 (例如: defect_0001_area_1234.bmp)")
        print("2. 否 (例如: defect_0001.bmp)")
        
        while True:
            area_choice = input("請輸入選擇 (1 或 2): ").strip()
            if area_choice == "1":
                include_area = True
                break
            elif area_choice == "2":
                include_area = False
                break
            else:
                print("請輸入 1 或 2")
        
        mapping_df = analyzer.rename_files_by_defect_ratio(prefix=prefix, sort_order=sort_order, include_area=include_area)
        
        if mapping_df is not None:
            analyzer.preview_rename_results(mapping_df, preview_count=15)
            
            confirm = input(f"\n確定要重命名 {len(mapping_df)} 個檔案嗎？(y/n): ").strip().lower()
            
            if confirm == 'y':
                analyzer.execute_file_rename(mapping_df, create_backup=True)
                print("重命名完成！")
            else:
                print("已取消重命名。")
    else:
        print(f"找不到分析結果檔案: {results_file}")
        print("請先執行完整分析，或確認檔案路徑正確。")

if __name__ == "__main__":
    import sys
    
    print("鏡片缺陷檢測系統")
    print("1. 完整分析 + 重命名")
    print("2. 僅執行重命名 (使用現有分析結果)")
    
    mode = input("請選擇模式 (1 或 2): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        rename_only_mode()
    else:
        print("請輸入 1 或 2")
        sys.exit(1)
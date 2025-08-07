import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

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
        
        # 取樣區域大小
        corner_size = int(min(h, w) * sample_size)
        edge_size = int(min(h, w) * sample_size) 
        
        # 8個採樣點：4個角落 + 4個邊緣中點
        sample_regions = [
            # 四個角落
            self.gray_image[0:corner_size, 0:corner_size],                          # 左上角
            self.gray_image[0:corner_size, w-corner_size:w],                        # 右上角
            self.gray_image[h-corner_size:h, 0:corner_size],                        # 左下角
            self.gray_image[h-corner_size:h, w-corner_size:w],                      # 右下角
            
            # 四個邊緣中點
            self.gray_image[0:edge_size, w//2-edge_size//2:w//2+edge_size//2],      # 上邊
            self.gray_image[h-edge_size:h, w//2-edge_size//2:w//2+edge_size//2],    # 下邊
            self.gray_image[h//2-edge_size//2:h//2+edge_size//2, 0:edge_size],      # 左邊
            self.gray_image[h//2-edge_size//2:h//2+edge_size//2, w-edge_size:w]     # 右邊
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
        
        threshold = 6

        print(f"\nthreshold: {threshold:.1f}")
        # print(f"顏色差異閾值: {threshold:.1f} (平均差異: {mean_diff:.1f}, 標準差: {std_diff:.1f})")
        
        # 建立二值化遮罩 奶綠 閃電狼 +
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
        
        # 計算defect區域的平均顏色值 我睜開雙眼看著空白 忘記你對我的期待 讀完了依賴 我很快就離開 
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
        self.background_color = self.estimate_background_color()
        
        return self.defects, anomaly_mask, diff_map
        
    def draw_defects(self, show_area_text=True, show_color_diff=True):
        if self.original_image is None or not self.defects:
            return None
            
        result_img = self.original_image.copy()
        
        for i, defect in enumerate(self.defects):
            # 根據類型選擇顏色
            if defect.type == "白色defect":
                color = (0, 0, 255)      # 紅色標記白色defect
            elif defect.type == "黑色defect":
                color = (255, 0, 0)      # 藍色標記黑色defect
            else:
                color = (0, 255, 255)    # 黃色標記灰色defect 
            
            # 繪製輪廓
            cv2.drawContours(result_img, [defect.contour], -1, color, 2)
            
            # 標記重心
            cv2.circle(result_img, defect.center, 3, color, -1)
            
            # 顯示資訊
            if show_area_text:
                if show_color_diff:
                    text = f"{i+1}: {defect.area:.0f}px (diff:{defect.color_diff:.1f})"
                else:
                    text = f"{i+1}: {defect.area:.0f}px"
                    
                cv2.putText(result_img, text, 
                           (defect.center[0]-50, defect.center[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        return result_img
        
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

"""

"""
def main():
    # 初始化檢測器
    detector = ColorBasedDefectDetector()
    
    # 載入影像
    image_path = r"x:\Project\ASSESSMENT\AOI_Defect_img\defect_2273_area_3114.bmp"
    
    try:
        detector.load_image(image_path)
        
        # 檢測defect - 調整敏感度參數
        defects, anomaly_mask, diff_map = detector.detect_all_defects(
            sensitivity=1.5,     # 敏感度：越小越敏感，越大越保守
            min_area=15         # 最小面積過濾
        )
        
        # 顯示統計資訊
        detector.print_defect_summary()
        
        # 繪製結果
        result_image = detector.draw_defects(show_area_text=False, show_color_diff=True)
        
        # 顯示影像 
        plt.figure(figsize=(12, 9))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(detector.original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Img')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(detector.gray_image, cmap='gray')
        plt.title('Grayscale img')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(diff_map, cmap='hot')
        plt.title('color diff map')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(anomaly_mask, cmap='gray')
        plt.title('detected abnormal areas')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Defect results')
        plt.axis('off')
        """
        # 面積分布直方圖
        plt.subplot(2, 3, 6)
        if defects:
            areas = [d.area for d in defects]
            plt.hist(areas, bins=10, alpha=0.7)
            plt.title('Defect area distribution')
            plt.xlabel('area (pixels)')
            plt.ylabel('Quantity')
        """

        plt.tight_layout()
        plt.show()
        

    except Exception as e:
        print(f"錯誤: {e}")

def test_sensitivity(image_path):
    """測試不同敏感度參數的效果"""
    detector = ColorBasedDefectDetector()
    detector.load_image(image_path)
    
    sensitivities = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print("敏感度測試結果:")
    print("-" * 40)
    
    for sens in sensitivities:
        defects, _, _ = detector.detect_all_defects(sensitivity=sens)
        total_area = sum(d.area for d in defects) if defects else 0
        print(f"敏感度 {sens}: {len(defects)} 個defect, 總面積 {total_area:.0f}")

if __name__ == "__main__":
    main()
    
    # test_sensitivity(r"x:\Project\ASSESSMENT\AOI_Defect_img\DefectCode=6_ID=123_X=215847.5_Y=234443.5.bmp")
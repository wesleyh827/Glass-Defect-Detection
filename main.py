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
        
        threshold = 6 # 閾值可自行調整
        print(f"\nthreshold: {threshold:.1f}")
        
        # 建立二值化遮罩
        _, anomaly_mask = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
        
        # 形態學處理去除雜訊
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
        
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

    
 
def main():
    # 初始化檢測器
    detector = ColorBasedDefectDetector()
    
    # 載入影像
    image_path = "/your/image/path" # 自行修改路徑
    
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

        plt.tight_layout()
        plt.show()
        

    except Exception as e:
        print(f"錯誤: {e}")
"""

revenge game 
bokudachiwa moisakaru dadimoejuredenai
sekai asamuguwu yuri
I will always remember
The day you kissed my lips
Light as a feather
And it went just like this
No, it's never been better
Than the summer of two thousand and two

We were only eleven
But acting like grown-ups
Like we are in the present
Drinking from plastic cups
Singing "love is forever and ever"
Well, I guess that was true

Dancing on the hood
In the middle of the woods
Of an old Mustang
Where we sang
Songs with all our childhood friends
And it went like this, say

Oops, I got 99 problems singing bye, bye, bye
Hold up, if you wanna go and take a ride with me
Better hit me, baby, one more time
Paint a picture for you and me
Of the days when we were young, uh
Singing at the top of both our lungs

Now we're under the covers
Fast forward to eighteen
We are more than lovers
Yeah, we are all we need
When we're holding each other
I'm taken back to two thousand and two

Dancing on the hood
In the middle of the woods
Of an old Mustang
Where we sang
Songs with all our childhood friends
And it went like this, say

Oops, I got 99 problems singing bye, bye, bye
Hold up, if you wanna go and take a ride with me
Better hit me, baby, one more time
Paint a picture for you and me
Of the days when we were young
Singing at the top of both our lungs
On the day we fell in love
On the day we fell in love

Dancing on the hood
In the middle of the woods
Of an old Mustang
Where we sang
Songs with all our childhood friends
Oh, now

Oops, I got 99 problems singing bye, bye, bye
Hold up, if you wanna go and take a ride with me
Better hit me, baby, one more time
Paint a picture for you and me
Of the days when we were young, uh
Singing at the top of both our lungs
On the day we fell in love
On the day we fell in love
On the day we fell in love
On the day we fell in love
On the day we fell in love, love, love

"""
if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np

TARGET_DIR = "C:/Project505/dataset"
IMAGES_DIR = os.path.join(TARGET_DIR, "images")
MASKS_DIR = os.path.join(TARGET_DIR, "masks")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

print("🎨 Constructing 3-Channel Multi-Class Retinal Training Pairs...")

for i in range(1, 21):
    filename = f"IDRiD_{i:02d}.jpg"
    
    # Create background canvas: 512x512 matrix
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Initialize a 3-channel ground truth mask matrix matching the image size
    # Channel 0: Hemorrhages, Channel 1: Exudates, Channel 2: Microaneurysms
    mask_3ch = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Base eye profile sphere circle
    cv2.circle(img, (256, 256), 240, (15, 60, 200), -1) 
    
    # Background baseline vessels
    for j in range(5):
        start_point = (256 + np.random.randint(-30, 30), 256 + np.random.randint(-30, 30))
                # FIXED: Extract individual tuple coordinate positions using indices [0] and [1]
        end_point = (start_point[0] + np.random.randint(-180, 180), start_point[1] + np.random.randint(-180, 180))

        cv2.line(img, start_point, end_point, (10, 20, 100), thickness=np.random.randint(2, 5))

    # Seed Type 1: Hemorrhages (Marked on Mask Channel 0)
    for _ in range(np.random.randint(3, 8)):
        cx, cy, r = np.random.randint(100, 412), np.random.randint(100, 412), np.random.randint(6, 12)
        cv2.circle(mask_3ch, (cx, cy), r, (255, 0, 0), -1) # Red channel tracker
        cv2.circle(img, (cx, cy), r, (5, 5, 40), -1)

    # Seed Type 2: Hard Exudates (Marked on Mask Channel 1)
    for _ in range(np.random.randint(3, 8)):
        cx, cy, r = np.random.randint(100, 412), np.random.randint(100, 412), np.random.randint(4, 10)
        cv2.circle(mask_3ch, (cx, cy), r, (0, 255, 0), -1) # Green channel tracker
        cv2.circle(img, (cx, cy), r, (200, 230, 255), -1)

    # Seed Type 3: Microaneurysms (Marked on Mask Channel 2)
    for _ in range(np.random.randint(5, 12)):
        cx, cy, r = np.random.randint(100, 412), np.random.randint(100, 412), np.random.randint(2, 5)
        cv2.circle(mask_3ch, (cx, cy), r, (0, 0, 255), -1) # Blue channel tracker
        cv2.circle(img, (cx, cy), r, (10, 10, 80), -1)

    cv2.imwrite(os.path.join(IMAGES_DIR, filename), img)
    cv2.imwrite(os.path.join(MASKS_DIR, filename), mask_3ch)

print(f"🎉 Complete! 20 multi-class targets successfully written to disk.")

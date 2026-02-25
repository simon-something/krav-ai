import cv2
import numpy as np
import os

os.makedirs("tests/fixtures", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("tests/fixtures/test_strike.avi", fourcc, 30.0, (640, 480))
for i in range(90):  # 3 seconds at 30fps
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x = int(100 + i * 4)
    cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
    out.write(frame)
out.release()

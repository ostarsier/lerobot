import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_camera_window(title, initial_size=(480, 640)):
    """为每个摄像头创建独立的窗口"""
    plt.ion()
    fig = plt.figure(figsize=(initial_size[1]/100, initial_size[0]/100))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_title(title)
    img = ax.imshow(np.zeros((initial_size[0], initial_size[1], 3), dtype=np.uint8))
    return fig, ax, img

def update_camera_window(img, frame):
    """更新相机窗口的图像"""
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.set_data(frame_rgb)
    plt.draw()
    plt.pause(0.001)
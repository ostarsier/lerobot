import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
from queue import Queue

class CameraFeed:
    """
    从单个摄像头获取数据的类。
    """
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.frame_queue = Queue(maxsize=1)  # 用于在线程之间传递帧
        self.is_running = False
        self.thread = None

        if not self.cap.isOpened():
            print(f"无法打开摄像头 (索引 {self.camera_index})")
            return

    def start(self):
        """启动摄像头数据获取线程。"""
        if not self.is_running and self.cap.isOpened():
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()

    def stop(self):
        """停止摄像头数据获取线程并释放摄像头。"""
        if self.is_running:
            self.is_running = False
            if self.thread and self.thread.is_alive():
                self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
            print(f"摄像头 (索引 {self.camera_index}) 已释放")

    def _capture_frames(self):
        """内部方法，循环捕获帧并将最新帧放入队列。"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)  # 非阻塞地放入队列
                except queue.Full:
                    pass  # 如果队列已满，丢弃旧帧
            else:
                print(f"无法从摄像头 (索引 {self.camera_index}) 读取帧，停止捕获。")
                self.stop()
                break
            time.sleep(0.01)  # 可以调整捕获频率

    def get_latest_frame(self):
        """获取最新的捕获到的帧，如果没有则返回 None。"""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None

class FrameDisplay(tk.Toplevel):
    """
    用于显示单个摄像头帧的 Tkinter 窗口。
    """
    def __init__(self, parent, camera_name, camera_feed):
        super().__init__(parent)
        self.title(f"Camera Feed - {camera_name}")
        self.camera_name = camera_name
        self.camera_feed = camera_feed
        self.image_label = tk.Label(self)
        self.image_label.pack()
        self.is_running = True
        self.update_frame()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        if not self.is_running:
            return

        frame = self.camera_feed.get_latest_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        self.after(10, self.update_frame)

    def on_closing(self):
        print(f"关闭摄像头窗口: {self.camera_name}")
        self.is_running = False
        self.destroy()

def capture_multiple_cameras_separated(camera_config):
    """
    将摄像头数据获取和显示分离，使用线程和队列。

    Args:
        camera_config (dict): 一个字典，键是摄像头的名称（字符串），
                                值是摄像头的索引（整数，通常从 0 开始）。
    """
    root = tk.Tk()
    root.title("Multiple Cameras")
    root.withdraw()  # 隐藏主窗口
    camera_feeds = {}
    displays = {}

    for name, index in camera_config.items():
        feed = CameraFeed(index)
        camera_feeds[name] = feed
        if feed.cap.isOpened():
            feed.start()
            display = FrameDisplay(root, name, feed)
            displays[name] = display

    root.mainloop()

if __name__ == "__main__":
    import queue  # 需要显式导入 queue 模块

    # 定义要使用的摄像头及其索引
    camera_config = {
        "Front Camera": 0,
        "External Camera": 2,
        # 可以添加更多摄像头
    }
    capture_multiple_cameras_separated(camera_config)
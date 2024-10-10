import threading
import os

import cv2
from ultralytics import YOLO
import socket
import time
import numpy as np
from collections import deque
# from my_tracker import ParticleFilter


class VideoReader(threading.Thread):
    def __init__(self, video_path, fps=30):
        super(VideoReader, self).__init__()
        self.cap = cv2.VideoCapture(video_path)
        self.fps = fps
        self.last_frame = None
        self.stopped = False
        self.time = 0

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break

            self.time += 1
            self.last_frame = frame
            # cv2.imshow("video", frame)
            cv2.waitKey(int(1000 / self.fps))

    def stop(self):
        self.stopped = True
        self.cap.release()

class VideoReaderEthernet(threading.Thread):
    def __init__(self, fps=30):
        super(VideoReaderEthernet, self).__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('10.0.0.120', 12345))  # Alıcı bilgisayarın Ethernet IP adresi ve portu
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.fps = fps
        self.last_frame = None
        self.stopped = False
        self.time = 0

    def run(self):
        while not self.stopped:
            data, addr = self.sock.recvfrom(65536)
            frame = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            # if frame is not None:
            #     self.stopped = True
            #     break

            self.time += 1
            self.last_frame = frame
            # cv2.imshow("video", frame)
            cv2.waitKey(int(1000 / self.fps))

    def stop(self):
        self.stopped = True
        self.sock.close()

class YOLOThread(threading.Thread):
    def __init__(self, model, fps=1):
        super(YOLOThread, self).__init__()
        self.model = model
        self.fps = fps
        self.last_result = None
        self.stopped = False
        self.frame = None
        self.time = None
        self.result_time = None
        self.result_frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame, current_time):
        with self.lock:
            self.frame = np.copy(frame)
            self.time = current_time

    def run(self):
        while not self.stopped:
            if self.frame is not None:
                with self.lock:
                    frame_copy = np.copy(self.frame)
                    time_copy = self.time
                start_time = time.time()
                results = self.model(frame_copy, verbose=False)
                elapsed_time = time.time() - start_time

                with self.lock:
                    self.result_frame = frame_copy
                    self.result_time = time_copy
                    self.last_result = results[0]

                print("yolo fps", 1 / elapsed_time)

            time.sleep(1 / self.fps)

    def stop(self):
        self.stopped = True


def euclidean_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)


def get_frames_since(frame_buffer, start_time):
    """Retrieve all frames from the buffer since start_time."""
    return [frame for t, frame in frame_buffer if t >= start_time]


def main():
    video_reader_fps = 20
    yolo_fps = 3
    n = 0
    skip_yolo = 5
    conf_th = 0.6
    skip_yolo = max(skip_yolo, video_reader_fps // yolo_fps)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_file = 'output_video.mp4'
    # frame_size = (848, 480)
    # out = cv2.VideoWriter(output_file, fourcc, video_reader_fps, frame_size)
    video_reader = VideoReader(r"C:\Users\Alihan\Downloads\WhatsApp Video 2024-09-07 at 15.33.31.mp4", fps=video_reader_fps)
    # video_reader = VideoReaderEthernet(video_reader_fps)
    model = YOLO("midback.pt")
    yolo_thread = YOLOThread(model, fps=yolo_fps)
    video_reader.start()
    yolo_thread.start()

    tracker = None
    tracker_model1 = cv2.TrackerKCF_create
    tracker_model2 = cv2.TrackerCSRT_create
    # tracker_model3 = ParticleFilter

    frame_buffer = deque(maxlen=10)
    lost_tracking = True
    success = False
    x, y, w, h = 0, 0, 0, 0

    while not video_reader.stopped:

        frame = video_reader.last_frame
        video_time = video_reader.time
        result = np.copy(frame)

        if frame is None:
            continue

        frame_buffer.append((video_time, frame))
        yolo_thread.set_frame(frame, video_time)
        if yolo_thread.last_result is not None:
            if lost_tracking or skip_yolo < n:
                n = 0
                bboxes = yolo_thread.last_result.boxes.xywh.cpu().numpy()
                confidences = yolo_thread.last_result.boxes.conf.cpu().numpy()
                if len(bboxes) > 0:
                    highest_conf_index = 0  # confidences.argmax()
                    highest_conf_bbox = bboxes[highest_conf_index]
                    highest_conf_value = confidences[highest_conf_index]
                    if highest_conf_value > conf_th:
                        cx, cy, w, h = map(int, highest_conf_bbox)
                        x1 = round(cx - w / 2)
                        y1 = round(cy - h / 2)
                        tracker = tracker_model2()
                        tracker.init(yolo_thread.result_frame, (x1, y1, w, h))
                        lost_tracking = False
                        for buffered_frame in get_frames_since(frame_buffer, yolo_thread.result_time):
                            success, bbox = tracker.update(buffered_frame)
                            if not success:
                                lost_tracking = True
                                break
            else:
                n += 1
                success, bbox = tracker.update(result)

        if success:
            x, y, w, h = map(int, bbox)
            cx = round(x - w / 2)
            cy = round(y - h / 2)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)


        else:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("YOLO Detection and Tracking", result)
        # out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # out.release()
    video_reader.stop()
    yolo_thread.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

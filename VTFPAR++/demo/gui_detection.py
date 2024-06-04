import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from test import *
def make_square(img):
    # 获取图像的高度和宽度
    h, w, _ = img.shape

    # 计算填充后的图像的大小
    size = max(h, w)

    # 创建一个新的黑色背景图像
    square_img = np.zeros((size, size, 3), dtype=np.uint8)

    # 计算将图像放置在中心位置的坐标
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    # 将图像放置在中心位置
    square_img[y_offset:y_offset + h, x_offset:x_offset + w, :] = img

    square_img=cv2.resize(square_img,(256,256))
    return square_img

def detect_people_yolo(frame, net, output_folder):
    height, width, channels = frame.shape

    # 将图像传递给模型进行前向传播
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 解析模型输出，提取行人位置
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 corresponds to the class 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 非最大值抑制，确保每个行人只被检测一次
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 截取帧图像并保存到专属文件夹
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        crop_img = frame[y:y+h, x:x+w]
        crop_img=make_square(crop_img)
        if crop_img.size!=0:
            # 获取行人的标识符
            person_id = f"person_{i}"

            # 检查该行人是否已经有专属文件夹，没有则创建
            # if person_id not in person_folders:
            #     person_folder = os.path.join(output_folder, person_id)
            #     os.makedirs(person_folder, exist_ok=True)
            #     person_folders[person_id] = person_folder

            # 保存截图到专属文件夹
            img_filename = f"snapshot_{len(os.listdir(output_folder)) + 1}.jpg"
            img_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(img_path, crop_img)

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    video_path.set(file_path)
    cap = cv2.VideoCapture(file_path)
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=image)
    video_label.config(image=photo)
    video_label.image = photo
    cap.release()

def start_detection():
    video_file = video_path.get()
    if not video_file:
        return

    cap = cv2.VideoCapture(video_file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        detect_people_yolo(frame, net, output_folder)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        video_label.config(image=photo)
        video_label.image = photo
        window.update_idletasks()
        window.update()

    cap.release()
    #window.destroy()

attr_words = [
    'top short', #top length 0
    'bottom short', #bottom length 1
    'shoulder bag','backpack',#shoulder bag #backpack 2 3
    'hat', 'hand bag', 'long hair', 'female',# hat/hand bag/hair/gender 4 5 6 7
    'bottom skirt', #bottom type 8
    'frontal', 'lateral-frontal', 'lateral', 'lateral-back', 'back', 'pose varies',#pose[9:15]
    'walking', 'running','riding', 'staying', 'motion varies',#motion[15:20]
    'top black', 'top purple', 'top green', 'top blue','top gray', 'top white', 'top yellow', 'top red', 'top complex',#top color [20 :29]
    'bottom white','bottom purple', 'bottom black', 'bottom green', 'bottom gray', 'bottom pink', 'bottom yellow','bottom blue', 'bottom brown', 'bottom complex',#bottom color[29:39]
    'young', 'teenager', 'adult', 'old'#age[39:43]
]
index_list=[0,1,2,3,4,5,6,7,8,9,15,20,29,39,43]
group="top length, bottom length, shoulder bag, backpack, hat, hand bag, hair, gender, bottom type, pose, motion, top color, bottom color, age"

def update_text():
    result=main()
    text_widget.delete(1.0,tk.END)
    result=torch.sigmoid(result).squeeze()>0.45
    print(result)
    for i,item in enumerate(result):
        if item==True:
            print(item)
            if i in index_list:
                text_widget.insert(tk.END,f"{attr_words[i]},")

# 创建GUI窗口
window = tk.Tk()
window.title("Pedestrian Detection Demo")
window.geometry('640x320')

# 加载YOLO模型
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 创建存储截图的主文件夹
output_folder = "person_snapshots"
os.makedirs(output_folder, exist_ok=True)

# 用于跟踪每个人的专属文件夹
person_folders = {}

# 创建GUI组件
video_path = tk.StringVar()
video_label = tk.Label(window)
video_label.pack()

video_button = tk.Button(window, text="Select Video", command=select_video)
video_button.config(height=2,width=20)
video_button.pack()

start_button = tk.Button(window, text="Start Detection", command=start_detection)
start_button.config(height=2,width=20)
start_button.pack()


text_widget = tk.Text(window, wrap=tk.WORD, width=60, height=5)
text_widget.pack(pady=10)

btn=tk.Button(window,text='生成结果',command=update_text)
btn.pack()


window.mainloop()

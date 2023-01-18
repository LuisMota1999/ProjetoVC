import os
import subprocess
from roboflow import Roboflow

"""
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_gpu.txt

-> Download Dataset to 'data/'
        

-> Make sure to create a 'data/custom-data.yaml' file such as follow:
        train: ./data/train/images
        val: ./data/valid/images
        test: ./data/test/images
        
        nc: 7
        names: ["ball","crowd","goal","goalkeeper","player","stick"]
            
-> Update 'cfg/training/yolov7-custom.yaml' according to number of classes of new dataset:
        nc: 7  # number of classes
        (...)

# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
# wget -P /yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

-> After Training Successfull:
        COPY: yolov7/runs/train/yolov7-custom/weights/best.pt
        TO: yolov7/
        RENAME: yolov7/yolov7_custom.pt

"""

DEF_EPOCHS = 100
DEF_BATCH = 32
DEF_RELEASE = "yolov7.pt"
DEF_WORKERS = 1
DEF_CONFIDENCE = 0.3

yolov7_image_size_map = {
    "yolov7.pt": 640,
    "yolov7x.pt": 640,
    "yolov7-w6.pt": 1280,
    "yolov7-e6.pt": 1280,
    "yolov7-d6.pt": 1280,
    "yolov7-e6e.pt": 1280,
}


def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


print("\n====================== YOLOv7 ======================\n")
yolov7_dir = 'yolov7/'
if not (os.path.exists(yolov7_dir) and os.path.isdir(yolov7_dir)):
    print(f">>> '{yolov7_dir}' directory not found."
          f"\n\tPlease, download it as .zip an extract it to the root of this project."
          f"\n\t(https://github.com/WongKinYiu/yolov7)"
          f"\n>>> Downloaded?\n\t Install Dependencies! 'pip install -r requirements.txt'")
    exit(-1)
else:
    print("Cloning -> [PASSED]")

print("\n====================== DATASET ======================\n")
dataset_dir = yolov7_dir + "data"
if not (os.path.exists(yolov7_dir + "data/train") and os.path.exists(yolov7_dir + "data/test") and
        os.path.exists(yolov7_dir + "data/valid")):
    print("-> Not Found")

    rf = Roboflow(api_key="YTXhfQI7gXmSKksYA721")
    project = rf.workspace("visao-computacional").project("roller-hockey")
    dataset = project.version(4).download("yolov7")
    print("Move 'train', 'val' and 'test' folders to 'yolov7/data/'")

if "custom-data.yaml" not in os.listdir(dataset_dir):
    print(f"[ERROR]\t'custom-data.yaml' not found in '{dataset_dir}'\n\t\tManually Create It!")
    exit(-1)
else:
    print("Download + Custom .yaml -> [PASSED]")

print("\n====================== TRAINING ======================\n")
# ["yolov7.pt", "yolov7x.pt", "yolov7-w6.pt", "yolov7-e6.pt", "yolov7-d6.pt", "yolov7-e6e.pt"]
yolov7_releases = [f_path for f_path in os.listdir(yolov7_dir) if f_path.endswith(".pt") and "yolov7" in f_path]

if not any(item in os.listdir(yolov7_dir) for item in yolov7_releases):
    print("YOLOv7 Releases NOT Found!")
    exit(-1)

# ------------------------------------------------------------------------------------ Input Release
while True:
    try:
        release = input("\033[1mEnter Release Version:\033[0m\n")
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    else:
        if release == "":
            print(f"\tEmpty Input!\n\t-> Setting Default: {DEF_RELEASE}\n")
            release = DEF_RELEASE
            break
        if release not in yolov7_releases:
            print("\t[ERROR] Invalid Release!")
            print(f"\t Available Releases: {yolov7_releases}")
            print(f"\t Release Exists But Not Showing? Download it to this directory!\n")
            continue
        elif not any(item in os.listdir(yolov7_dir) for item in yolov7_releases):
            print("\tYOLOv7 Release NOT Found in Directory!")
        break
PIXELS = yolov7_image_size_map.get(release, 640)

# ------------------------------------------------------------------------------------ Batch Size
while True:
    try:
        batch = int(input("\033[1mEnter Batch Size:\033[0m\n"))
    except ValueError:
        print(f"\tSorry, I didn't understand that.\n\t-> Setting Default: {DEF_BATCH}\n")
        batch = DEF_BATCH
        break
        # continue
    else:
        if batch < 16:
            print("\t[ERROR] Invalid!")
            print(f"\t Batch Size >= 16!")
            continue
        break

# ------------------------------------------------------------------------------------ Num. Epochs
while True:
    try:
        epochs = int(input("\033[1mEnter Number of Epochs:\033[0m\n"))
    except ValueError:
        print(f"\tSorry, I didn't understand that.\n\t-> Setting Default: {DEF_EPOCHS}\n")
        epochs = DEF_EPOCHS
        break
        # continue
    else:
        if epochs < 1:
            print("\t[ERROR] Invalid!")
            print(f"\t Number of Epochs >= 1!")
            continue
        break

# ------------------------------------------------------------------------------------ Num. Workers
while True:
    try:
        workers = int(input("\033[1mEnter Number of Workers:\033[0m\n"))
    except ValueError:
        print(f"\tSorry, I didn't understand that.\n\t-> Setting Default: {DEF_WORKERS}\n")
        workers = DEF_WORKERS
        break
        # continue
    else:
        if workers < 1:
            print("\t[ERROR] Invalid!")
            print(f"\t Number of workers >= 1!")
            continue
        break

# ------------------------------------------------------------------------------------ Confidence
while True:
    try:
        conf = int(input("\033[1mEnter Confidence Value:\033[0m\n"))
    except ValueError:
        print(f"\tSorry, I didn't understand that.\n\t-> Setting Default: {DEF_CONFIDENCE}\n")
        conf = DEF_CONFIDENCE
        break
        # continue
    else:
        if conf > 1 or conf < 0:
            print("\t[ERROR] Invalid!")
            print(f"\t Confidence must be between 0 and 1!")
            continue
        break

print(f"cd {yolov7_dir}")
print("\n>>>>>\tTRAINING COMMAND\t<<<<<")
print(
    f"python train.py --workers {workers} --device 0 --batch-size {batch} --epochs {epochs} --img {PIXELS} {PIXELS} --data data/custom-data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7_custom --weights {release}")

print(
    "\t-> After Training Successfull... \n\t-> COPY: yolov7/runs/train/yolov7-custom/weights/best.pt\n\t-> TO: yolov7/\n\t-> RENAME 'yolov7/best.pt' TO: 'yolov7/yolov7_custom.pt'")

print("\n>>>>>\tIMAGE INFERENCE COMMAND\t<<<<<")
print("\t->  Single Image:")
print(
    f"python detect.py --weights yolov7_custom.pt --conf {conf} --img-size {PIXELS} --source REPLACE_WITH_PATH_TO_IMAGE.jpg")
print("\t->  Test Folder:")
print(f"python detect.py --weights yolov7_custom.pt --conf {conf} --source data/test/images")
print("\t->  Test Videos:")
print(
    f"python detect.py --weights yolov7_custom.pt --conf {conf} --img-size {PIXELS} --source videos/test.mp4 --name video_inference")
print("\t-> Results At: yolov7/runs/detect/exp/")

print("\n>>>>>\tTESTING COMMAND\t<<<<<")
print(
    f"python test.py --data data/custom-data.yaml --img {PIXELS} --batch {batch} --conf 0.001 --iou 0.65 --device 0 --weights yolov7_custom.pt --name yolov7_custom_testing")

import torch

torch.cuda.empty_cache()

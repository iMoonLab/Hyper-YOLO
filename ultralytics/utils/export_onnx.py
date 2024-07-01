import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    model = 'hyper-yolos.pt'
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    filename = model.export(imgsz=640, format='ONNX', half=True, int8=False, device='0', verbose=False)


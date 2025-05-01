import cv2, numpy as np

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL    = "MobileNetSSD_deploy.caffemodel"
CLASSES  = ["background","aeroplane","bicycle","bird","boat",
            "bottle","bus","car","cat","chair","cow","diningtable",
            "dog","horse","motorbike","person","pottedplant",
            "sheep","sofa","train","tvmonitor"]

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
    raise RuntimeError("Cannot open video file")

frame_idx = 0
found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # resize + blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    # collect “person” confidences
    confs = [
        float(detections[0,0,i,2])
        for i in range(detections.shape[2])
        if int(detections[0,0,i,1]) == CLASSES.index("person")
    ]

    if confs:
        print(f"Detected person(s) in frame {frame_idx}, top scores: {confs[:5]}")
        found = True
        break

if not found:
    print("No people found in any frame up to frame", frame_idx)
cap.release()

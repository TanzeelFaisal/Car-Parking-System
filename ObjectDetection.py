import copy

import cv2
from ultralytics import YOLO


model = YOLO("weights/best.pt")
drawRect = []                           # positions to draw rect where mouse clicked

click_count = 0
start_pos = None


def eventmousebutton(event, x, y, flags, params):
    global click_count, start_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count == 0:
            start_pos = (x, y)
            click_count += 1
        elif click_count == 1:
            end_pos = (x, y)
            drawRect.append((start_pos[0], start_pos[1], end_pos[0], end_pos[1]))
            click_count = 0
    elif event == cv2.EVENT_RBUTTONDOWN:
        k = 0
        for x1, y1, x2, y2 in drawRect:
            y_check = y1 <= y <= y2 or y2 <= y <= y1
            x_check = x1 <= x <= x2 or x2 <= x <= x1
            if x_check and y_check:
                drawRect.pop(k)
            k = k + 1


# Video Capture Detection
cap = cv2.VideoCapture('inference/videos/vid4.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")
_, img = cap.read()
img = cv2.resize(img, (640, 480))
temp = copy.deepcopy(img)
while True:
    img = copy.deepcopy(temp)
    for x1, y1, x2, y2 in drawRect:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Parkings", img)
    cv2.setMouseCallback("Parkings", eventmousebutton)
    if cv2.waitKey(25) & 0xFF == ord(' '):
        cv2.destroyAllWindows()
        break  # 1ms response delay


print("playing video")
cap = cv2.VideoCapture('inference/videos/vid.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

while cap.isOpened():
    occupied_count = 0
    ret, inputs = cap.read()
    if ret == True:
        inputs = cv2.resize(inputs, (640, 480))
        for x1, y1, x2, y2 in drawRect:
            cv2.rectangle(inputs, (x1, y1), (x2, y2), (0, 255, 0), 2)
        results = model(inputs, conf=0.5, stream=True)
        for result in results:
            boxes = result.boxes.data.numpy()
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                c_x = (x1+x2)//2
                c_y = (y1+y2)//2
                conf = float(box[4])
                class_id = int(box[5])
                st_point = (x1, y1)
                end_point = (x2,y2)
                center = (c_x, c_y)
                color = (0, 255, 0)
                cv2.circle(inputs, center, 2, color,2)
                for x1, y1, x2, y2 in drawRect:
                    y_check = y1 <= c_y <= y2 or y2 <= c_y <= y1
                    x_check = x1 <= c_x <= x2 or x2 <= c_x <= x1
                    if x_check and y_check:
                        occupied_count += 1
                        cv2.circle(inputs, center, 4, (0,0,255), 4)
                        cv2.rectangle(inputs, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        break

            countmsg = str(occupied_count) + " Occupied out of " + str(len(drawRect)) + " Spaces"
            cv2.putText(inputs, countmsg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Frame', inputs)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord(' '):
            break

    # Break the loop
    else:
        break


cap.release()
cv2.destroyAllWindows()


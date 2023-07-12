import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# 指定手部追蹤模型
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# 使用drawing_utils函示畫出偵測到的點座標
mpDraw = mp.solutions.drawing_utils

# 設定點和連結的樣式
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=10)
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)

pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        # 轉換BRG至RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        # 設定視窗的高度和寬度
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)

                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 35, (0, 0, 255), cv2.FILLED)
                    print(i, xPos, yPos)
        # 設定fps
        cTime = time.time()
        fps = 1/(cTime - pTime) 
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
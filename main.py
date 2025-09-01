import cv2
import mediapipe as mp
import serial
import time

# Conexión con Arduino (cambia COM4 por el que te salga en el administrador de dispositivos)
arduino = serial.Serial('COM3', 9600)
time.sleep(2)

# Inicialización de MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h)])

            # Bounding box (cuadro gris)
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (150, 150, 150), 2)

            # ---- Detección de dedos ----
            fingers = [0, 0, 0, 0, 0]  # pulgar, índice, medio, anular, meñique

            # Pulgar (depende de la orientación, este ejemplo es mano derecha)
            if lmList[4][1] > lmList[3][1]:
                fingers[0] = 1

            # Índice, medio, anular, meñique
            tipIds = [8, 12, 16, 20]
            for i, id in enumerate(tipIds):
                if lmList[id][2] < lmList[id - 2][2]:
                    fingers[i + 1] = 1

            # Enviar datos al Arduino en formato binario tipo "01001"
            fingers_str = "".join(map(str, fingers))
            arduino.write((fingers_str + "\n").encode())

            # Mostrar en pantalla qué dedos están arriba
            cv2.putText(frame, f"Dedos: {fingers_str}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("LedByHand", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()

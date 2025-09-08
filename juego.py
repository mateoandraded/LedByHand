import cv2
import mediapipe as mp
import serial
import time
import random

# Conexión con Arduino (ajusta el puerto COM según corresponda)
arduino = serial.Serial('COM3', 9600)
time.sleep(2)

# Inicialización de MediaPipe (misma conf que tu main)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("http://192.168.68.56:4747/video")


# =========================
# Funciones auxiliares
# =========================

def mostrar_inicio():
    """Parpadeo inicial de todos los LEDs dos veces"""
    for _ in range(2):
        arduino.write(("11111\n").encode())  # todos encendidos
        time.sleep(0.5)
        arduino.write(("00000\n").encode())  # todos apagados
        time.sleep(1)

def generar_secuencia(n=5):
    """Genera una lista de secuencias de UN SOLO dedo por paso"""
    secuencia = []
    for _ in range(n):
        dedo = random.randint(0, 4)  # índice del dedo (0 a 4)
        paso = ["0"] * 5
        paso[dedo] = "1"
        secuencia.append("".join(paso))
    return secuencia

def mostrar_secuencia(secuencia):
    """Arduino muestra la secuencia lentamente"""
    for paso in secuencia:
        arduino.write((paso + "\n").encode())
        time.sleep(1)
        arduino.write(("00000\n").encode())
        time.sleep(0.5)

def leer_dedos(lmList):
    """Devuelve string binaria de dedos levantados"""
    fingers = [0, 0, 0, 0, 0]  # [pulgar, índice, medio, anular, meñique]

    # Pulgar (mano derecha; si es izquierda hay que invertir signo)
    if lmList[4][1] > lmList[3][1]:
        fingers[0] = 1

    # Índice, medio, anular, meñique
    tipIds = [8, 12, 16, 20]
    for i, id in enumerate(tipIds):
        if lmList[id][2] < lmList[id - 2][2]:
            fingers[i + 1] = 1

    return "".join(map(str, fingers))

def mostrar_ganador():
    """Animación de victoria"""
    for i in range(5):
        paso = ["0"] * 5
        paso[i] = "1"
        arduino.write(("".join(paso) + "\n").encode())
        time.sleep(0.4)
    arduino.write(("00000\n").encode())

def mostrar_error():
    """Animación de error (parpadeo doble)"""
    for _ in range(2):
        arduino.write(("11111\n").encode())
        time.sleep(0.5)
        arduino.write(("00000\n").encode())
        time.sleep(0.5)

# =========================
# Juego principal
# =========================

mostrar_inicio()
secuencia = generar_secuencia(5)
print("Secuencia generada:", secuencia)

mostrar_secuencia(secuencia)

intento = []
estado_anterior = "00000"  # estado previo de la mano
print("👉 Repite la secuencia con tu mano...")

while len(intento) < len(secuencia):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append([id, int(lm.x * w), int(lm.y * h)])

            dedos = leer_dedos(lmList)

            # Mostrar en pantalla el binario detectado
            cv2.putText(frame, dedos, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)

            # Solo registrar si cambia de estado y no es todo apagado
            if dedos != estado_anterior and dedos != "00000":
                intento.append(dedos)
                estado_anterior = dedos
                print("Intento:", intento)
                time.sleep(0.8)  # anti rebotes

    cv2.imshow("Juego LedByHand", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()

# =========================
# Validar intento
# =========================
if intento == secuencia:
    print("✅ Ganaste!")
    mostrar_ganador()
else:
    print("❌ Fallaste!")
    mostrar_error()

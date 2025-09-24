import cv2
import mediapipe as mp
import serial
import time
import random

# -------------------
# Configuración Arduino y cámara
# -------------------
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600
VIDEO_SOURCE = 0  # cámara por defecto

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hand_landmark_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=1)
hand_connection_style = mp_draw.DrawingSpec(color=(128, 0, 128), thickness=2)

# Conexión Arduino
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)
    print(f"Conectado a Arduino en {ARDUINO_PORT}")
except:
    print("No se pudo conectar a Arduino")
    arduino = None

# Inicializar detector de manos
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Inicializar cámara
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------
# Logos y overlays
# -------------------
scale_factor = 1
logo = cv2.imread("niot_logo.png", cv2.IMREAD_UNCHANGED)
if logo is not None:
    logo = cv2.resize(logo, (350, 240))
foto = cv2.imread("espol.png", cv2.IMREAD_UNCHANGED)
if foto is not None:
    foto = cv2.resize(foto, (190, 100))

# Función para poner overlay con transparencia
def poner_overlay(base, overlay, offset_x=0, offset_y=0, esquina='derecha'):
    if overlay is None or overlay.shape[2] != 4:
        return base
    h_ol, w_ol = overlay.shape[:2]
    if esquina == 'derecha':
        x1 = base.shape[1] - w_ol - offset_x
    else:
        x1 = offset_x
    x2 = x1 + w_ol
    y2 = base.shape[0] - offset_y
    y1 = y2 - h_ol

    # Validar límites
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, base.shape[1])
    y2 = min(y2, base.shape[0])

    roi = base[y1:y2, x1:x2]
    overlay_rgb = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    # Ajustar tamaño si es necesario
    if roi.shape[:2] != overlay_rgb.shape[:2]:
        overlay_rgb = cv2.resize(overlay_rgb, (roi.shape[1], roi.shape[0]))
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

    base[y1:y2, x1:x2] = (roi * (1 - mask) + overlay_rgb * mask).astype("uint8")
    return base

# -------------------
# Variables del juego
# -------------------
COLUMN_X = [200, 400, 600, 800, 1000]  # posiciones de los LEDs en pantalla
ZONE_Y = 600
NOTE_SPEED = 2
score = 0
notas = []
vidas = 5  # NUEVO: sistema de vidas

last_note_time = time.time()
note_interval = 2  # 1 nota por segundo

# -------------------
# Funciones del juego
# -------------------
def generar_nota():
    col = random.randint(0, 4)
    notas.append({"col": col, "y": 0, "hit": False})

def enviar_a_arduino(fingers_str):
    if arduino:
        try:
            arduino.write((fingers_str + "\n").encode())
        except:
            pass

# -------------------
# Loop principal
# -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    fingers_str = "00000"

    # Detección de mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   hand_landmark_style, hand_connection_style)

            h, w, c = frame.shape
            lm_list = [[id, int(lm.x*w), int(lm.y*h)] for id, lm in enumerate(hand_landmarks.landmark)]

            # Bounding box
            x_coords = [lm[1] for lm in lm_list]
            y_coords = [lm[2] for lm in lm_list]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Detección dedos
            fingers = [0]*5
            if results.multi_handedness:
                hand_label = results.multi_handedness[0].classification[0].label
                is_right_hand = hand_label == "Right"
            else:
                is_right_hand = True

            # Pulgar
            fingers[0] = 1 if (lm_list[4][1] < lm_list[3][1] if is_right_hand else lm_list[4][1] > lm_list[3][1]) else 0
            # Otros dedos
            tip_ids = [8, 12, 16, 20]
            for i, id in enumerate(tip_ids):
                fingers[i+1] = 1 if lm_list[id][2] < lm_list[id-2][2] else 0

            fingers_str = "".join(map(str, fingers))
            if not is_right_hand:
                fingers_str = fingers_str[::-1]

            enviar_a_arduino(fingers_str)

            # Dibujar bounding box
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (128,0,128), 2)
            label = "Derecha" if is_right_hand else "Izquierda"
            cv2.putText(frame, f"Mano: {label}", (x_min-20, y_min-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,0,128), 2)

    # Generar notas
    if time.time() - last_note_time > note_interval:
        generar_nota()
        last_note_time = time.time()

    # Dibujar notas
    for nota in notas:
        nota['y'] += NOTE_SPEED
        colores = [(255,0,0),(0,0,255),(255,255,255),(0,255,255),(0,255,0)]
        color = colores[nota['col']]
        if nota['hit']:
            color = (0,255,0)
        cv2.circle(frame, (COLUMN_X[nota['col']], nota['y']), 30, color, -1)

    cv2.line(frame, (0, ZONE_Y), (1280, ZONE_Y), (0,255,255),3)

    # Revisar aciertos
    for nota in notas:
        if ZONE_Y-20 < nota['y'] < ZONE_Y+20:
            if fingers_str[nota['col']] == '1' and not nota['hit']:
                nota['hit'] = True
                score += 1

    # Verificar si alguna nota se perdió
    for nota in notas:
        if nota['y'] >= 720 and not nota['hit']:
            vidas -= 1
            notas.remove(nota)
            if vidas <= 0:
                print("Game Over")
                break

    notas = [n for n in notas if n['y'] < 720 + 50]

    # -------------------
    # Aplicar overlays
    # -------------------
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Logo inferior derecha
    resized_frame = poner_overlay(resized_frame, logo, offset_x=20, offset_y=0, esquina='derecha')
    # Foto inferior izquierda
    resized_frame = poner_overlay(resized_frame, foto, offset_x=30, offset_y=50, esquina='izquierda')

    # Score, vidas e instrucciones
    cv2.putText(resized_frame, f"Score: {score}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(resized_frame, f"Vidas: {vidas}", (50,90), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(resized_frame, "Presiona 'q' para salir", (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.imshow("Juego 1", resized_frame)

    if vidas <= 0:
        cv2.putText(resized_frame, "Perdiste :(", (500, 360), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        cv2.imshow("Juego 1", resized_frame)
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------
# Liberar recursos
# -------------------
cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()

import cv2
import time
import serial
import random
import mediapipe as mp

# ------------------------
# Configuración Arduino
# ------------------------
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)
except:
    print("No se pudo conectar con Arduino")
    arduino = None

# ------------------------
# Configuración juego
# ------------------------
colors = {
    "VERDE": (0, 255, 0),
    "ROJO": (0, 0, 255),
    "AZUL": (255, 0, 0),
    "BLANCO": (255, 255, 255),
    "AMARILLO": (0, 255, 255)
}

codes = {
    "VERDE": "00001",
    "ROJO": "01000",
    "AZUL": "10000",
    "BLANCO": "00100",
    "AMARILLO": "00010"
}

vidas = 5
score = 0
palabras_acertadas = 0
tiempo_palabra = 10  # segundos
bonus_time = 1.5    # tiempo de animación
game_duration = 60  # duración total en segundos

# ------------------------
# Inicializar cámara y MediaPipe
# ------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow("Juego de Reflejos", cv2.WINDOW_NORMAL)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ------------------------
# Funciones auxiliares
# ------------------------
def put_text_center(frame, text, y_ratio, font_scale, color, thickness):
    h, w, _ = frame.shape
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = int((w - size[0]) / 2)
    y = int(h * y_ratio)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def show_feedback(message, score_text, led_code=None, duration=bonus_time):
    # Prender LED antes de animación
    if arduino and led_code:
        arduino.write((led_code + "\n").encode())

    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # Mostrar mensaje morado grandote
        put_text_center(frame, message, 0.5, 6, (128, 0, 128), 12)
        put_text_center(frame, score_text, 0.1, 2, (255, 255, 255), 3)
        cv2.imshow("Juego de Reflejos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    # Apagar LED después de la animación
    if arduino:
        arduino.write(b"00000\n")
    return True

# ------------------------
# Loop principal
# ------------------------
game_start = time.time()
while vidas > 0 and (time.time() - game_start) < game_duration:
    color_actual = random.choice(list(colors.keys()))
    start_time = time.time()  # inicio del intento
    acertado = False

    while time.time() - start_time < tiempo_palabra:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # Mostrar palabra y score
        put_text_center(frame, color_actual, 0.5, 5, colors[color_actual], 10)
        put_text_center(frame, f"Vidas: {vidas}  Score: {score}", 0.1, 2, (255, 255, 255), 3)

        # Mostrar tiempo total restante
        tiempo_restante = max(0, int(game_duration - (time.time() - game_start)))
        put_text_center(frame, f"Tiempo Total: {tiempo_restante}s", 0.85, 2, (255, 255, 0), 3)

        # Mostrar tiempo del intento actual como cuenta regresiva
        tiempo_intento = max(0, int(tiempo_palabra - (time.time() - start_time)))
        put_text_center(frame, f"Tiempo Intento: {tiempo_intento}s", 0.92, 2, (0, 255, 255), 3)

        # Redimensionar ventana
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (w, h))

        # ------------------------
        # Detección de mano con MediaPipe
        # ------------------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        fingers_str = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm_list = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]

            fingers = [0, 0, 0, 0, 0]
            # Pulgar
            fingers[0] = 1 if lm_list[4][0] < lm_list[3][0] else 0
            # Dedos restantes
            tip_ids = [8, 12, 16, 20]
            for i, id in enumerate(tip_ids):
                fingers[i + 1] = 1 if lm_list[id][1] < lm_list[id - 2][1] else 0

            fingers_str = "".join(map(str, fingers))
            if arduino:
                arduino.write((fingers_str + "\n").encode())

        # ------------------------
        # Verificar si acertó el color
        # ------------------------
        if fingers_str == codes[color_actual]:
            acertado = True
            score += 1
            palabras_acertadas += 1
            if palabras_acertadas % 4 == 0 and tiempo_palabra > 2:  # mínimo 1 segundo
                tiempo_palabra -= 1
            if arduino:
                arduino.write(b"HIT\n")
            show_feedback("¡ACERTASTE!", f"Score: {score}", led_code=codes[color_actual], duration=bonus_time/2)
            break  # pasa a la siguiente palabra

        cv2.imshow("Juego de Reflejos", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vidas = 0
            break

    # Si no acertó
    if not acertado:
        vidas -= 1
        if arduino:
            arduino.write(b"BEEP\n")
        show_feedback("FALLASTE", f"Score: {score}", led_code="11111")  # ejemplo de LED fallo

# ------------------------
# Fin del juego
# ------------------------
if (time.time() - game_start) >= game_duration and vidas > 0:
    show_feedback("¡TIEMPO! GANASTE!", f"Score: {score}", led_code="11111")  # todos los LEDs prendidos
    if arduino:
        arduino.write(b"SUCCESS\n")
elif vidas > 0 and arduino:
    arduino.write(b"SUCCESS\n")
elif arduino:
    show_feedback("GAME OVER", f"Score: {score}", led_code="11111")
    arduino.write(b"DEFEAT\n")

cap.release()
cv2.destroyAllWindows()

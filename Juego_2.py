import cv2
import time
import serial
import random
import mediapipe as mp

heart_img = cv2.imread("corazon.png", cv2.IMREAD_UNCHANGED)
img_acertaste = cv2.imread("acertaste.png", cv2.IMREAD_UNCHANGED)
img_fallaste = cv2.imread("fallaste.png", cv2.IMREAD_UNCHANGED)
img_ganaste   = cv2.imread("ganaste.png", cv2.IMREAD_UNCHANGED)
img_gameover  = cv2.imread("gameover.png", cv2.IMREAD_UNCHANGED)
logo = cv2.imread("niot_logo.png", cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (350, 240))
foto = cv2.imread("espol.png", cv2.IMREAD_UNCHANGED)
foto = cv2.resize(foto, (190, 100))

# ------------------------
# Configuración Arduino
# ------------------------
ARDUINO_PORT = 'COM5'
BAUD_RATE = 9600
fuente = 0
fuente1 = "http://192.168.68.67:4747/video"

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
tiempo_palabra = 8  # segundos
bonus_time = 1.5
game_duration = 60

# ------------------------
# Inicializar cámara y MediaPipe
# ------------------------
cap = cv2.VideoCapture(fuente)
if not cap.isOpened():
    print("Error: no se puede abrir la cámara")
    exit()

# Resolución y enfoque mejorado
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FOCUS, 0)  # desactivar autofocus para evitar blur
cv2.namedWindow("Juego de Reflejos", cv2.WINDOW_NORMAL)

mp_hands = mp.solutions.hands
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

def send_to_arduino(msg, with_game=True):
    if arduino:
        if with_game:
            arduino.write((f"{msg},JUEGO\n").encode())
        else:
            arduino.write((msg + "\n").encode())

def overlay_watermarks(frame, logo, foto):
    h_frame, w_frame, _ = frame.shape

    # Logo: esquina inferior derecha
    if logo is not None and logo.shape[2] == 4:
        overlay = logo[..., :3]
        mask = logo[..., 3:]
        h, w = overlay.shape[:2]
        y1, y2 = h_frame - h, h_frame
        x1, x2 = w_frame - w - 20, w_frame - 20
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = (roi * (1 - mask / 255) + overlay * (mask / 255)).astype("uint8")

    # Foto: esquina inferior izquierda
    if foto is not None and foto.shape[2] == 4:
        overlay2 = foto[..., :3]
        mask2 = foto[..., 3:]
        h2, w2 = overlay2.shape[:2]
        y1, y2 = h_frame - h2 - 50, h_frame - 50
        x1, x2 = 30, 30 + w2
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = (roi * (1 - mask2 / 255) + overlay2 * (mask2 / 255)).astype("uint8")

def show_feedback(img, duration=1.5):
    """
    Muestra una imagen PNG de feedback centrada durante 'duration' segundos.
    """
    start = time.time()
    h_frame, w_frame, _ = cap.read()[1].shape  # Tomamos tamaño del frame

    h_img, w_img = img.shape[:2]
    x_offset = (w_frame - w_img) // 2
    y_offset = (h_frame - h_img) // 2

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        overlay_image(frame, img, x_offset, y_offset)
        overlay_watermarks(frame, logo, foto)
        cv2.imshow("Juego de Reflejos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True

def overlay_image(frame, img, x, y):
    """Superpone img (con alpha) sobre frame en la posición x,y"""
    h, w = img.shape[:2]
    if img.shape[2] == 4:  # tiene canal alpha
        alpha_s = img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_s * img[:, :, c] +
                                      alpha_l * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = img



def draw_hearts(frame, vidas, heart_img, x_start=50, y_start=50, scale=0.05, spacing=10):
    h_heart, w_heart = heart_img.shape[:2]
    new_w = int(w_heart * scale)
    new_h = int(h_heart * scale)
    resized_heart = cv2.resize(heart_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    for i in range(vidas):
        x = x_start + i * (new_w + spacing)
        y = y_start
        overlay_image(frame, resized_heart, x, y)

def start_animation_images(frame_cap, img_paths, countdown=5, arduino_beep=True,
                           x_offset=390, y_offset=110, tiempo_total=60):
    """
    Intro animaciones:
    - Intro 1 y 2: imágenes normales
    - Intro 3: muestra 5 corazones
    - Intro 4: muestra tiempo total + (Like para comenzar) tras 1s,
               se queda esperando pulgar arriba
    """

    estado_intro = 1
    for idx, path in enumerate(img_paths, start=1):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"No se pudo cargar la imagen: {path}")
            continue

        if arduino:
            arduino.write("PLING\n".encode())

        start_time = time.time()
        while True:
            ret, frame = frame_cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # -------------------------
            # Intro 1 y 2 → solo imágenes
            # -------------------------
            if idx in [1, 2]:
                overlay_image(frame, img, x_offset, y_offset)
                if time.time() - start_time >= 3:
                    break

            # -------------------------
            # Intro 3 → corazones
            # -------------------------
            elif idx == 3:
                overlay_image(frame, img, x_offset, y_offset)
                draw_hearts(frame, 5, heart_img, x_start=50, y_start=20,
                            scale=0.08, spacing=15)
                if time.time() - start_time >= 3:
                    break

            # -------------------------
            # Intro 4 → tiempo total + esperar Like
            # -------------------------
            elif idx == 4:
                h, w, _ = frame.shape

                draw_hearts(frame, 5, heart_img, x_start=50, y_start=20,
                            scale=0.08, spacing=15)

                tiempo_restante = tiempo_total
                text_size_total = cv2.getTextSize(f"Total: {tiempo_restante}s",
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                x_total = w - text_size_total[0] - 20
                cv2.putText(frame, f"Total: {tiempo_restante}s", (x_total, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 35, 100), 2)

                overlay_image(frame, img, x_offset, y_offset)

                if time.time() - start_time >= 1:
                    put_text_center(frame, "(Like para comenzar)", 0.95,
                                    0.8, (87, 35, 1000), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm_list = [(int(lm.x * w), int(lm.y * h))
                               for lm in hand_landmarks.landmark]

                    hand_label = results.multi_handedness[0].classification[0].label
                    is_right_hand = hand_label == "Right"

                    thumb_up = lm_list[4][1] < lm_list[3][1] and lm_list[4][1] < lm_list[2][1]
                    other_fingers_down = all(lm_list[tip][1] > lm_list[tip-2][1] for tip in [8, 12, 16, 20])

                    if thumb_up and other_fingers_down:
                        print("Pulgar arriba detectado → comienza el juego")
                        return True

            cv2.imshow("Juego de Reflejos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    return True
imagenes_intro = ["intro1.png", "intro2.png", "intro3.png", "intro4.png"]

# ------------------------
# Loop principal
# ------------------------

if not start_animation_images(cap, imagenes_intro, countdown=5):
    cap.release()
    cv2.destroyAllWindows()
    exit()

game_start = time.time()
while vidas > 0 and (time.time() - game_start) < game_duration:
    color_actual = random.choice(list(colors.keys()))
    start_time = time.time()
    acertado = False

    while time.time() - start_time < tiempo_palabra:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        put_text_center(frame, color_actual, 0.5, 5, colors[color_actual], 10)
        draw_hearts(frame, 5, heart_img, x_start=50, y_start=20,
                    scale=0.08, spacing=15)

        h, w, _ = frame.shape

        # Tiempo total esquina superior derecha
        tiempo_restante = max(0, int(game_duration - (time.time() - game_start)))
        text_size_total = cv2.getTextSize(f"Total: {tiempo_restante}s", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x_total = w - text_size_total[0] - 20
        cv2.putText(frame, f"Total: {tiempo_restante}s", (x_total, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (87, 35, 100), 2)

        # Tiempo de intento centrado debajo del color
        tiempo_intento = max(0, int(tiempo_palabra - (time.time() - start_time)))
        text_size_color = cv2.getTextSize(color_actual, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
        y_color = int(h * 0.5)
        text_size_intento = cv2.getTextSize(f"{tiempo_intento}s", cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        x_intento = int((w - text_size_intento[0]) / 2)
        y_intento = y_color + text_size_color[1] + 20
        cv2.putText(frame, f"{tiempo_intento}s", (x_intento, y_intento),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, colors[color_actual], 4)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        fingers_str = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm_list = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]

            # Detectar mano izquierda o derecha
            hand_label = results.multi_handedness[0].classification[0].label
            is_right_hand = hand_label == "Right"

            # Detección pulgar mejorada
            if is_right_hand:
                fingers = [1 if lm_list[4][0] < lm_list[3][0] else 0, 0, 0, 0, 0]
            else:
                fingers = [1 if lm_list[4][0] > lm_list[3][0] else 0, 0, 0, 0, 0]

            # Dedos restantes (índice, medio, anular, meñique)
            tip_ids = [8, 12, 16, 20]
            for i, id in enumerate(tip_ids):
                fingers[i + 1] = 1 if lm_list[id][1] < lm_list[id - 2][1] else 0

            # Invertir para mano izquierda para LEDs
            if not is_right_hand:
                fingers = fingers[::-1]

            fingers_str = "".join(map(str, fingers))
            send_to_arduino(fingers_str, with_game=True)

        if fingers_str == codes[color_actual]:
            acertado = True
            score += 1
            palabras_acertadas += 1
            if palabras_acertadas % 3 == 0 and tiempo_palabra > 2:
                tiempo_palabra -= 2
            send_to_arduino("HIT", with_game=False)
            show_feedback(img_acertaste, duration=bonus_time/2)
            break
        overlay_watermarks(frame, logo, foto)
        cv2.imshow("Juego de Reflejos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vidas = 0
            break

    if not acertado:
        vidas -= 1
        send_to_arduino("BEEP", with_game=False)
        show_feedback(img_fallaste)

# ------------------------
# Fin del juego
# ------------------------
if (time.time() - game_start) >= game_duration and vidas > 0:
    show_feedback(img_ganaste)
    send_to_arduino("SUCCESS", with_game=False)
elif vidas > 0:
    send_to_arduino("SUCCESS", with_game=False)
else:
    show_feedback(img_gameover)
    send_to_arduino("DEFEAT", with_game=False)

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import serial
import time


# Configuración inicial
ARDUINO_PORT = 'COM5'
BAUD_RATE = 9600
VIDEO_SOURCE1 = "http://192.168.68.62:4747/video"
VIDEO_SOURCE = 0
# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Estilos para dibujar los landmarks (puntos blancos y líneas moradas finas)
hand_landmark_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=1)
hand_connection_style = mp_draw.DrawingSpec(color=(128, 0, 128), thickness=2)

try:
    # Conexión con Arduino
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)  # Esperar a que se establezca la conexión
    print(f"Conexión establecida con Arduino en {ARDUINO_PORT}")
except serial.SerialException:
    print(f"Error: No se pudo conectar con Arduino en {ARDUINO_PORT}")
    arduino = None

# Inicializar detección de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Configurar cámara para mejor calidad
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

# Configurar resolución y calidad de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FOCUS, 0)  # Desactivar autofocus para evitar blur

# Variables para controlar la frecuencia de envío
last_send_time = 0
send_interval = 0.1  # Enviar datos cada 100ms

# Factor de escala para reducir el tamaño de la ventana (0.0 a 1.0)
scale_factor = 1  # Reduce a 70% del tamaño original
logo = cv2.imread("niot_logo.png", cv2.IMREAD_UNCHANGED)  # Lee PNG con transparencia
logo = cv2.resize(logo, (350, 240))  # Ajusta tamaño de la marca de agua
foto = cv2.imread("espol.png", cv2.IMREAD_UNCHANGED)
foto = cv2.resize(foto, (190 ,100))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede recibir el frame")
        break

    # Voltear frame horizontalmente para una experiencia espejo
    frame = cv2.flip(frame, 1)

    # Convertir a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar landmarks y conexiones con nuevos estilos
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmark_style,
                hand_connection_style
            )

            # Obtener dimensiones del frame
            h, w, c = frame.shape
            lm_list = []

            # Convertir landmarks a coordenadas de píxeles
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append([id, int(lm.x * w), int(lm.y * h)])

            # Calcular bounding box
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Detección de dedos
            fingers = [0, 0, 0, 0, 0]  # pulgar, índice, medio, anular, meñique

            # Detectar mano izquierda o derecha
            if results.multi_handedness:
                hand_label = results.multi_handedness[0].classification[0].label
                is_right_hand = hand_label == "Right"
            else:
                # Por defecto asumimos mano derecha si no se detecta
                is_right_hand = True

            # CORRECCIÓN: Detección mejorada del pulgar
            # Usamos el punto de referencia 2 (muñeca) para determinar la posición del pulgar
            if is_right_hand:
                # Para mano derecha: pulgar abierto si está a la izquierda de la base
                fingers[0] = 1 if lm_list[4][1] < lm_list[3][1] else 0
            else:
                # Para mano izquierda: pulgar abierto si está a la derecha de la base
                fingers[0] = 1 if lm_list[4][1] > lm_list[3][1] else 0

            # Índice, medio, anular, meñique
            tip_ids = [8, 12, 16, 20]
            for i, id in enumerate(tip_ids):
                fingers[i + 1] = 1 if lm_list[id][2] < lm_list[id - 2][2] else 0

            # Convertir a string binario
            fingers_str = "".join(map(str, fingers))

            # Si es mano izquierda, invertir el orden para los LEDs
            # Orden original: pulgar, índice, medio, anular, meñique
            # Para mano izquierda: menique, anular, medio, índice, pulgar
            if not is_right_hand:
                fingers_str = fingers_str[::-1]

            # Enviar datos al Arduino con una frecuencia controlada
            current_time = time.time()
            if arduino and (current_time - last_send_time) > send_interval:
                try:
                    mano_tipo = "MAIN_DERECHA" if is_right_hand else "MAIN_IZQUIERDA"
                    mensaje = f"{fingers_str},{mano_tipo}\n"
                    arduino.write(mensaje.encode())
                    last_send_time = current_time
                    print(f"Enviado: {mensaje.strip()}")
                except serial.SerialException:
                    print("Error al enviar datos al Arduino")
                    arduino = None


            # Mostrar si es mano izquierda o derecha
            hand_type = "Derecha" if is_right_hand else "Izquierda"


            # Dibujar bounding box
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (128, 0, 128), 2)
            label = f"Mano: {hand_type}"
            cv2.putText(frame, label, (x_min - 20, y_min - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

    # Instrucciones en pantalla
    cv2.putText(frame, "Presiona 'q' para salir", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Reducir el tamaño de la ventana manteniendo la relación de aspecto
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    if logo is not None and logo.shape[2] == 4:  # Asegurar que tiene canal alfa
        overlay = logo[..., :3]
        mask = logo[..., 3:]  # Canal alfa
        h, w = overlay.shape[:2]
        # esquina inferior derecha (usa resized_frame en vez de frame)
        offset_x = 20
        y1, y2 = resized_frame.shape[0] - h, resized_frame.shape[0]
        x1, x2 = resized_frame.shape[1] - w - offset_x, resized_frame.shape[1] - offset_x
        roi = resized_frame[y1:y2, x1:x2]
        resized_frame[y1:y2, x1:x2] = (roi * (1 - mask / 255) + overlay * (mask / 255)).astype("uint8")

    max_width = 700

    if foto.shape[1] > max_width:
        scale = max_width / foto.shape[1]
        new_w = int(foto.shape[1] * scale)
        new_h = int(foto.shape[0] * scale)
        foto = cv2.resize(foto, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if foto is not None and foto.shape[2] == 4:  # asegura canal alfa
        overlay2 = foto[..., :3]
        mask2 = foto[..., 3:]
        h2, w2 = overlay2.shape[:2]

        offset_x = 30  # mover a la derecha
        offset_y = 50  # subir un poco desde abajo

        # esquina inferior izquierda con desplazamiento
        y1, y2 = resized_frame.shape[0] - h2 - offset_y, resized_frame.shape[0] - offset_y
        x1, x2 = offset_x, offset_x + w2

        roi = resized_frame[y1:y2, x1:x2]
        resized_frame[y1:y2, x1:x2] = (
                roi * (1 - mask2 / 255) + overlay2 * (mask2 / 255)
        ).astype("uint8")

    cv2.imshow("Control de LEDs con Manos", resized_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
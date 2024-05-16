import os
import cv2
import numpy as np
import mediapipe as mp

# Función para detectar puntos clave utilizando MediaPipe
def mediapipe_detection(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # CONVERSIÓN DE COLOR BGR A RGB
    imagen.flags.writeable = False  # La imagen ya no es modificable
    resultados = modelo.process(imagen)  # Realizar predicción
    imagen.flags.writeable = True  # La imagen vuelve a ser modificable
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)  # CONVERSIÓN DE COLOR RGB A BGR
    return imagen, resultados

# Función para dibujar landmarks de keypoints detectados para visualización
def dibujar_landmarks_estilizados(imagen, resultados):
    # Dibujar conexiones faciales
    mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                               mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))

# Función para extraer keypoints corporales detectados
def extraer_keypoints(resultados):
    keypoints_pose = np.array([[res.x, res.y, res.z, res.visibility] for res in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33*4)
    return keypoints_pose

def procesar_video(ruta_video, categoria,ruta_dataset):
    window = []
    cap = cv2.VideoCapture(ruta_video)

    # Especificar el codec de video para MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    nombre_video_sin_procesar = os.path.basename(ruta_video)

    ruta_video_procesado = os.path.join(ruta_dataset, 'video_procesado', categoria)
    ruta_numpy_procesado = os.path.join(ruta_dataset, 'numpy_procesado', categoria)

    if not os.path.exists(ruta_video_procesado):
        os.makedirs(ruta_video_procesado)
    

    if not os.path.exists(ruta_numpy_procesado):
        os.makedirs(ruta_numpy_procesado)
    

    nombre_archivo_video_procesado = os.path.splitext(nombre_video_sin_procesar)[0] + '.mp4'
    ruta_archivo_video_procesado = os.path.join(ruta_video_procesado, nombre_archivo_video_procesado)


    nombre_archivo_numpy_procesado = os.path.splitext(nombre_video_sin_procesar)[0] + '.npy'
    ruta_archivo_numpy_procesado = os.path.join(ruta_numpy_procesado, nombre_archivo_numpy_procesado)

    # Crear el objeto VideoWriter
    video_procesado = cv2.VideoWriter(ruta_archivo_video_procesado, fourcc, 10, (600, 400))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cv2.destroyAllWindows()
                break

            imagen, resultados = mediapipe_detection(frame, holistic)
            imagen = cv2.resize(imagen, (600, 400), interpolation=cv2.INTER_AREA)

            dibujar_landmarks_estilizados(imagen, resultados)

            keypoints = extraer_keypoints(resultados)
            window.append(keypoints)

            h, w, c = imagen.shape
            opImg = np.zeros([h, w, c], dtype=np.uint8)
            opImg.fill(255)
            dibujar_landmarks_estilizados(opImg, resultados)

            # cv2.imshow("Captura Video", imagen)
            # cv2.waitKey(1)

            # cv2.imshow("Pose Extraída", opImg)
            # cv2.waitKey(1)

            video_procesado.write(opImg)

        cap.release()
        video_procesado.release()

    np.save(ruta_archivo_numpy_procesado, np.array(window))


# Inicializando el modelo y herramientas de dibujo de MediaPipe
mp_holistic = mp.solutions.holistic  # Modelo Holístico
mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo

# Obtener la ruta actual
ruta_actual = os.getcwd()

ruta_raiz= os.path.dirname(ruta_actual)


# Crear las carpetas 'dataset' y 'video' dentro de la ruta actual
ruta_dataset = os.path.join(ruta_raiz, 'Data_Set')
ruta_videos = os.path.join(ruta_dataset, 'videos')

# Comprobar si las carpetas ya existen, si no, crearlas
if not os.path.exists(ruta_dataset):
    os.makedirs(ruta_dataset)

if not os.path.exists(ruta_videos):
    os.makedirs(ruta_videos)

# Obtener las categorias 
categoria_videos=os.listdir(ruta_videos)

# Recorre las categorias 
for categoria in categoria_videos:
    ruta_categoria = os.path.join(ruta_videos, categoria)
    nombre_videos = os.listdir(ruta_categoria) 

    for nombre_video in nombre_videos:
        ruta_video = os.path.join(ruta_categoria, nombre_video)
        procesar_video(ruta_video,categoria,ruta_dataset)
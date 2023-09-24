import cv2
import numpy as np
import imutils

coordenadas = [
    (581, 214, 53, 109),
    (532, 221, 46, 102),
    (453, 212, 39, 107),
    (409, 215, 40, 103),
    (344, 209, 41, 112),
    (304, 211, 36, 108),
    (200, 213, 41, 116),
    (156, 213, 40, 115),
    (80, 231, 50, 122),
    (34, 242, 43, 110),
    (3, 246, 29, 105)
]

video_cap = cv2.VideoCapture('3. Feria\Feria5.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Inicializar el estado y color de cada rectángulo
rectangles_state = {}
for i, rect in enumerate(coordenadas):
    rectangles_state[i] = {'color': (0, 255, 0), 'detected': False}

while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dibujar cada rectángulo en el frame
    # Crear una lista de puntos extremos para los rectángulos
    area_pts = []
    for (x, y, width, height) in coordenadas:
        pt1 = (x, y)
        pt2 = (x + width, y)
        pt3 = (x + width, y + height)
        pt4 = (x, y + height)
        area_pts.append(np.array([pt1, pt2, pt3, pt4]))

    # Dibujar los rectángulos principales utilizando drawContours y respetar el estado y color de cada uno
    for i, rect_pts in enumerate(area_pts):
        cv2.drawContours(frame, [np.array(rect_pts)], -1, rectangles_state[i]['color'], 2)

    # Crear una máscara de las áreas de los rectángulos
    imAux = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    for pts in area_pts:
        cv2.drawContours(imAux, [np.array(pts)], -1, (255), -1)
    imAux = cv2.bitwise_and(gray, gray, mask=imAux)

    fgmask = fgbg.apply(imAux)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv2.contourArea(cnt) > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 25, 0), 2)
            # Verificar si el contorno se encuentra dentro de algún rectángulo principal
            for i, rect_pts in enumerate(area_pts):
                if cv2.pointPolygonTest(rect_pts, (x + w // 2, y + h // 2), False) >= 0:
                    rectangles_state[i]['color'] = (0, 0, 255)  # Cambiar el color a rojo
                    rectangles_state[i]['detected'] = True

    # Restaurar el color a verde si no se detecta una persona en ese rectángulo
    cont_desocupados = 0
    for i, rect_state in rectangles_state.items():
        if not rect_state['detected']:
            rect_state['color'] = (0, 255, 0)
            cont_desocupados += 1
        # Reiniciar el estado para el siguiente frame
        rect_state['detected'] = False

    texto = f'Espacios desocupados: {cont_desocupados}'
    (tw, th) = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (10, 30 - th), (10 + tw, 30), (0, 0, 0), -1)
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Aplicar la máscara de las áreas de los rectángulos en el frame original
    cv2.imshow('frame', frame)
    cv2.imshow('imAux', fgmask)
    k = cv2.waitKey(40) & 0xFF
    if k == 27:
        break

video_cap.release()
cv2.destroyAllWindows()

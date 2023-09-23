import cv2
import pickle
import numpy as np

estacionamientos = []
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

video = cv2.VideoCapture('video.mp4')

estado_espacios = [False] * len(estacionamientos)
cont_desocupados = 0

while True:
    check, img = video.read()
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5, 5), np.int8)
    imgDil = cv2.dilate(imgMedian, kernel)

    for i, (x, y, w, h) in enumerate(estacionamientos):
        espacio = imgDil[y:y + h, x:x + w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if not estado_espacios[i] and count < 900:
            estado_espacios[i] = True
            cont_desocupados += 1
            
        elif estado_espacios[i] and count >= 900:
            estado_espacios[i] = False
            cont_desocupados -= 1
        
        if estado_espacios[i]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Muestra el contador en la ventana de video actualizada en tiempo real
    cv2.putText(img, f'Espacios desocupados: {cont_desocupados}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('video', img)
    cv2.waitKey(10)

import cv2
import cv2.aruco as aruco

capture = cv2.VideoCapture(0)

# Crie um dicionário para rastrear os círculos em cada marcador
circle_tracker = []

def findAruco(img, marker_size=6, total_markers=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    parameters = aruco.DetectorParameters()
    bbox, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters=parameters)
    
    if ids is not None:
        for i in range(len(ids)):
            id = ids[i][0]
            bbox_points = bbox[i][0]

            # Calcular o centro do marcador
            cX = int((bbox_points[0][0] + bbox_points[1][0] + bbox_points[2][0] + bbox_points[3][0]) / 4)
            cY = int((bbox_points[0][1] + bbox_points[1][1] + bbox_points[2][1] + bbox_points[3][1]) / 4)

            # Atualizar a posição do círculo para o centro do marcador
            circle_tracker.append((cX,cY))

    return bbox, ids

while True:
    ret, img = capture.read()

    bbox, ids = findAruco(img)
    
    # Desenhar os círculos com base nas posições dos marcadores
    for coord in circle_tracker:
        cv2.circle(img, coord, 5, (0, 0, 255), -1)  # Desenha um círculo vermelho

    if cv2.waitKey(1) == 27:
        break

    cv2.imshow("img", img)

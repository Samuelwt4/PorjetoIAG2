from ultralytics import YOLO
import cv2
from time import sleep
import math

delay = 30
pos_linha = 550
distancia_max = 50

carros = 0
motos = 0
caminhoes = 0

veiculos_rastreado = []

cores = {
    'car': (255, 0, 0),
    'motorcycle': (0, 255, 0),
    'truck': (0, 0, 255)
}

nomes_pt = {
    'car': 'Carro',
    'motorcycle': 'Moto',
    'truck': 'Caminhao'
}

model = YOLO('yolov5s.pt')
cap = cv2.VideoCapture('C:/Users/Samuel/Downloads/Nova pasta/Vehicle-Counter/video.mp4')

def distancia(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sleep(1 / delay)
    results = model.predict(source=frame, conf=0.3, verbose=False)[0]
    cv2.line(frame, (25, pos_linha), (1200, pos_linha), (0, 255, 0), 3)

    novos_centros = []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls = int(r.cls[0])
        label = model.names[cls]

        if label not in ['car', 'motorcycle', 'truck']:
            continue

        largura = x2 - x1
        altura = y2 - y1
        aspecto = largura / altura if altura != 0 else 0

        if largura < 50 or altura < 50:
            continue

        if label == 'car' and (largura > 260 or altura > 260) and aspecto < 2.0:
            label = 'truck'

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cor = cores[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(frame, nomes_pt[label], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

        novos_centros.append({'centro': (cx, cy), 'label': label, 'contado': False})

    veiculos_atualizados = []

    for novo in novos_centros:
        centro_novo = novo['centro']
        label_novo = novo['label']
        contado_novo = False
        achou = False

        for v in veiculos_rastreado:
            dist = distancia(centro_novo, v['centro'])

            if dist < distancia_max and label_novo == v['label']:
                if not v['contado']:
                    # De cima para baixo → soma
                    if v['centro'][1] < pos_linha <= centro_novo[1]:
                        if label_novo == 'car':
                            carros += 1
                        elif label_novo == 'motorcycle':
                            motos += 1
                        elif label_novo == 'truck':
                            caminhoes += 1
                        contado_novo = True
                        print(f"{nomes_pt[label_novo]} passou ↓. Total: Carros {carros}, Motos {motos}, Caminhões {caminhoes}")

                    # De baixo para cima → subtrai
                    elif v['centro'][1] > pos_linha >= centro_novo[1]:
                        if label_novo == 'car':
                            carros = max(0, carros - 1)
                        elif label_novo == 'motorcycle':
                            motos = max(0, motos - 1)
                        elif label_novo == 'truck':
                            caminhoes = max(0, caminhoes - 1)
                        contado_novo = True
                        print(f"{nomes_pt[label_novo]} voltou ↑. Total: Carros {carros}, Motos {motos}, Caminhões {caminhoes}")

                veiculos_atualizados.append({
                    'centro': centro_novo,
                    'label': label_novo,
                    'contado': v['contado'] or contado_novo
                })
                achou = True
                break

        if not achou:
            veiculos_atualizados.append(novo)

    veiculos_rastreado = veiculos_atualizados

    # Exibe contadores na tela
    cv2.rectangle(frame, (10, 10), (420, 120), (0, 0, 0), -1)
    cv2.putText(frame, f'Carros: {carros}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cores['car'], 2)
    cv2.putText(frame, f'Motos: {motos}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, cores['motorcycle'], 2)
    cv2.putText(frame, f'Caminhoes: {caminhoes}', (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, cores['truck'], 2)

    cv2.imshow("Contador de Veículos - IA", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

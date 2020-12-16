# -*- coding: utf-8 -*-

# Face detection - https://github.com/alexcamargoweb/face-detection
# Detecção de faces com OpenCV e Deep Learning.
# Adrian Rosebrock, Face detection with OpenCV and deep learning. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/.
# Acessado em: 10/12/2020. 
# Arquivo: train_face-detector.py
# Execução via Spyder/Linux

# importa os pacotes necessários
import numpy as np
import argparse
import cv2, os, time
import imutils
from imutils.video import VideoStream

# caminho das imagens
DATASET = './datasets'
# arquivo de deploy do Caffee (estrutura da rede)
PROTOTXT = './detector/deploy.prototxt'
# modelo pré-treinado do Caffee (pesos da rede)
DETECTOR =  './detector/res10_300x300_ssd_iter_140000.caffemodel'
# fator de confiança
CONFIDENCE = 0.5

# carrega o modelo do disco
print("[INFO] carregando modelo pré-treinado...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, DETECTOR)
# inicializa o stream de vídeo pela câmera
print("[INFO] iniciando stream de vídeo...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

# faz um loop sobre os frames do vídeo
while True:
    # pega o frame do vídeo e redimensiona para 800px
	frame = vs.read()
	frame = imutils.resize(frame, width = 800)
 
    # pega as dimensões do frame e converte em um blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
    # passa o blob para a rede e obtém as detecções
	net.setInput(blob)
	detections = net.forward()
    
    # faz um loop sobre as detecções
	for i in range(0, detections.shape[2]):
        # extrai a confiança associada à detecção
		confidence = detections[0, 0, i, 2]
    	# filtra as detecções fracas, garantindo que a confiança é
    	# maior do que a confiança mínima
		if confidence < CONFIDENCE:
			continue
		# processa as coordenadas x e y da bouding box do objeto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
        # desenha a boudinhd box no rosto com a probabilidade associada
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
        # exibe o ratângulo e o texto
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
	# exibe o frame de saída
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# encerra o loop caso o usuário pressione o "q"
	if key == ord("q"):
		break
    
# limpa a execução
cv2.destroyAllWindows()
vs.stop()
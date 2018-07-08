import cv2
#import open cv
import sys
import numpy as np
#import numpy for scientific calculations
from matplotlib import pyplot as plt
import time

#
#Guilherme Bacca & Peterson Boni
#

green=(0,255,0)
red=(255,0,0)
blue=(0,0,255)


def maior_contorno(image):
	image=image.copy()

	_ , contours , hierarchy=cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contour_sizes=[(cv2.contourArea(contour),contour) for contour in contours]
	if(len(contour_sizes) <= 0):
		return [], []
	biggest_contour=max(contour_sizes,key=lambda x:x[0])[1]
	mask=np.zeros(image.shape,np.uint8)
	cv2.drawContours(mask,[biggest_contour],-1,255,-1)

	return biggest_contour,mask

   

def overlay_mask(mask,image):
	rgb_mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
	img=cv2.addWeighted(rgb_mask,0.5,image,0.5,0)
	return img


def circle_contour(image,contour):
	imagem_com_elipse=image.copy()
	ellipse=cv2.fitEllipse(contour)
	cv2.ellipse(imagem_com_elipse,ellipse,green,2,1)
	return imagem_com_elipse


def show(image):
	plt.figure(figsize=(10,10))
	plt.imshow(image,interpolation='nearest')


#ver se é maçã
def maca(image):

	#PRE PROCESSAMENTO

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	#min e max de cores
	min_color=np.array([0,100,80])
	max_color=np.array([10,256,256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2=np.array([170,100,80])
	max_color2=np.array([180,256,256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	#somar mascaras com as cores
	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	#fechamento e abertura das mascaras
	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	#cv2.imshow('cont', mask_cleaned)
	#encontrar o maior contorno
	big_contour,mask_fruit=maior_contorno(mask_cleaned)

	#se não acha contorno, retorna pq não é essa fruta
	if (len(big_contour) <= 0):
		return 0, []

	#desenha uma borra na regiao da fruta
	overlay=overlay_mask(mask_cleaned,image)

	#circula a fruta
	#circled=circle_contour(image,big_contour)
	#show(circled)

	bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

	print('tam maça ',len(big_contour))
	#if (len(big_contour) < 500):
	#	return []
	#else:
	return len(big_contour), bgr


#ver se é banana
def banana(image):

	#pre processamento

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	#min e max cores para encontrar
	min_color = np.array([20, 50, 50])
	max_color = np.array([30, 256, 256])


	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2 = np.array([60, 50, 50])
	max_color2 = np.array([70, 256, 256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	#somar as mascaras com as cores
	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	#fechamento e abertura das mascaras
	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	#encontrar o maior contorno
	big_contour,mask_fruit=maior_contorno(mask_cleaned)

	#se não há contorno, retorna, pq não é essa fruta
	if(len(big_contour) <= 0):
		return 0, []

	#borrar area da fruta
	#overlay=overlay_mask(mask_cleaned,image)

	#circular fruta
	#circled=circle_contour(image,big_contour)
	#show(circled)

	bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
	print('tam banana ',len(big_contour))
	#if(len(big_contour) < 600):
	#	return []
	#else:
	return len(big_contour), bgr


#ver se é laranja
def laranja(image):

	#pre processamento

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	#minimo de cores
	min_color=np.array([20,100,150])
	max_color=np.array([80,160,256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	#maximo
	min_color2=np.array([80,180,100])
	max_color2=np.array([256,256,256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	#somar as mascaras de max e min
	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	#fechamento e abertura
	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	#maior contorno da mascara
	big_contour,mask_fruit=maior_contorno(mask_cleaned)

	#se nao achou contorno, volta, pq não é essa
	if (len(big_contour) <= 0):
		return 0, []

	#borrar a area da fruta
	#overlay=overlay_mask(mask_cleaned,image)

	#fazer um circulo na imagem
	#circled=circle_contour(image,big_contour)
	#show(circled)

	bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

	print('tam laranja ',len(big_contour))
	#if (len(big_contour) < 200):
	#	return []
	#else:
	return len(big_contour), bgr





#input image

#imagem = cv2.imread('frutas/banana/banana1.jpg')
#imagem = cv2.imread('frutas/apple/maca6.jpg')
#imagem = cv2.imread('frutas/laranja/laranja10.jpg')




'''if(len(sys.argv) < 2):
	imagem = cv2.imread('frutas/frutas/fruta1.jpg')
else:
	print('entrei')
	imagem = cv2.imread(sys.argv[1])'''


imagem = []
for contador in range(1, 23):
	imagem = cv2.imread('frutas/frutas/fruta'+str(contador)+'.jpg')
	tam1, result1 = banana(imagem)
	tam2, result2 = maca(imagem)
	tam3, result3 = laranja(imagem)

	print('result1: ',tam1)
	print('result2: ',tam2)
	print('result3: ',tam3)

	if(tam1 > tam2 and tam1 > tam3):
		result_banana = result1
		largura = imagem.shape[1]
		altura = imagem.shape[0]
		texto = '{}'.format('Banana')
		fonte = cv2.FONT_HERSHEY_COMPLEX
		escala = 2
		grossura = 3
		tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
		cv2.putText(result_banana, texto, (30, 50), fonte, escala,
					(0, 0, 0), grossura)
		cv2.imshow('Fruta', result_banana)
		#cv2.imwrite('banana_new.jpg',result_banana)
	elif(tam2 > tam1 and tam2 > tam3):
		result_apple = result2
		largura = imagem.shape[1]
		altura = imagem.shape[0]
		texto = '{}'.format('Apple')
		fonte = cv2.FONT_HERSHEY_DUPLEX
		escala = 2
		grossura = 3
		tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
		cv2.putText(result_apple, texto, (30, 50), fonte, escala,
					(0, 0, 0), grossura)
		cv2.imshow('Fruta', result_apple)
	elif(tam3 > tam1 and tam3 > tam2):
		resulta_laranja = result3#draw_strawberry(imagem)
		largura = imagem.shape[1]
		altura = imagem.shape[0]
		texto = '{}'.format('Laranja')
		fonte = cv2.FONT_HERSHEY_DUPLEX
		escala = 2
		grossura = 3
		tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
		cv2.putText(resulta_laranja, texto, (30, 50), fonte, escala, (0, 0, 0), grossura)
		cv2.imshow('Fruta', resulta_laranja)
	else:
		img = cv2.imread('yo2.jpg')
		cv2.imshow('sem fruta', img)

	cv2.waitKey(0)

cv2.waitKey(0)
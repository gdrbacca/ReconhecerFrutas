import cv2
#import open cv
import numpy as np
#import numpy for scientific calculations
from matplotlib import pyplot as plt
#display the image


green=(0,255,0)
red=(255,0,0)
blue=(0,0,255)


def find_biggest_contour(image):
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



def draw_apple(image):

	#PRE PROCESSAMENTO

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	min_color=np.array([0,100,80])
	max_color=np.array([10,256,256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2=np.array([170,100,80])
	max_color2=np.array([180,256,256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

	if (len(big_contour) <= 0):
		return 0, []

	#desenha uma borra na regiao da fruta
	overlay=overlay_mask(mask_cleaned,image)

	#circula a fruta
	circled=circle_contour(image,big_contour)

	#show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

	print('tam maça ',len(big_contour))
	#if (len(big_contour) < 500):
	#	return []
	#else:
	return len(big_contour), bgr

def draw_banana(image):

	#PRE PROCESSING OF IMAGE

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	min_color = np.array([20, 50, 50])
	max_color = np.array([30, 256, 256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2 = np.array([60, 50, 50])
	max_color2 = np.array([70, 256, 256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)
	big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

	if(len(big_contour) <= 0):
		return 0, []

	overlay=overlay_mask(mask_cleaned,image)

	circled=circle_contour(image,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)
	print('tam banana ',len(big_contour))
	#if(len(big_contour) < 600):
	#	return []
	#else:
	return len(big_contour), bgr


def draw_strawberry(image):

	#PRE PROCESSING OF IMAGE

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	min_color=np.array([20,100,150])
	max_color=np.array([80,160,256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2=np.array([80,180,100])
	max_color2=np.array([256,256,256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)

	big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

	if (len(big_contour) <= 0):
		return 0, []
	overlay=overlay_mask(mask_cleaned,image)

	circled=circle_contour(image,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

	print('tam laranja ',len(big_contour))
	#if (len(big_contour) < 200):
	#	return []
	#else:
	return len(big_contour), bgr





#input image

#imagem = cv2.imread('frutas/banana/banana1.jpg')
#imagem = cv2.imread('frutas/apple/maca3.jpg')
imagem = cv2.imread('frutas/laranja/laranja11.jpg')

#apple=cv2.imread('apple.jpg')
#banana=cv2.imread('frutas/banana/bananas.jpg')
#strawberry=cv2.imread('strawberry.jpg')
#fruit=cv2.imread('fruit.jpg')

#process image
#result_apple=draw_apple(apple)
#result_banana=draw_banana(apple)
#result_strawberry=draw_strawberry(strawberry)
#result_fruit=draw_banana(fruit)

tam1, result1 = draw_banana(imagem)
tam2, result2 = draw_apple(imagem)
tam3, result3 = draw_strawberry(imagem)

print('result1: ',tam1)
print('result2: ',tam2)
print('result3: ',tam3)

if(tam1 > tam2 and tam1 > tam3):
	_, result_banana = draw_banana(imagem)
	largura = imagem.shape[1]
	altura = imagem.shape[0]
	texto = '{}'.format('Banana')
	fonte = cv2.FONT_HERSHEY_COMPLEX
	escala = 2
	grossura = 3
	tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
	cv2.putText(result_banana, texto, (30, 50), fonte, escala,
				(0, 0, 0), grossura)
	cv2.imshow('Banana', result_banana)
	#cv2.imwrite('banana_new.jpg',result_banana)
elif(tam2 > tam1 and tam2 > tam3):
	result_apple = draw_apple(imagem)
	largura = imagem.shape[1]
	altura = imagem.shape[0]
	texto = '{}'.format('Apple')
	fonte = cv2.FONT_HERSHEY_DUPLEX
	escala = 2
	grossura = 3
	tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
	cv2.putText(result_apple, texto, (30, 50), fonte, escala,
				(0, 0, 0), grossura)
	cv2.imshow('Maca', result_apple)
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
	cv2.imshow('Laranja', resulta_laranja)
else:
	img = cv2.imread('yo2.jpg')
	cv2.imshow('sem fruta', img)
#output image

#cv2.imwrite('apple_new.jpg',result_apple)

#cv2.imwrite('strawberry_new.jpg',result_strawberry)
#cv2.imwrite('fruit_new.jpg',result_fruit)

cv2.waitKey(0)
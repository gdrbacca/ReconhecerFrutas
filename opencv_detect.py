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

	overlay=overlay_mask(mask_cleaned,image)

	circled=circle_contour(overlay,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

	return bgr

def draw_banana(image):

	#PRE PROCESSING OF IMAGE

	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	maxsize=max(image.shape)

	scale=700/maxsize

	image=cv2.resize(image,None,fx=scale,fy=scale)

	image_blur=cv2.GaussianBlur(image,(7,7),0)

	image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

	min_color=np.array([20,50,50])
	max_color=np.array([30,256,256])

	mask1=cv2.inRange(image_blur_hsv,min_color,max_color)

	min_color2=np.array([60,50,50])
	max_color2=np.array([70,256,256])

	mask2=cv2.inRange(image_blur_hsv,min_color2,max_color2)

	mask=mask1+mask2

	kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

	mask_closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask_cleaned=cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)
	big_contour,mask_fruit=find_biggest_contour(mask_cleaned)

	if(len(big_contour) <= 0):
		return []

	cv2.imshow("das", mask_fruit)
	overlay=overlay_mask(mask_cleaned,image)

	circled=circle_contour(overlay,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)
	print(len(big_contour))
	if(len(big_contour) < 200):
		return []
	else:
		return bgr


def draw_strawberry(image):

	#PRE PROCESSING OF IMAGE

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

	overlay=overlay_mask(mask_cleaned,image)

	circled=circle_contour(overlay,big_contour)

	show(circled)

	bgr=cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)

	return bgr





#input image


apple=cv2.imread('apple.jpg')
banana=cv2.imread('frutas/banana/bananas.jpg')
strawberry=cv2.imread('strawberry.jpg')
fruit=cv2.imread('fruit.jpg')
#process image
result_apple=draw_apple(apple)
#result_banana=draw_banana(apple)
result_strawberry=draw_strawberry(strawberry)
#result_fruit=draw_banana(fruit)

if(len(draw_banana(banana)) > 0):
	result_banana = draw_banana(banana)
	largura = banana.shape[1]
	altura = banana.shape[0]
	texto = '{}'.format('Banana')
	fonte = cv2.FONT_HERSHEY_DUPLEX
	escala = 2
	grossura = 3
	tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)
	cv2.putText(result_banana, texto, (30, 50), fonte, escala,
				(0, 0, 0), grossura)
	cv2.imshow('reconhecida', result_banana)
	#cv2.imwrite('banana_new.jpg',result_banana)
#output image

cv2.imwrite('apple_new.jpg',result_apple)

cv2.imwrite('strawberry_new.jpg',result_strawberry)
#cv2.imwrite('fruit_new.jpg',result_fruit)

cv2.waitKey(0)
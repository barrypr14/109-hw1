import tkinter as tk
import numpy as np
import cv2

root = tk.Tk()
width = 760
height = 320
screenwidth = root.winfo_screenwidth()  
screenheight = root.winfo_screenheight()  
size = '%dx%d+%d+%d' % (width, height, (screenwidth - width)/2, (screenheight - height)/2)
root.geometry(size) # width*height + pos_x + pos_y
root.title('OpenCv-Hw1-2020')

frame1 = tk.LabelFrame(root , width = 160 , height = 300 , text = 'Image Processing').grid(row = 0 , column = 0 , padx = 10 )
frame2 = tk.LabelFrame(root , width = 160 , height = 300 , text = 'Image Smoothing').grid(row = 0 , column = 2 , padx = 10 )
frame3 = tk.LabelFrame(root , width = 160 , height = 300 , text = 'Edge Detection').grid(row = 0 , column = 4 , padx = 10 )
frame4 = tk.LabelFrame(root , width = 200 , height = 300 , text = 'Transformation').grid(row = 0 , column = 6 , padx = 10 )

def wait():
    while True :
        if cv2.waitKey(1) == 27 :
            break
######################################## image processing 
uncle_roger_image = cv2.imread('uncle_roger.jpg')
flower_image = cv2.imread('flower.jpg')

def load_image():
    cv2.imshow('1.1 load image' , uncle_roger_image)

    print('1.1 uncle_roger.jpg shape : ')
    print('width : ' , uncle_roger_image.shape[1])
    print('height : ' , uncle_roger_image.shape[0])

    wait()
    cv2.destroyWindow('1.1 load image')

def color_seperation () :
    cv2.imshow('1.2 color seperation origin' , flower_image)

    b = flower_image.copy()
    b[:,:,1] = 0
    b[:,:,2] = 0
    
    g = flower_image.copy()
    g[:,:,0] = 0
    g[:,:,2] = 0

    r = flower_image.copy()
    r[:,:,0] = 0
    r[:,:,1] = 0

    cv2.imshow('1.2 color seperation B' , b)
    cv2.imshow('1.2 color seperation G' , g)
    cv2.imshow('1.2 color seperation R' , r)

    wait()

    cv2.destroyWindow('1.2 color seperation origin')
    cv2.destroyWindow('1.2 color seperation B')
    cv2.destroyWindow('1.2 color seperation G')
    cv2.destroyWindow('1.2 color seperation R')

flipping = cv2.flip(uncle_roger_image,1)

def image_flipping():
    cv2.imshow('1.3 image origin' , uncle_roger_image)
    cv2.imshow('1.3 image after flipping' , flipping)

    wait()
    cv2.destroyWindow('1.3 image origin')
    cv2.destroyWindow('1.3 image after flipping')

def blending():
    def update(x) :
        pass

    cv2.namedWindow('1.4 blending image')
    cv2.createTrackbar('blend' , '1.4 blending image' , 0 , 255 , update)
    cv2.setTrackbarPos('blend' , '1.4 blending image' , 0)

    while True :
        alpha = cv2.getTrackbarPos('blend' , '1.4 blending image')
        alpha = alpha/255
        blend = np.uint8(alpha * uncle_roger_image + (1 - alpha) * flipping)
        cv2.imshow('1.4 blending image' , blend)
        if cv2.waitKey(1) == 27 :
            break
    cv2.destroyWindow('1.4 blending image')

btn1_1 = tk.Button(frame1 , text = '1.1 Load image' , width = 17 , height = 2 , command = load_image).place(x = 25 , y = 30)
btn1_2 = tk.Button(frame1 , text = '1.2 Color Seperation' , width = 17 , height = 2 , command = color_seperation).place(x = 25 , y =97)
btn1_3 = tk.Button(frame1 , text = '1.3 Image Flipping' , width = 17 , height = 2 , command = image_flipping).place(x = 25 , y =164)
btn1_4 = tk.Button(frame1 , text = '1.4 Blending' , width = 17 , height = 2 , command = blending).place(x = 25 , y = 230 )

#################################################### Image Smoothing

cat_image = cv2.imread('cat.jpg')

def median_filter() :
    cv2.imshow('2.1 the origin image' , cat_image)
    cat_median = cv2.medianBlur(cat_image , 7)
    cv2.imshow('2.1 the image after median filter' , cat_median)

    wait()
    cv2.destroyWindow('2.1 the origin image')
    cv2.destroyWindow('2.1 the image after median filter')

def guassin_filter() :
    cv2.imshow('2.2 the origin image' , cat_image)
    cat_guassin = cv2.GaussianBlur(cat_image , (3,3) , 0)
    cv2.imshow('2.2 the image after guassin filter' , cat_guassin)

    wait()
    cv2.destroyWindow('2.2 the origin image')
    cv2.destroyWindow('2.2 the image after guassin filter')

def bilateral() :
    cv2.imshow('2.3 the origin image' , cat_image)
    cat_bilateral = cv2.bilateralFilter(cat_image , 9 , 90 , 90)

    cv2.imshow('2.3 the image after bilateral filter' , cat_bilateral)

    wait()
    cv2.destroyWindow('2.3 the origin image')
    cv2.destroyWindow('2.3 the image after bilateral filter')

btn2_1 = tk.Button(frame2 , text = '2.1 Median Filter' , width = 17 , height = 2 , command = median_filter).place(x = 205 , y = 60)
btn2_2 = tk.Button(frame2 , text = '2.2 Guassin Filter' , width = 17 , height = 2 , command = guassin_filter).place(x = 205 , y =127)
btn2_3 = tk.Button(frame2 , text = '2.3 Bilateral Filter' , width = 17 , height = 2 , command = bilateral).place(x = 205 , y =194)

################################################################# Guassin Blur
from PIL import Image

chihiro_image = Image.open('chihiro.jpg') 

chihiro_width = chihiro_image.size[0]
chihiro_height = chihiro_image.size[1]


for i in range (chihiro_width) :
    for j in range(chihiro_height) :
        #chihiro_grayscale[i,j]  = int(( chihiro_image[i,j,0] * 15 + chihiro_image[i,j,1] * 75 + chihiro_image[i,j,2] * 38 ) /128 )
        rgb = chihiro_image.getpixel((i,j))
        gray = int((rgb[0]+rgb[1]+rgb[2])/3)
        rgb = (gray,gray,gray)
        chihiro_image.putpixel((i,j),rgb)

chihiro_grayscale = np.uint8(chihiro_image)

#3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp((-(x**2+y**2)))

chihiro_guassinBlur = np.zeros((chihiro_height,chihiro_width))
#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

for i in range (1,chihiro_width-2) :
    for j in range(1,chihiro_height-2) :
        chihiro_guassinBlur[j,i] = np.sum(chihiro_grayscale[j-1:j+2,i-1:i+2,0] * gaussian_kernel)

chihiro_guassinBlur = np.uint8(chihiro_guassinBlur)

def guassin_blur() :
    #cv2.imshow('3.1 the origin image' , chihiro_image)
    cv2.imshow('3.1 the image after grayscale' , chihiro_grayscale)
    cv2.imshow('3.1 the image after grayscale and guassin blur' , chihiro_guassinBlur)
    wait()
    #cv2.destroyWindow('3.1 the origin image')
    cv2.destroyWindow('3.1 the image after grayscale')
    cv2.destroyWindow('3.1 the image after grayscale and guassin blur')


############### sobel x
sobelx = np.zeros(chihiro_guassinBlur.shape)

for i in range (1 , chihiro_width -2 ) :
    for j in range (1 , chihiro_height - 2) :
        sobelx[j,i] = np.abs(np.sum(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * chihiro_guassinBlur[j-1:j+2 , i-1:i+2 ]))

sobelx = sobelx*(255/np.max(chihiro_guassinBlur))
sobelx = np.uint8(sobelx)
################ sobel y
sobely = np.zeros(chihiro_guassinBlur.shape)
for i in range (1 , chihiro_width -2 ) :
    for j in range (1 , chihiro_height - 2) :
        sobely[j,i] = np.abs(np.sum(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) * chihiro_guassinBlur[j-1:j+2 , i-1:i+2 ]))

sobely = sobely*(255/np.max(chihiro_guassinBlur))
sobely = np.uint8(sobely)

def sobelX () :
    cv2.imshow('image after sobel x' , sobelx)

    wait()
    cv2.destroyWindow('image after sobel x')

def sobelY() :
    cv2.imshow('image after sobel y' , sobely)

    wait()
    cv2.destroyWindow('image after sobel y')

def magnitude():
    chihiro_sobelMagnitude = np.zeros(chihiro_guassinBlur.shape) 
    for i in range(chihiro_width):
        for j in range(chihiro_height) :
            chihiro_sobelMagnitude[j,i] = np.sqrt(sobelx[j,i]**2 + sobely[j,i]**2)

    chihiro_sobelMagnitude = np.uint8(chihiro_sobelMagnitude)

    cv2.imshow('magnitude',chihiro_sobelMagnitude)

    wait()
    cv2.destroyWindow('magnitude')

btn3_1 = tk.Button(frame3 , text = '3.1 Guassin Blur' , width = 17 , height = 2 , command = guassin_blur).place(x = 385 , y = 30)
btn3_2 = tk.Button(frame3 , text = '3.2 Sobel X' , width = 17 , height = 2 , command = sobelX).place(x = 385 , y =97)
btn3_3 = tk.Button(frame3 , text = '3.3 Sobel Y' , width = 17 , height = 2 , command = sobelY).place(x = 385 , y =164)
btn3_4 = tk.Button(frame3 , text = '3.4 Magnitude' , width = 17 , height = 2 , command = magnitude).place(x = 385 , y = 230 )

####################################### Transformation

label_1 = tk.Label(frame4 , text = 'Rotation').place(x = 555 , y = 30)
entry1 = tk.Entry(frame4 , width = 10 , bd = 5)
entry1.place(x = 620 , y = 30)
label_1_unit = tk.Label(frame4 , text = 'degree').place(x = 700 , y = 30)
label_2 = tk.Label(frame4 , text = 'Scale').place(x = 555 , y = 85)
entry2 = tk.Entry(frame4 , width = 10 , bd = 5)
entry2.place(x = 620 , y = 75)
label_3 = tk.Label(frame4 , text = 'Tx').place(x = 555 , y = 130)
entry3 = tk.Entry(frame4 , width = 10 , bd = 5)
entry3.place(x = 620 , y = 130)
label_3_unit = tk.Label(frame4 , text = 'pixel').place(x = 700 , y = 130)
label_4 = tk.Label(frame4 , text = 'Ty').place(x = 555 , y = 185)
entry4 = tk.Entry(frame4 , width = 10 , bd = 5)
entry4.place(x = 620 , y = 185)
label_4_unit = tk.Label(frame4 , text = 'pixel').place(x = 700 , y = 185)

def transformation():
    parrot_image = cv2.imread('parrot.jpg')
    HV = np.float32([[1,0,entry3.get()],[0,1,entry4.get()]])
    rows , cols = parrot_image.shape[:2]
    parrot_image_HV = cv2.warpAffine(parrot_image,HV,(rows,cols))

    parrot_image_scale = cv2.resize(parrot_image_HV,None, fx = float(entry2.get()) , fy= float(entry2.get()) , interpolation=cv2.INTER_AREA)

    rows_scale,cols_scale =parrot_image_scale.shape[:2]

    M = cv2.getRotationMatrix2D((rows_scale/2,cols_scale/2) , int(entry1.get()) , 1)

    parrot_image_result = cv2.warpAffine(parrot_image_scale,M,(rows_scale,cols_scale))

    cv2.imshow('transformation' , parrot_image_result)

    wait()
    cv2.destroyWindow('transformation')

btn4 = tk.Button(frame4 , text = 'Transformation' , width = 17 , height = 2 , command = transformation).place(x = 590 , y = 240 )
root.mainloop()
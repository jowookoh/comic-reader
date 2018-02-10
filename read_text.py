
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import math
from keras.models import load_model




redni=1
ukupnoZaPrevod=[]
ucitaneImg = []
for im in glob.glob("for_reading/*.png"):
    n= cv2.imread(im)
    ucitaneImg.append(n)

def load_image(loaded):
    return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def image_bin_txt(image_gs):
    height, width = image_gs.shape[0:2]
    ret,image_bin = cv2.threshold(image_gs, 40, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((2,4)) 
    return cv2.dilate(image, kernel, iterations=10)

def dilate_inv(image):
    kernel = np.ones((5,1)) 
    return cv2.dilate(image, kernel, iterations=1)

def scale_to_range(image): 
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))        
    return ready_for_ann

def erode(image):
    kernel = np.ones((4,2)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def winner(output): 
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    recenica=""
    for output in outputs:
        recenica+=str((alphabet[winner(output)]))
    
    return recenica

def dbsc(konture,eps,minKon):
    samo_slova=[]
    labels = [0]*len(konture)
    C=0
    for k in range(0, len(konture)):
        if not (labels[k] == 0):
            continue
        komsije = pretrazi_region(konture, k, eps)
        if len(komsije) < minKon:
            labels[k] = -1
        else: 
            C += 1
            growCluster(konture, labels, k, komsije, C, eps, minKon)       
    for ind in range(len(labels)):
        if labels[ind]!=-1:
            samo_slova.append(konture[ind])
    return samo_slova    

def growCluster(konture, labels, k, komsije, C, eps, minKon):
    labels[k] = C
    i = 0
    while i < len(komsije):         
        Pn = komsije[i]
        if labels[Pn]==-1:
            labels[Pn] = C
            PnKomsije = pretrazi_region(konture, Pn, eps-15)
            komsije = komsije + PnKomsije            
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnKomsije = pretrazi_region(konture, Pn, eps-15)
            komsije = komsije + PnKomsije         
        i += 1        
def pretrazi_region(konture, k, eps):
    komsije = []
    for Pn in range(0, len(konture)):
        konture=np.array(konture)
        if np.linalg.norm(konture[k] - konture[Pn]) < eps:
           komsije.append(Pn)            
    return komsije

def select_roi(image_orig, image_bin):

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if h < 50 and h > 30 and w >22 and w< 70  :
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,0,0),5)
    return image_orig

def find_obl(prvaSlova):
    hsv = cv2.cvtColor(prvaSlova, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])   
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  
    res = cv2.bitwise_and(prvaSlova,prvaSlova, mask= mask)
    return res


def find_cele_obl(original,original_b, oblacici_b):
    _, oblaci, _ = cv2.findContours(oblacici_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    koordinate_oblaci = []
    for oblak in oblaci: # za svaku konturu            
        x,y,w,h = cv2.boundingRect(oblak)
        area = cv2.contourArea(oblak) 
        crnobelo = original_b[y:y+h,x:x+w]
        procenat = cv2.countNonZero(crnobelo) / (w*h)        
        _, _, rotation = cv2.minAreaRect(oblak)  
        rotation = int(round(100*(abs(math.sin(math.radians(rotation))) + abs(math.cos(math.radians(rotation))))))
        rect = cv2.minAreaRect(oblak)   
        box = cv2.boxPoints(rect)
        box = np.int0(box) 
        cv2.drawContours(original,[box],0,(0,0,255),2)          
        if ((procenat>0.7 and rotation < 110 ) or (area>64000 and procenat>0.5)):      
            if(w>185 and h>30):   
                x=x-20
                y=y-20
                w=w+20
                h=h+20     
                cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),5)   
                koordinate_oblaci.append((x,y,w,h))
       
    koordinate_oblaci = sorted(koordinate_oblaci, key = lambda nzm: nzm[1])
    return koordinate_oblaci,original

def trazi_tekst(koordinate,original_b):
    izvuceni_tekstovi=[]
    izvuceni_koo=[]
    for oblak_koo in koordinate:        
        x,y,w,h=oblak_koo
        izvucen = original_b[y:y+h,x:x+w]
        izvuceni_tekstovi.append(izvucen)   
        izvuceni_koo.append((x,y,w,h))
    return ime,izvuceni_tekstovi,izvuceni_koo

def trazi_slova(image_orig, image_bin):
    _, contours,_ = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    slovo=[]
    samo_slova=[]
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)      
        if h < 50 and h > 30 and w< 80  :        
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),3)
            if w>45:
                w1=round(w/2)
                slovo.append([x,y,w1,h])
                slovo.append([x+w1,y,w1,h])
            else: 
                slovo.append([x,y,w,h])    
      
    samo_slova=dbsc(slovo,70,6)      
    if len(samo_slova)==0:
        return regions_array
    for samo in samo_slova:
        x,y,w,h=samo
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,0,0),2)
    samo_slova.sort(key=lambda b: b[1])    
    line_bottom = samo_slova[0][1]+samo_slova[0][3]-1
    line_begin_idx = 0    
    for i in range(1,len(samo_slova)):               
        if samo_slova[i][1] > line_bottom:           
            samo_slova[line_begin_idx:i] = sorted(samo_slova[line_begin_idx:i], key=lambda b: b[0])
            line_begin_idx = i      
        line_bottom = max(samo_slova[i][1]+samo_slova[i][3]-1, line_bottom)
    samo_slova[line_begin_idx:] = sorted(samo_slova[line_begin_idx:], key=lambda b: b[0])
    
    for i in range(0, len(samo_slova)):
        distance=0
        distanceY=0
        x=samo_slova[i][0]
        y=samo_slova[i][1]
        w=samo_slova[i][2]
        h=samo_slova[i][3]
        if(i<len(samo_slova)-1):
            nextX = samo_slova[i+1][0]
            nextY = samo_slova[i+1][1]
            distance= nextX-(x+w)
            distanceY= nextY-(y+h)
        region = image_bin[y:y+h+1,x:x+w+1]
        region=resize_region(region)     
        regions_array.append(invert(region)) 
        if(abs(distance)>10 or distanceY>5):   
            regions_array.append(invert(razmak))  
    return regions_array
 
def ispisi_tekst(izvuceni_tekst,original,izvuceni_koo,redni):
    model = load_model('my_model.h5') 
    for index in range(len(izvuceni_tekst)):
        izvuceni_tekst[index]=image_bin(izvuceni_tekst[index])
        sortirana_slova = trazi_slova(original,izvuceni_tekst[index])
        if len(sortirana_slova)>0:            
            x,y,w,h=izvuceni_koo[index]
            test_inputs = prepare_for_ann(sortirana_slova)
            result = model.predict(np.array(test_inputs, np.float32))
            procitano=display_result(result, alphabet)    
            text_file = open('result/'+str(redni)+"_stranica.txt", "a+")
            text_file.write("OBLAK %d:  %s \n\n" % (index,procitano))
            text_file.close()
   
razmak = cv2.imread('slova/razmak/razmak.png')
razmak = load_image(razmak)
razmak= image_gray(razmak)
razmak= image_bin(razmak)
alphabet = [0,1,2,3,4,5,6,7,8,9,'A','B','C','Č','Ć','D','Đ','E','F','G','H','I','J','K','/','L','M','N','O','P','R',' ','S','Š','T','U','V','W','X','-','Y','Z','Ž']

for loadedim in ucitaneImg:    
    
    
    image = load_image(loadedim)   
    original = image.copy()
    orginal_fin= image.copy()
    image_g = image_gray(image)
    image_b = image_bin(image_g)
    image_b_txt = image_bin_txt(image_g)
    original_b = image_b.copy()
    original_b_txt = image_b_txt.copy()

    prvaSlova = select_roi(image,image_b)
    oblacici = find_obl(prvaSlova)
    oblacici = dilate(oblacici)    
    oblacici = cv2.cvtColor(oblacici, cv2.COLOR_BGR2GRAY)   
    _, oblacici_b = cv2.threshold(oblacici, 0, 255, cv2.THRESH_OTSU)     
    original_b_txt = erode(original_b_txt)    
    
    koordinate_oblaci,naoblacena = find_cele_obl(original,original_b,oblacici_b)
    ime,izvuceni_tekstovi,izvuceni_koo = trazi_tekst(koordinate_oblaci,original_b_txt)        
    ispisi_tekst(izvuceni_tekstovi,orginal_fin,izvuceni_koo,redni) 
    
    redni+=1

# -*- coding: utf-8 -*-
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import glob
import os

plt.rcParams['figure.figsize'] = 16, 12
#config InlineBackend.figure_format='svg'

options={
    'model':'cfg/tiny-yolo-voc-3c.cfg',
    'load' :4000 ,
    'threshold' : 0.04,
    'gpu' : 0.7        
}
tfnet = TFNet(options)

d = {
'alan':'Alan Ford',
'bob':'Bob Rock',
'jedan':'Broj Jedan',
'sef': 'Sef'
}
redni=1
ucitaneImg = []
for im in glob.glob("for_reading/*.png"):
    n= cv2.imread(im)
    ucitaneImg.append(n)

for img in ucitaneImg:    
    tl_list=[]
    br_list=[]
    label_list=[]
    result=tfnet.return_predict(img)    
    for index in range(0, len(result)):
        tl=(result[index]['topleft']['x'],result[index]['topleft']['y'])
        br=(result[index]['bottomright']['x'],result[index]['bottomright']['y'])
        label = result[index]['label']
        tl_list.append(tl)
        br_list.append(br)
        label_list.append(label)        
        
    for ind in range(0,len(tl_list)):
        for i in range(0, len(tl_list)):            
            if tl_list[ind][0]>tl_list[i][0] and tl_list[ind][1]>tl_list[i][1] and br_list[ind][0]<br_list[i][0] and br_list[ind][1]<br_list[i][1]:
                break
        else:                
            label = d[label_list[ind]]
            img=cv2.rectangle(img,tl_list[ind],br_list[ind],(0,255,0),5)            
            img=cv2.putText(img,label,tl_list[ind],cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)
    #plt.imshow(img)
    #plt.show()
    ime = os.path.join('result',(str(redni)+'_stranica.png'))
    cv2.imwrite(ime,img)
    redni+=1
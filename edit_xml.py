# -*- coding: utf-8 -*-
from xml.etree import ElementTree as et
import glob

for i in glob.glob('sss/*.xml'):
    tree=et.parse(i)     
    root = tree.getroot()
    folder=root.findall('folder')[0]
    folder.text='images'
    pat=root.findall('path')[0]
    root.remove(pat)
    sors=root.findall('source')[0]
    root.remove(sors)
    filename=root.findall('filename')[0]
    name=filename.text.split('.')[0]    
    
    
    tree.write('annotations/'+name+'.xml')


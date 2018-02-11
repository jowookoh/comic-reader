# comic-reader

Ovaj repozitorijum sadrzi kod i materijale za prepoznavanje lica i teksta u stripu Alan Ford

U direkotrijumu result nalaze se primeri dobrog rada programa

Da bi se kod za **prepoznavanje teksta** mogao pokrenuti potrebno je:
1) instalirati tensorflow, opencv, keras
2) koristiti vec istrenirani model ANN za prepoznavanje slova - my_model.h5 (ili istrenirati novi model uz pomoc ocr_net.py skripte)
3) pokrenuti skriptu read_text.py


Da bi se kod za **detekciju i prepoznavanje lica** mogao pokrenuti potrebno je:
1) instalirati tensorflow 
2) preuzeti darkflow repozitorijum https://github.com/thtrieu/darkflow - koji sadrzi YOLO modele 
3) preuzeti tiny-yolo-voc weights sa sajta https://pjreddie.com/darknet/yolo/ i smestiti ih u bin direktorijum
4) instalirati cython pomocu python3 setup.py build_ext --inplace
5) konfiguraciju tiny-yolo-voc-3c.cfg premestiti u cfg direktorijum sa ostalim konfiguracijama
6) moguce je koristiti tezine koje sam ja dobio treniranjem mreze iz direktorijuma ckpt ili istrenirati svoje tezine
7) istrenirati tezine pomocu komande python flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation annotations --dataset images 
8) u skripti recognize.py, zameniti vrednost 4000 u recniku options sa vrednoscu koja je najveca u ckpt direktorijumu (istrenirane tezine se automatski smestaju u ckpt)
8) pokrenuti recognize.py skriptu

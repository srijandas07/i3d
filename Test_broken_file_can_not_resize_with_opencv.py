from os.path import basename
from os.path import join
from os.path import splitext
import os
import sys
import cv2
import glob

def main():
    #path
    ori_dir = "/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/"
    #folder name list
    name_list = open ('/data/stars/user/rdai/smarthomes/ls.txt')
    name_folder = name_list.read().split('\n')
    name_list.close()

    for name in name_folder:
        name_root1 = join(ori_dir, name)
        images = glob.glob(name_root1+ "/*")
        images.sort()
        for i in range(len(images)):
            try:
                #try to read and resize img
                print (images[i])
                img=cv2.resize(cv2.imread(images[i]), (224, 224))
                #print("sucess_read")
            #except (cv.error, OpenCV Error, cv2.error):
            except Exception as err:
                print (images[i])
                print("Broken!")
                with open("a_broken_file.txt","a+") as f:
                    f.write(str(images[i])+"\n")

if __name__ == '__main__':
    main()

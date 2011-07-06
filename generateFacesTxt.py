#!/usr/bin/python

import os

path='/home/cstucker/dev/facerec/data/facedb'
totalPpl = 0
facesTxt = open('faces.txt','w')

pDirList = os.listdir(path)
for personDir in pDirList:
    totalPpl += 1
    fDirList = os.listdir(path + '/' + personDir)
    for files in fDirList:
        facesTxt.write(str(totalPpl) + ',')
        facesTxt.write(personDir + ',')
        facesTxt.write(path + '/')
        facesTxt.write(personDir + '/')
        facesTxt.write(files + '\n')

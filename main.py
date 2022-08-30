import os
from kivy.app import App
import kivy.uix.image
import numpy as np
import treatments
from model import TensorFlowModel
import tensorflow as tf
import threading
from kivy.logger import Logger
from kivy.uix.label import Label
from kivy.logger import Logger
from android.permissions import Permission, request_permissions, check_permission
from android.storage import app_storage_path, primary_external_storage_path, secondary_external_storage_path
import time  
# from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
# import matplotlib.pyplot as plt
#from kivy_garden.graph import Graph, MeshLinePlot
import cv2
from pathlib import Path
from PIL import Image
from numpy import genfromtxt
import math


def log(msg):
  Logger.info(msg)

def check_permissions(perms):
  for perm in perms:
      if check_permission(perm) != True:
          return False
  return True


def detect_invalid_form(pathForm1,pathForm2,pathForm3,model):

  if(os.path.exists(pathForm1)):    # création d'une boucle while qui s'executera tant que True == True
        fin = open(pathForm1, "rt")
        # the output file which stores result
        fout = open(pathForm2, "wt")
        # iteration for each line in the input file
        for line in fin:  
            fout.write(line.replace(',', '.'))
        fin.close()
        fout.close()
  if(os.path.exists(pathForm2)):
      fin1 = open(pathForm2, "rt")
      fin2 = open(pathForm3, "wt")
      for line in fin1:  
            fin2.write(line.replace(' ', ','))
      fin1.close()
      fin2.close()
      
  def distancepix(xi1, yi1, x2, y2):
      return math.sqrt(pow((x2 - xi1), 2) + pow((y2 - yi1), 2))


  def plusproche(xi, yi, x2, y2):
      dis = [];
      xi1 = xi - 1
      yi1 = yi - 1
      d1 = distancepix(xi1, yi1, x2, y2)
      dis.append(d1)

      xi2 = xi
      yi2 = yi - 1
      d2 = distancepix(xi2, yi2, x2, y2)
      dis.append(d2)

      xi3 = xi + 1
      yi3 = yi - 1
      d3 = distancepix(xi3, yi3, x2, y2)
      dis.append(d3)

      xi4 = xi - 1
      yi4 = yi
      d4 = distancepix(xi4, yi4, x2, y2)
      dis.append(d4)

      xi5 = xi + 1
      yi5 = yi
      d5 = distancepix(xi5, yi5, x2, y2)
      dis.append(d5)

      xi6 = xi - 1
      yi6 = yi + 1
      d6 = distancepix(xi6, yi6, x2, y2)
      dis.append(d6)

      xi7 = xi
      yi7 = yi + 1
      d7 = distancepix(xi7, yi7, x2, y2)
      dis.append(d7)

      xi8 = xi + 1
      yi8 = yi + 1
      d8 = distancepix(xi8, yi8, x2, y2)
      dis.append(d8)

      c = dis.index(min(dis))

      if (c == 0):
          xn = xi1
          yn = yi1
      elif (c == 1):
          xn = xi2
          yn = yi2
      elif (c == 2):
          xn = xi3
          yn = yi3
      elif (c == 3):
          xn = xi4
          yn = yi4
      elif (c == 4):
          xn = xi5
          yn = yi5
      elif (c == 5):
          xn = xi6
          yn = yi6
      elif (c == 6):
          xn = xi7
          yn = yi7
      elif (c == 7):
          xn = xi8
          yn = yi8
      return xn, yn


  def exten1(im1, y, x):
      im = im1
      im[y, x] = 0

      im[y, x + 1] = 0
      im[y - 1, x + 1] = 0
      im[y - 1, x] = 0

      im[y - 1, x - 1] = 0

      im[y, x - 1] = 0

      im[y + 1, x - 1] = 0

      im[y + 1, x] = 0

      im[y + 1, x + 1] = 0

      im[y, x + 2] = 0

      im[y - 2, x] = 0

      im[y, x - 2] = 0

      im[y + 2, x] = 0
      return im
  data = genfromtxt(pathForm3, delimiter=',', dtype=float)
  tline1 = "/storage/emulated/0/Documents/Khouloud/im8.tif";
  X = cv2.imread(tline1);
  X[:,:] = 255;
  for k in range(len(data)):
      data[k, 0] = round(data[k, 0])
      data[k, 1] = round(data[k, 1])

  im = cv2.resize(X, (84, 84), interpolation=cv2.INTER_AREA)


  matr = data
  x = data[:, 0]
  y = data[:, 1]

  l = len(matr)
  c = len(matr[0])
  limit = []
  for i in range(1, l - 1):
      if ((matr[i, 0] == 0 and matr[i, 1] == 0 and matr[i + 1, 0] != 0 and matr[i + 1, 1] != 0) or (
              matr[i, 0] == 0 and matr[i, 1] == 0 and matr[i - 1, 0] != 0 and matr[i - 1, 1] != 0)):
          limit.append(i)
  x11 = []
  y11 = []
  for ii in range(len(x)):
      if (x[ii] != 0):
          x11.append(x[ii])
          y11.append(y[ii])

  minxf = min(x11);
  maxxf = max(x11);
  minyf = min(y11);
  maxyf = max(y11);

  xf = []
  yf = []

  ns = len(limit)

  for ii in range(0, ns, 2):
      xs = matr[limit[ii] + 1: limit[ii + 1], 0]
      ys = matr[limit[ii] + 1: limit[ii + 1], 1]

      x = xs
      y = ys
      xf = []
      yf = []
      for i in range(0, len(x) - 2):
          x1 = x[i]
          x2 = x[i + 1]
          y1 = y[i]
          y2 = y[i + 1]
          xf.append(x1)
          yf.append(y1)
          xi = x1
          yi = y1
          while ((xi != x2) or (yi != y2)):
              xn, yn = plusproche(xi, yi, x2, y2)

              xf.append(xn)
              yf.append(yn)
              xi = xn
              yi = yn
          xf = xf[0:len(xf) - 1]  ####
          yf = yf[0:len(yf) - 1]
      for j in range(len(xf)):
          xf[j] = math.ceil(((xf[j] - minxf + 1) * 58) / (maxxf - minxf + 1)) + 3
          yf[j] = math.ceil(((yf[j] - minyf + 1) * 58) / (maxyf - minyf + 1)) + 3
      c = len(xf)
      for i in range(c):
          # print(xf[i])
          # time.sleep(3)
          im = exten1(im, yf[i], xf[i])

  im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
  path2 = '/storage/emulated/0/Documents/Khouloud/letter.png'
  cv2.imwrite(path2, im);

  test_img = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE) )


  test_img = test_img / 255.0
  #new_model = tf.keras.models.load_model('/storage/emulated/0/Documents/Khouloud/trained_model.h5')
  model.load(os.path.join(os.getcwd(), 'model.tflite'))
  #model1=model.load(os.path.join('/storage/emulated/0/Documents/Khouloud/', 'model.tflite'))
  samples_to_predict = []
  samples_to_predict.append(test_img)
  samples_to_predict = np.array(samples_to_predict)
  predictions = model.pred(samples_to_predict)

  classes = np.argmax(predictions, axis = 1)
  # print(classes)
  # print(predictions[0][classes])
  return str(classes)


def Analyse():

  perms = [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
    
  if  check_permissions(perms)!= True:
      request_permissions(perms)    # get android permissions     
      exit()
  
  model = TensorFlowModel()
  declenche =True    # app has to be restarted; permissions will work on 2nd start
  while (declenche):
        fileName = r"/storage/emulated/0/Documents/Khouloud/input.txt"
        if(os.path.exists(fileName)):    # création d'une boucle while qui s'executera tant que True == True
          fin = open(fileName, "rt")
          # the output file which stores result
          fout = open("/storage/emulated/0/Documents/Khouloud/input2.txt", "wt")
          # iteration for each line in the input file
          for line in fin:
              # replacing the string and write to output file
              fout.write(line.replace(',', '.'))
          #closing the input and output files
          fin.close()
          fout.close()
          file_out="/storage/emulated/0/Documents/Khouloud/input2.txt"
          input=open(file_out,"r")
          l=input.readlines()
          input.close()
          s="0 0 0 0"+"\n"
          l.insert(0,s)
          input=open(file_out,"w")
          l = "".join(l)
          input.write(l)
          input.close()
          with open(file_out,'r') as rf:
              lines = rf.readlines()
          line_number=0
          with open(file_out,'w') as write_file:
              for line in lines:
                  line_number =line_number+1
                  if line_number == len(lines):
                      pass
                  else:
                      write_file.write(line)
          m=np.loadtxt(file_out)
          with open('/storage/emulated/0/Documents/Khouloud/expected.txt', 'r') as f:
              expected = f.read() 
          reference="/storage/emulated/0/Documents/Khouloud/ref_"+expected+".inkml"
          ref = treatments.lecture_online(reference) 
          trace = m
          path1='/storage/emulated/0/Documents/Khouloud/trained_model.h5'
          path2='/storage/emulated/0/Documents/Khouloud/trace.png'
          
          pathForm1='/storage/emulated/0/Documents/Khouloud/traceForm.txt'
          pathForm2='/storage/emulated/0/Documents/Khouloud/trace_inter.txt'
          pathForm3='/storage/emulated/0/Documents/Khouloud/tracesForm.txt'

          result1=treatments.analyse_penlift(ref,trace)
          result2=treatments.detect_overflow(trace,expected)
          result3=treatments.detect_reverse_direction(trace,expected)
          result4=treatments.detect_invalid_order(ref,trace,expected)
          result5=detect_invalid_form(pathForm1,pathForm2,pathForm3,model)
          penlifts =bytes(result1, encoding = "utf-8")
          overflows =bytes(result2, encoding = "utf-8")
          directions =bytes(result3, encoding = "utf-8")
          orders =bytes(result4, encoding = "utf-8")
          forms =bytes(result5, encoding = "utf-8")
          try:
              Logger.info('Got requested permissions')    
            
              fname1 = os.path.join( primary_external_storage_path(),'Documents/Khouloud/penlifts.txt')
              log('writing to: %s' %fname1)
            
              with open(fname1, 'wb') as f1:  # write testfile
                  f1.write(penlifts)

              fname2 = os.path.join( primary_external_storage_path(),'Documents/Khouloud/overflows.txt')
              log('writing to: %s' %fname2)
            
              with open(fname2, 'wb') as f2:        # write testfile
                  f2.write(overflows)

              fname3 = os.path.join( primary_external_storage_path(),'Documents/Khouloud/directions.txt')
              log('writing to: %s' %fname3)
            
              with open(fname3, 'wb') as f3:        
                f3.write(directions)
              
              fname4 = os.path.join( primary_external_storage_path(),'Documents/Khouloud/order.txt')
              log('writing to: %s' %fname4)
              
              with open(fname4, 'wb') as f4:        
                f4.write(orders)
              
              fname5 = os.path.join( primary_external_storage_path(),'Documents/Khouloud/form.txt')
              log('writing to: %s' %fname5)
              
              with open(fname5, 'wb') as f5:        
                f5.write(forms)
                
          except:
              log('could not write to external storage ... missing permissions ?')    
          #return "succes"
        else:
          time.sleep(2)
class MyApp(App):
    def build(self):
        Analyse()
        return Label(text = '' )   # <---- calling testwrite() here
MyApp().run()

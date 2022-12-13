import cv2, numpy as np
from fastai.vision.all import *
from fastai.vision import *
from PIL import Image
from fastai.vision.widgets import Image
import tensorflow as tf
from collections import deque

'''
1. add your imports
2. load your model
'''
votes_max = 20
votes = deque([], maxlen=votes_max)

fastai_model = load_learner("fa_export_3_pretrained.pkl")
fastai_model.no_logging()
fastai_model.no_bar()
if torch.cuda.is_available():
  fastai_model.dls.to(device='cuda')

sequental_model = tf.keras.models.load_model('seq_model2/export_model2')

efficientnet_model = tf.keras.models.load_model('efficientnet/export_model')

labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
selected_models = ['fastai', 'EfficientNET', 'Sequential']
webcam = cv2.VideoCapture(0)
print("width", webcam.get(cv2.CAP_PROP_FRAME_WIDTH)) #width
print("height", webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height

select_model = 0
def on_slider_change(x):
  pass

(_, im) = webcam.read() #im is BGR
cv2.imshow('frame',im)

cv2.createTrackbar('Select model', 'frame', 0, 2, on_slider_change)

while True:
  select_model = cv2.getTrackbarPos('Select model','frame')

  (_, im) = webcam.read() #im is BGR

  im = cv2.flip(im, flipCode = 1)
  cv2.rectangle(im, (80, 80), (354, 354), (0, 255, 0), 2)

  hand = im[80:354, 80:354]
  hand = cv2.resize(hand, (512,512)) #512 is image size 

  rgb = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
  fai_channels = rgb.transpose(2,0,1)

  if select_model == 0:
    with fastai_model.no_bar(), fastai_model.no_logging():
      pred_class,pred_idx,outputs = fastai_model.predict(tensor(rgb.astype(np.uint8)))
  elif select_model == 1:
    img = np.expand_dims(rgb, axis=0)
    pred_idx = np.argmax(efficientnet_model.predict(img))
  elif select_model == 2:
    img = np.expand_dims(rgb, axis=0)
    pred_idx = np.argmax(sequental_model.predict(img))

  votes.append(labels[pred_idx])
  vote_counter = Counter(votes).most_common(1)
  
  prediction = vote_counter[0][0]
  prediction_percentage = vote_counter[0][1] / votes_max
  
  cv2.putText(im, prediction + "(%.2f)" % (prediction_percentage) + ":" + selected_models[select_model], (40, 380), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
  r = im[1]


  cv2.imshow('frame',im)
  key = cv2.waitKey(10)
  if key == 27:
      break

  # print(fai_channels)
  # print(im.shape)
  # print(im.transpose(2,0,1).shape)
  # pred_class,pred_idx,outputs = model.predict(tensor(rgb.astype(np.uint8)))
  # print(labels[pred_idx])
  # print(pred_idx)
  # print(pred_class,pred_idx,outputs)
  # rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  # b,g,r = cv2.split(im)
  # cv2.imshow('OpenCV', im)
  # print("====")
  # print(type(im.astype(np.uint8)))
  # data = torch.tensor(np.ascontiguousarray(np.flip(im, 2)).transpose(2,0,1))
  # print(data)
  # pred_class,pred_idx,outputs = model.predict(data)
  # print(pred_class.obj)
  	
  # print(tensor(im).shape)

  # hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
  # # define range of blue color in HSV
  # lower_blue = np.array([110,50,50])
  # upper_blue = np.array([130,255,255])
  # # Threshold the HSV image to get only blue colors
  # mask = cv2.inRange(hsv, lower_blue, upper_blue)
  # # Bitwise-AND mask and original image
  # res = cv2.bitwise_and(im,im, mask= mask)
  # cv2.imshow('frame',im)
  # cv2.imshow('mask',mask)
  # cv2.imshow('res',res)
cv2.destroyAllWindows()












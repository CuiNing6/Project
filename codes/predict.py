import numpy as np
from PIL import Image
import os
import csv
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

# 指定图片尺寸
target_size = (229, 229) #fixed size for InceptionV3 architecture

# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]

def write_to_csv(aug_softmax,imggs):
    with open('resultb.csv', 'a',newline='',encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile,dialect='excel')
        for c in range(0,30):
            spamwriter.writerow([int(str(imggs).split('.')[0]), c+1, preds_output[c]])

model = load_model('my_model3.h5')# 载入模型

imgDir='E:/Pig_Identification_project/database/test_img'
imgFoldName='test_B'
imgs = os.listdir(imgDir+imgFoldName)
imgNum = len(imgs)
# preds_all=np.zeros([3000,30])
for i in range (imgNum):
    imgg = Image.open(imgDir+imgFoldName+"/"+imgs[i])
    preds = predict(model, imgg, target_size)
    preds = np.array(preds)
    preds_s=np.zeros((30,))
    preds_s[0]=preds[0]
    preds_s[9]=preds[1]
    preds_s[10]=preds[2]
    preds_s[11]=preds[3]
    preds_s[12]=preds[4]
    preds_s[13]=preds[5]
    preds_s[14]=preds[6]
    preds_s[15]=preds[7]
    preds_s[16]=preds[8]
    preds_s[17]=preds[9]
    preds_s[18]=preds[10]
    preds_s[1]=preds[11]
    preds_s[19]=preds[12]
    preds_s[20]=preds[13]
    preds_s[21]=preds[14]
    preds_s[22]=preds[15]
    preds_s[23]=preds[16]
    preds_s[24]=preds[17]
    preds_s[25]=preds[18]
    preds_s[26]=preds[19]
    preds_s[27]=preds[20]
    preds_s[28]=preds[21]
    preds_s[2]=preds[22]
    preds_s[29]=preds[23]
    preds_s[3]=preds[24]
    preds_s[4]=preds[25]
    preds_s[5]=preds[26]
    preds_s[6]=preds[27]
    preds_s[7]=preds[28]
    preds_s[8]=preds[29]

    preds_sum=np.sum(preds_s)
    print(i)
    print(preds_sum)
    preds_output=preds_s/preds_sum
    print(preds_output.shape)
    preds_output = np.array(preds_output)
    for j in range(0,30):
        preds_output[j] = preds_output[j]/1.000001
        if preds_output[j]>=1-10E-15:
            preds_output[j]=1-10E-15
        elif preds_output[j]<=10E-15:
            preds_output[j]=10E-15
    print(preds_output.shape)
    imggs=imgs[i]
    write_to_csv(preds_output,imggs)



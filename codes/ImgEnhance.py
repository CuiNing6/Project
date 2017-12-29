from PIL import Image
from PIL import ImageEnhance
import os
import glob

def rename(path):
    filelist=glob.glob(os.path.join(path,'*.jpg'))

    i = 1

    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)

            dst = os.path.join(os.path.abspath(
                path), '01' + format(str(i), '0>4s') + '.jpg')
            os.rename(src, dst)
            i = i + 1

def brightness(filess,root_path):

    for image in filess:
        img = Image.open(image)
        #亮度增强
        enh_bri = ImageEnhance.Brightness(img)
        brightness = 1.5
        image_brightened = enh_bri.enhance(brightness)
        image_brightened.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def color(filess,root_path):

    for image in filess:
        img = Image.open(image)
        #色度增强
        enh_col = ImageEnhance.Color(img)
        color = 1.5
        image_colored = enh_col.enhance(color)
        image_colored.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def Contrast(filess,root_path):

    for image in filess:
        img = Image.open(image)
        #对比度增强
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        image_contrasted = enh_con.enhance(contrast)
        image_contrasted.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def Sharpness(filess,root_path):

    for image in filess:
        img = Image.open(image)
        #锐度增强
        enh_sha = ImageEnhance.Sharpness(img)
        sharpness = 5.0
        image_sharped = enh_sha.enhance(sharpness)
        image_sharped.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def FLIP_LEFT_RIGHT(filess,root_path):

    for image in filess:
        img = Image.open(image)
        left_right = img.transpose(Image.FLIP_LEFT_RIGHT)
        left_right.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def ROTATE_90(filess,root_path):

    for image in filess:
        img = Image.open(image)
        left_right = img.transpose(Image.ROTATE_90)
        left_right.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)

def ROTATE_270(filess,root_path):

    for image in filess:
        img = Image.open(image)
        left_right = img.transpose(Image.ROTATE_270)
        left_right.save(os.path.join(root_path,os.path.basename(image)))
        rename(root_path)


if __name__ == '__main__':

    # for i in range(0,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('brightness...')
    #     brightness(filess,root_dir)
    # for i in range(0,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('color...')
    #     color(filess,root_dir)
    # for  i in range(16,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('Contrast...')
    #     Contrast(filess,root_dir)
    # for i in range(0,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('Sharpness...')
    #     Sharpness(filess,root_dir)
    # for i in range(0,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('FLIP_LEFT_RIGHT...')
    #     FLIP_LEFT_RIGHT(filess,root_dir)
    # for i in range(0,30):
    #     i=i+1
    #     print(i)
    #     root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
    #     dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
    #     filess=glob.glob(os.path.join(dir,'*.jpg'))
    #     print('ROTATE_90...')
    #     ROTATE_90(filess,root_dir)
    for i in range(26,27):
        i=i+1
        print(i)
        root_dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_final'+'/'+str(i)
        dir='I:/Deep Learning/deeplearning.ai-master/COURSE 4 Convolutional Neural Networks/Week 04/Face Recognition/train_5img'+'/'+str(i)
        filess=glob.glob(os.path.join(dir,'*.jpg'))
        print('ROTATE_270...')
        ROTATE_270(filess,root_dir)

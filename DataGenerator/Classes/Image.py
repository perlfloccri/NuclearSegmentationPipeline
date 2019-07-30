import cv2
import numpy as np
from tifffile import imread
import random
from scipy.signal import medfilt2d
import scipy.fftpack as fp
import math
from scipy.ndimage import gaussian_filter, morphology
from random import randint
import skimage.transform as ski_transform
import os

class AnnotatedObject:
    mean_x = 0
    mean_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    img_ind = 0

    def __init__(self,img_ind = 0,obj_ind = 0, mean_x = None, mean_y = None, min_x = None, min_y = None, max_x = None, max_y = None):
        self.img_ind = img_ind
        self.obj_ind = obj_ind
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

class Image:

    raw = 0

    def pre_process_img(img, color='gray'):
        if color is 'gray':
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                print("Error in Conversion to grayscale: maybe already grayscale?")
        elif color is 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            pass

        img = img.astype(np.float32)
        img /= 255.0
        return img

    def getRaw(self):
        return self.raw

class AnnotatedImage(Image):

    mask = 0

    def readFromPath(self,image_path,mask_path,type=None):
        self.raw = Image.pre_process_img(imread(image_path),color='gray')
        if type=='uint16':
            self.mask = imread(mask_path).astype(np.uint16)
        else:
            self.mask = imread(mask_path).astype(np.uint8)
    def createWithArguments(self,image,mask):
        self.raw = image
        self.mask = mask
    def getMask(self):
        return self.mask
    def getCroppedAnnotatedImage(self,annotation):
        tmp = AnnotatedImage()
        tmp.createWithArguments(self.raw[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y] * (self.mask[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y]==annotation.obj_ind).astype(np.uint8),self.mask[annotation.min_x:annotation.max_x,annotation.min_y:annotation.max_y]==annotation.obj_ind)
        return tmp
    def getMeanMaskObjectSize(self):
        total_sum = np.uint64(0);
        for i in range(self.mask.max()):
           total_sum = total_sum + np.square((self.mask==(i+1)).sum())
        A = np.sqrt(total_sum / self.mask.max())
        return (2*np.sqrt(A/np.pi)).astype(np.int16)

    def createBackground(self,width,height,bg):
        number = bg.__len__()
        try:
            raw=np.random.choice(bg, (width, height),replace=False)
        except:
            raw = np.random.choice(bg, (width, height), replace=True)
        return gaussian_filter(raw, sigma=3)

    def filterLowFrequencies(self,img=None,n=50):
        if img is None:
            image = self.raw
        else:
            image = img
        F1 = fp.fft2(image.astype(float))
        F2 = fp.fftshift(F1)

        (w, h) = image.shape

        if (0):
            # Cutting of Frequencies
            half_w, half_h = int(w/2), int(h/2)
            # low pass filter
            a=np.ones((w,h))
            a[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0
            a = np.abs(1-a).astype(np.complex128)

        if (1):
            # Gauss filtering of frequencies
            x, y = np.meshgrid(np.linspace(-math.ceil(w/2), math.ceil(w/2), w), np.linspace(-math.ceil(h/2), math.ceil(h/2), h))
            d = np.sqrt(x * x + y * y)
            sigma, mu = float(n), 0.0
            a = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            a = a.astype(np.complex128)

        F2 = F2 * a# select all but the first 50x50 (low) frequencies

        im1 = fp.ifft2(fp.ifftshift(F2)).real
        if img is None:
            self.raw = im1 #medfilt2d(im1,3)
        else:
            return im1

    def readFromPathOnlyImage(self,image_path):
        if os.path.basename(image_path).split('.')[1] == 'jpg':
            self.raw = Image.pre_process_img(cv2.imread(image_path),color='gray')
        else:
            self.raw = Image.pre_process_img(imread(image_path),color='gray')
        self.mask = np.zeros_like(self.raw,dtype=np.uint8)

class ArtificialAnnotatedImage(AnnotatedImage):

    running_mask = 1
    number_nuclei = 0
    griddy = 0

    def __init__(self,width=None,height=None, number_nuclei = None, probabilityOverlap=0,background=None):
        if background is not None:
            #s = np.random.poisson(1, (256, 256))
            #self.raw = np.random.uniform(low=background[0],high=background[1],size=(256,256))
            #self.raw = np.random.uniform(low=background[0], high=background[1], size=(256,256)) #
            self.raw = self.createBackground(width,height,background)

        else:
            self.raw=np.zeros((height,width))
        self.mask = np.zeros((height,width))
        self.tmp_mask =  np.zeros((height,width))
        self.number_nuclei = number_nuclei
        self.griddy = gridIterable(width=width,height=height,nrObjects=number_nuclei,probabilityOverlap=probabilityOverlap)

    def addImageAtRandomPosition(self,image):
        rand_x = random.randint(0,int(self.raw.shape[0]-image.getRaw().shape[0]))
        rand_y = random.randint(0,int(self.raw.shape[1] - image.getRaw().shape[1]))

        [x, y] = np.where(image.getMask() == 1)
        self.raw[x+rand_x, y+rand_y] = image.getRaw()[x, y]
        self.running_mask = self.running_mask + 1
        self.mask[x+rand_x, y+rand_y] = image.getMask()[x, y] + self.running_mask

    def addImageAtGridPosition(self,image):
        pos = self.griddy.next()
        #print('minx: ' + str(pos.minx) + ', maxx: ' + str(pos.maxx) + ', miny: ' + str(pos.miny) + ', maxy: ' + str(pos.maxy))
        rand_x = random.randint(pos.minx,pos.maxx)
        rand_y = random.randint(pos.miny,pos.maxy)
        #[x, y] = np.where(image.getMask() == 1)
        tmp_image = image.getRaw()
        tmp_mask = image.getMask()
        eroded_mask = morphology.binary_erosion(tmp_mask.astype(bool), structure=np.ones((3, 3), dtype=np.int))
        tmp_image = tmp_image * eroded_mask
        tmp_mask = eroded_mask
        #tmp_image = ski_transform.resize(tmp_image, (int(tmp_image.shape[0] + 3), int(tmp_image.shape[1] + 3)), mode='reflect')
        #tmp_mask = (ski_transform.resize(eroded_mask, (int(eroded_mask.shape[0] + 3), int(eroded_mask.shape[1] + 3)),mode='reflect') > 0.5)
        [x, y] = np.where(tmp_mask == 1)

        added = 0
        visible = random.randint(0, 1)
        for i in range(0,x.__len__()):
            try:
                if (((x[i] + rand_x) > 0 ) & ((y[i] + rand_y) > 0 )):
                    added = 1
                    if (self.mask[x[i]+ rand_x, y[i] + rand_y] == 0):
                        self.raw[x[i] + rand_x, y[i] + rand_y] = tmp_image[x[i], y[i]]
                        self.mask[x[i] + rand_x, y[i] + rand_y] = tmp_mask[x[i], y[i]] * self.running_mask
                    else:

                        if visible == 1:
                            self.raw[x[i] + rand_x, y[i] + rand_y] += tmp_image[x[i], y[i]] * 0.2 #random.uniform(0.05,0.2)
                        else:
                            self.raw[x[i] + rand_x, y[i] + rand_y] = tmp_image[x[i], y[i]]
                            self.mask[x[i] + rand_x, y[i] + rand_y] = tmp_mask[x[i], y[i]] * self.running_mask

                    # TMP mask for creation of smooth borders
            except:
                e=0
        if added == 1:
            tmp = self.mask == self.running_mask
            self.tmp_mask *= (~tmp).astype(self.tmp_mask.dtype)
            self.tmp_mask += morphology.binary_erosion(tmp.astype(bool),structure=np.ones((3,3),dtype=np.int).astype(self.tmp_mask.dtype))
            self.running_mask = self.running_mask + 1

        return 1

    def transformToArtificialImage(image=None,useBorderObjects=False,background=None):
        if background is not None:
            raw = image.createBackground(image.getRaw().shape[0],image.getRaw().shape[1],background)
        else:
            raw = np.zeros((image.getRaw().shape[0],image.getRaw().shape[1]))
        mask = np.zeros((image.getRaw().shape[0],image.getRaw().shape[1]))
        running_mask = 0
        #for i in range(1, image.getMask().max()+1):
        for i in np.unique(image.getMask()):
            if i > 0:
                [x, y] = np.where(image.getMask() == i)
                #if (x.__len__() > 0):
                #if (~useBorderObjects): # border objects excluding could be implemented
                #    if ~((x.min() == 0) | (y.min() == 0) | (x.max() == image.getRaw().shape[0]) | (y.max() == image.getRaw().shape[1])):
                raw[x, y] = image.getRaw()[x, y]
                running_mask = running_mask + 1
                mask[x, y] = image.getMask()[x, y] + running_mask
        ret_img = AnnotatedImage()
        ret_img.createWithArguments(raw,mask)
        return ret_img

class AnnotatedObjectSet:
    #images = []
    #objects = []
    #path_to_imgs = []

    def __init__(self):
        self.images = []
        self.path_to_imgs = []
        self.objects = []

    def addObjectImage(self,image=None,useBorderObjects=False, path_to_img=None,tissue=None,scale=None,is_spot=False):
        self.images.append(image)
        if path_to_img:
            self.path_to_imgs.append(path_to_img)
        curr_img_index = self.images.__len__() - 1
        if (scale) == None and (tissue == None):
            thresh = 0
        if scale== 1:
            if tissue == 'Ganglioneuroma':
                thresh = 500
            else:
                thresh = 1000
        else:
            if tissue == 'Ganglioneuroma':
                thresh = 30
            else:
                thresh = 500
        # overrule thresh for spot image
        if is_spot:
            thresh = 0
        for i in range(1,image.getMask().max()+1):
            if not is_spot:
                [x, y] = np.where(cv2.dilate((image.getMask() == i).astype(np.uint8),np.ones((5,5),np.uint8),iterations = 1))
            else:
                [x, y] = np.where(image.getMask() == i)
            if (x.__len__() > thresh):
                if ((x.min() < 3) | (y.min() < 3) | (x.max() > (image.getRaw().shape[0] - 3)) | (y.max() > (image.getRaw().shape[1] - 3))):
                    if useBorderObjects:
                        self.objects.append(AnnotatedObject(img_ind = curr_img_index, obj_ind = i, mean_x = x.mean(),mean_y = y.mean(),min_x = x.min(), min_y = y.min(),max_x = x.max(), max_y = y.max()))
                else:
                    self.objects.append(AnnotatedObject(img_ind=curr_img_index, obj_ind=i, mean_x=x.mean(), mean_y=y.mean(),min_x=x.min(), min_y=y.min(), max_x=x.max(), max_y=y.max()))

    def returnArbitraryObject(self):
        rand_int = random.randint(0,self.objects.__len__()-1)
        return self.images[self.objects[rand_int].img_ind].getCroppedAnnotatedImage(self.objects[rand_int])

    def returnObjectByIndex(self,int):
        return self.images[self.objects[int].img_ind].getCroppedAnnotatedImage(self.objects[int])

    def initializeReturnArbitraryUniqueObjects(self):
        self.available_indizes = list(range(0,self.objects.__len__() - 1))

    def returnArbitraryBrightObject(self,thresh=0.5,unique=False):
        mean = 0
        while (mean <= thresh):
            rand_int = random.randint(0, self.available_indizes.__len__()-1)
            index = self.available_indizes[rand_int]
            img = self.images[self.objects[rand_int].img_ind].getCroppedAnnotatedImage(self.objects[rand_int])
            values = np.where(img.getMask() == 1)
            mean = img.getRaw()[values].flatten().mean()
            #if ((mean <= thresh) or (unique)):
             #   del self.available_indizes[rand_int]
        #print ("Remaining objects in list: " + str(self.available_indizes.__len__()))
        return img

class gridIterable:

    def __init__(self,width=0,height=0,nrObjects=0,probabilityOverlap = 0):
        self.width=width
        self.height = height
        self.nrObjects = nrObjects
        self.nr_x = self.nr_y = np.sqrt(nrObjects)
        self.stepX = self.width / (self.nr_x-1)
        self.stepY = self.height / (self.nr_y-1)
        self.probabilityOverlap = 1-probabilityOverlap
        self.curr_ind = 0

    def __iter__(self):
        return self

    def next(self):
        if (self.curr_ind <= (self.nrObjects)):
            self.curr_ind = self.curr_ind + 1
            row = np.ceil(self.curr_ind / self.nr_x)
            column = self.curr_ind - self.nr_x * (row-1)
            a = Rectangle(minx=round(0+(column-1)*self.stepX + (self.stepX/2) * self.probabilityOverlap),maxx=round(0+column*self.stepX - (self.stepX/2) * self.probabilityOverlap-1),miny=round(0+(row-1)*self.stepY + (self.stepY/2) * self.probabilityOverlap),maxy=round(0+row*self.stepY - (self.stepY/2) * self.probabilityOverlap-1))

            return a
        else:
            raise StopIteration()

class Rectangle:
    def __init__(self,minx=None,maxx=None,miny=None,maxy=None):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy

class MosaicImage:
    def __init__(self,image,filtersize):
        self.image=image
        self.mosaic = np.zeros_like(image,dtype=image.dtype)
        self.filtersize=filtersize

    def calculate_mean(self,x,y):
        dist_x = self.filtersize[0]
        dist_y = self.filtersize[1]
        values_r = self.image[x:x+dist_x,y+dist_y,0].flatten()
        values_g = self.image[x:x + dist_x,y:y + dist_y, 1].flatten()
        values_b = self.image[x:x + dist_x,y:y + dist_y, 2].flatten()
        return (int(values_r.mean()),int(values_g.mean()),int(values_b.mean()))

    def placeImage(self,image,colors,position):
        dist_x = self.filtersize[0]
        dist_y = self.filtersize[1]
        img = ski_transform.resize(image, (dist_x,dist_y), mode='reflect')
        self.mosaic[position[0]:position[0]+dist_x,position[1]:position[1]+dist_y,0] = img * colors[0]
        self.mosaic[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,1] = img * colors[1]
        self.mosaic[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,2] = img * colors[2]

    def placeImageUsingOrigImage(self,image,position):
        dist_x = self.filtersize[0]
        dist_y = self.filtersize[1]
        img = ski_transform.resize(image, (dist_x, dist_y), mode='reflect')
        self.mosaic[position[0]:position[0]+dist_x,position[1]:position[1]+dist_y,0] = img * self.image[position[0]:position[0]+dist_x,position[1]:position[1]+dist_y,0]
        self.mosaic[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,1] = self.image[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,1]
        self.mosaic[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,2] = self.image[position[0]:position[0] + dist_x, position[1]:position[1] + dist_y,2]
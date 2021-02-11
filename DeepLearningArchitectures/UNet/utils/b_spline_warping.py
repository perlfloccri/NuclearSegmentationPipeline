
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ultrasound_data import TRAIN_DATA_FILE, AUG_TRAIN_DATA_FILE, IMG_SHAPE, VAL_SET

from scipy import ndimage
from scipy import interpolate
from scipy.ndimage.interpolation import map_coordinates

GRID_SIZE_X = 116
GRID_SIZE_Y = 105
MAX_DEFORM = 15
N_VARIATIONS = 9
MAX_SHIFT = 20

SHOW_SAMPLES = False


def random_padding(Image, Mask):
    """ randomly pad image and crop back to original size """

    padded_i = np.pad(Image, ((MAX_SHIFT // 2, MAX_SHIFT // 2), (MAX_SHIFT // 2, MAX_SHIFT // 2)), mode='constant')
    padded_m = np.pad(Mask, ((MAX_SHIFT // 2, MAX_SHIFT // 2), (MAX_SHIFT // 2, MAX_SHIFT // 2)), mode='constant')
    crops = np.random.random_integers(0, high=MAX_SHIFT, size=(1, 2))
    cropped_image = padded_i[crops[0, 0]:(crops[0, 0] + Image.shape[0]), crops[0, 1]:(crops[0, 1] + Image.shape[1])]
    cropped_mask = padded_m[crops[0, 0]:(crops[0, 0] + Image.shape[0]), crops[0, 1]:(crops[0, 1] + Image.shape[1])]

    return cropped_image, cropped_mask


def transforms_roi_scaled128():
    """ create transforms """

    from ultrasound_data import Scale, Crop, Combiner
    crop_img = Crop((420, 580), 15, 365, 165, 515)

    src_shape, dst_shape = (350, 350), (128, 128)
    transform_img = Scale(src_shape, dst_shape, True)
    transform_mask = Scale(src_shape, dst_shape, False)

    transform_img = Combiner(crop_img, transform_img)
    transform_mask = Combiner(crop_img, transform_mask)

    return transform_img, transform_mask, dst_shape


if __name__ == '__main__':
    """ main """

    transform_img, transform_mask, dst_shape = transforms_roi_scaled128()

    working_dir = '.'
    data_file = os.path.join(working_dir, TRAIN_DATA_FILE)
    aug_data_file = os.path.join(working_dir, AUG_TRAIN_DATA_FILE)

    data = np.load(data_file)
    n_samples = len(data)
    
    augmented_data = []
    
    # iterate train set
    start = time.time()
    for i in xrange(n_samples):
        print "processing image", i, "of", n_samples
        
        # extract data
        patient_id, img_id = data[i][0], data[i][1]
        I_orig = data[i][2]
        M_orig = data[i][3]
        
        augmented_data.append((patient_id, img_id, transform_img(I_orig), transform_mask(M_orig)))
        
        # do not augment validation set
        if patient_id in VAL_SET:
            continue
        
        # lay grid over test image
        if SHOW_SAMPLES:
            for i in xrange(0, I_orig.shape[1] + 1, GRID_SIZE_X):
                I_orig[:, i-2:i+3] = 0.0
            
            for i in xrange(0, I_orig.shape[0] + 1, GRID_SIZE_Y):
                I_orig[i-2:i+3, :] = 0.0        
        
        # create variations
        for i_var in xrange(N_VARIATIONS):
            
            # get image shape
            shape = I_orig.shape

            # get random crop of image
            I_cropped, M_cropped = random_padding(I_orig, M_orig)
            
            # define spline grid
            sx = np.arange(0, I_cropped.shape[1] + 1, GRID_SIZE_X)
            sy = np.arange(0, I_cropped.shape[0] + 1, GRID_SIZE_Y)
            
            # initialize deformation field
            sdx = np.zeros((len(sx), len(sy)))
            sdy = np.zeros((len(sx), len(sy)))
            
            # add random deformations
            for ix in xrange(len(sx)):
                for iy in xrange(len(sy)):
                    sdx[ix, iy] = np.random.randint(-MAX_DEFORM, MAX_DEFORM)
                    sdy[ix, iy] = np.random.randint(-MAX_DEFORM, MAX_DEFORM)
            
            # initialize spline interpolation
            rect_B_spline_x = interpolate.RectBivariateSpline(sx, sy, sdx)
            rect_B_spline_y = interpolate.RectBivariateSpline(sx, sy, sdy)
            
            # lay meshrid over image
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            
            # interpolate deformations
            dx = rect_B_spline_x.ev(x.flatten(), y.flatten()).reshape((I_cropped.shape[0], I_cropped.shape[1]))
            dy = rect_B_spline_y.ev(x.flatten(), y.flatten()).reshape((I_cropped.shape[0], I_cropped.shape[1]))
            
            # resampling meshgrid
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            
            # resample images
            I_trans = map_coordinates(I_cropped, indices, order=2).reshape(shape)
            M_trans = map_coordinates(M_cropped, indices, order=0).reshape(shape)
            
            # transform image
            I_trans = transform_img(I_trans)
            M_trans = transform_mask(M_trans)
            
            # keep augmented data
            augmented_data.append((patient_id, img_id, I_trans, M_trans))
            
            # show agmented data
            if SHOW_SAMPLES:
                plt.figure("Images %d" % i_var)
                plt.subplot(2, 2, 1)
                plt.imshow(I_orig, cmap=plt.cm.gray, interpolation='nearest')
                plt.subplot(2, 2, 2)
                plt.imshow(M_orig, cmap=plt.cm.gray, interpolation='nearest')
                plt.subplot(2, 2, 3)
                plt.imshow(I_trans, cmap=plt.cm.gray, interpolation='nearest')
                plt.subplot(2, 2, 4)
                plt.imshow(M_trans, cmap=plt.cm.gray, interpolation='nearest')
        
        plt.show(block=True)
    
    stop = time.time()
    print "Time Required:", stop - start
    
    print '\nCreating data array...'
    augmented_data = np.array(augmented_data,
                              dtype=[('patient_id', np.int), ('img_id', np.int),
                                     ('img', np.uint8, (dst_shape[0], dst_shape[1])),
                                     ('mask', np.uint8, (dst_shape[0], dst_shape[1]))])

    np.save(aug_data_file, augmented_data)

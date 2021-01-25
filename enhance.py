#coding:utf-8-
import cv2
import os
import numpy as np
import shutil

def mkdir(root):
    
    if os.path.exists(root) == True:
        shutil.rmtree(root)
    os.mkdir(root)
    
    return

def process(model, img_path, mask_path, save_path, bin_mode=True, dpi=500):
        
    threshold = 128
    imgs = os.listdir(img_path)
    for i in range(len(imgs)):
        img_name = imgs[i][:imgs[i].rfind('.')]
        print(img_name)
        img_file = cv2.imread(img_path + imgs[i], 0)
        old_shape = np.shape(img_file)
        if os.path.exists(mask_path + img_name + '.bmp'): # if segmentation result exists
            mask = cv2.imread(mask_path + img_name + '.bmp', 0)
        else: # for some situation no need of segmentation 
            mask = np.zeros(old_shape, dtype='uint8') + 255
        assert np.shape(mask) == np.shape(img_file)
        if dpi != 500:
            shape = (old_shape[0] * 500 // dpi, old_shape[1] * 500 // dpi)
            img_file = cv2.resize(img_file, (shape[1], shape[0]))
            mask = cv2.resize(mask, (shape[1], shape[0]))
        else:
            shape = old_shape
        row = (828 - shape[0]) // 2
        col = (828 - shape[1]) // 2
        img = np.zeros(np.shape(mask), dtype = 'float32')
        img[:,:] = img_file[:,:]
        img[mask == 0] = -255
        predict_img = np.zeros((1,828,828,1), dtype = 'float32') - 1
        predict_img[0,row:row+shape[0],col:col+shape[1],0] = img / 255.0
        mask_tmp = np.zeros((828,828), dtype = 'float32')
        mask_tmp[row:row+shape[0],col:col+shape[1]] = mask
        
        output = model.predict(predict_img)[0,:,:,0]*255

        output[mask_tmp == 0] = 0
        
        save_file = np.zeros(shape, dtype='uint8')
        save_file[:,:] = output[row:row+shape[0],col:col+shape[1]]
        
        if dpi != 500:
            save_file = cv2.resize(save_file, (old_shape[1], old_shape[0]))
        
        if bin_mode:
            output[output >= threshold] = 255
            output[output < threshold] = 0

        cv2.imwrite(save_path+img_name+'.bmp', save_file)

    print('') 
    
if __name__ == '__main__':
    import json
    from keras.models import model_from_json
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import keras.backend as K
    
    K.set_image_dim_ordering('tf')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    
    KTF.set_session(session)
    
    img_path = './img/'
    mask_path = './mask/'
    save_path = './enhanced/'
    dpi = 500

    with open("./model/enhancement.json",'r',encoding='utf-8') as json_file:
        model = json.load(json_file)
    enh_model = model_from_json(model)
    if os.path.exists("./model/enhancement_nist_weights.h5"):
        enh_model.load_weights("./model/enhancement_nist_weights.h5")
    
    mkdir(save_path)
    
    process(enh_model, img_path, mask_path, save_path, bin_mode=True, dpi=dpi)
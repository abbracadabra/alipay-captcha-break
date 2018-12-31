import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import PIL.Image as Image
import PIL
import PIL.ImageOps
import os
import uuid
import PIL.ImageFilter as ImageFilter
import colorsys
import cv2
import tensorflow as tf

im = cv2.imread(r'EKXJ.png')
im = 255 - np.maximum(im[:,:,1],im[:,:,2])
thresh,im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
#_, contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL  , cv2.CHAIN_APPROX_SIMPLE)

im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
outer = []
inner = []
hierarchy = np.squeeze(np.array(hierarchy))
for i in range(len(hierarchy)):
    if hierarchy[i,3] == -1:#parent is none
        outer.append(contours[i])
        sub = []
        for j in range(len(hierarchy)):
            if hierarchy[j,3] == i:
                sub.append(contours[j])
        inner.append(sub)

x_list=[]
valid_mask=[]
for i,cnt in enumerate(outer):
     [x,y,w,h] = cv2.boundingRect(cnt)
     x_list.append(x)
     if w*h>100:
         valid_mask.append(True)
     else:
         valid_mask.append(False)

order = np.argsort(x_list)
valid_mask = np.array(valid_mask)[order]
order_outer_contours = [outer[ix] for ix in order]
order_outer_contours = [cnt for i,cnt in enumerate(order_outer_contours) if valid_mask[i]==True]
order_sub_contour = [inner[ix] for ix in order]
order_sub_contour = [cnt for i,cnt in enumerate(order_sub_contour) if valid_mask[i]==True]


# dir = os.path.dirname(os.path.realpath(__file__))
# testpath = os.path.join(dir,'singletest')
ims = []
for i in  range(len(order_outer_contours)):
    im = np.zeros((30, 100)).astype(dtype='uint8')
    cv2.drawContours(im, [order_outer_contours[i]] + order_sub_contour[i], -1, (255), 1)
    im = Image.fromarray(np.array(im))
    im = im.crop(im.getbbox())
    im = im.resize((32,32))
    #im.save(r'D:\\Users\\yl_gong\\Desktop\\sss'+str(i)+".jpg",'JPEG');
    ims.append(np.array(im))

    # if not os.path.isdir(testpath):
    #     os.mkdir(testpath)
    # im.save(os.path.join(testpath,str(uuid.uuid4()) + ".jpg"),'JPEG')

saver = tf.train.import_meta_graph(r'singlemodel\captchabreak.ckpt.meta')
# for i in tf.get_default_graph().get_operations():
#     print(i.name)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(r"singlemodel"))
    graph = tf.get_default_graph()
    sds = sess.run([graph.get_tensor_by_name("output:0")],feed_dict={graph.get_tensor_by_name("input:0"):np.expand_dims(np.array(ims)/255.0,axis=-1)})
    for i in sds[0]:
        if i<10:
            print(chr(i+48),end='')
        else:
            print(chr(i-10+65),end='')



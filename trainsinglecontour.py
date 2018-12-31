import numpy as np
import tensorflow as tf
import os
from PIL import Image
import shutil

input = temp= tf.placeholder(dtype='float32', shape=(None,32,32,1), name='input')

temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())#(None,32,32,32)
temp = tf.layers.batch_normalization(temp)
temp=tf.contrib.layers.bias_add(temp,activation_fn=tf.nn.leaky_relu)
temp = tf.layers.dropout(temp,0.5)

temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())#(None,32,32,32)
temp = tf.layers.batch_normalization(temp)
temp=tf.contrib.layers.bias_add(temp,activation_fn=tf.nn.leaky_relu)
temp = tf.layers.dropout(temp,0.5)
temp = tf.layers.max_pooling2d(temp,pool_size=[2, 2], strides=2)
###
temp = tf.layers.conv2d(inputs=temp,filters=128,kernel_size=(3,3),padding="SAME",use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())#(None,32,32,32)
temp = tf.layers.batch_normalization(temp)
temp=tf.contrib.layers.bias_add(temp,activation_fn=tf.nn.leaky_relu)
temp = tf.layers.dropout(temp,0.75)

temp = tf.layers.conv2d(inputs=temp,filters=128,kernel_size=(3,3),padding="SAME",use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())#(None,32,32,32)
temp = tf.layers.batch_normalization(temp)
temp=tf.contrib.layers.bias_add(temp,activation_fn=tf.nn.leaky_relu)
temp = tf.layers.dropout(temp,0.75)
temp = tf.layers.max_pooling2d(temp,pool_size=[2, 2], strides=2)#(None,16,16,32)

temp = tf.layers.conv2d(inputs=temp,filters=256,kernel_size=(3,3),padding="SAME",use_bias=False,kernel_initializer=tf.keras.initializers.he_normal())#(None,32,32,32)
temp = tf.layers.batch_normalization(temp)
temp=tf.contrib.layers.bias_add(temp,activation_fn=tf.nn.leaky_relu)
temp = tf.layers.dropout(temp,0.75)
temp = tf.layers.max_pooling2d(temp,pool_size=[2, 2], strides=2)

temp = tf.layers.flatten(temp)
temp = tf.layers.dense(temp,36,kernel_initializer=tf.keras.initializers.he_normal())

output = tf.argmax(temp,axis=-1,name='output')

#loss
label_input = tf.placeholder(dtype='float32',shape=(None,36))
label = tf.reshape(label_input,(-1,36))
temp = tf.reshape(temp,(-1,36))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=temp,dim=-1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

#acc
a1 = tf.argmax(temp,axis=-1)
a2 = tf.argmax(label,axis=-1)
eq = tf.equal(a1,a2)

acc = tf.cast(tf.reduce_sum(tf.cast(eq,dtype='int8')),dtype='float32')/tf.cast(tf.shape(input)[0],dtype='float32')

dir = os.path.dirname(os.path.realpath(__file__))
trainpath = os.path.join(dir,'singlesss')
testpath = os.path.join(dir,'singletest')

def train(epochs):
    saver = tf.train.Saver()
    lossrec = []
    accrec = []
    bestacc = 0
    with tf.Session() as sess:
        saver.restore(sess, r'singlemodel/captchabreak.ckpt')
        valimg, vallabel = next(validategenerator(testpath))
        for i in range(epochs):
            for j, (trainimg, trainlabel) in enumerate(traingenerator(trainpath, 20)):
                _, trainacc, trainloss = sess.run([optimizer, acc, loss],feed_dict={input: trainimg, label_input: trainlabel});
                print("epoch:{} batch:{} trainloss:{:.4f} trainacc:{:.2f} ".format(i, j, trainloss,trainacc))
                if j%10==0:
                    valacc, valloss = sess.run([acc, loss],feed_dict={input: np.array(valimg), label_input: np.array(vallabel)})
                    print("validateloss:{:.4f} validateacc:{:.2f} ".format(valloss, valacc))
                    print(bestacc)
                    if valacc>bestacc:
                        bestacc = valacc
                        shutil.rmtree("singlemodel")
                        saver.save(sess, "singlemodel/captchabreak.ckpt")
            # keep some logs
            lossrec.append(valloss)
            accrec.append(valacc)
            print(lossrec)
            print(accrec)


def validategenerator(path):
    fs = os.listdir(path)
    yield fetch(fs,path)

def traingenerator(path,batch_size):
    fs = os.listdir(path);
    fs=np.random.permutation(fs)
    if batch_size == 0:
        batch_size = len(fs)
    offset=0
    while offset<len(fs):
        yield fetch(fs[offset:offset+batch_size],path)
        offset+=batch_size

def fetch(fs,path):
    imgs = []
    labels = []
    for i, fname in enumerate(fs):
        fp = os.path.join(path, fname)
        try:
            imp = Image.open(fp).resize((32, 32));
        except:
            continue
        imp = imp.convert('L')
        #imp = imp.point(lambda p: (p>240)*255)
        im = np.array(imp)/255.
        im = np.expand_dims(im, axis=-1)
        imp.close()

        c = fname[0].upper()
        lb = np.zeros((36))
        if ord(c) >= 48 and ord(c) <= 57:
            lb[ord(c) - 48] = 1
        elif ord(c) >= 65 and ord(c) <= 90:
            lb[ord(c) - 65 + 10] = 1
        imgs.append(im)
        labels.append(lb)
    return np.array(imgs), np.array(labels)

train(30)











        # saver.restore(sess, r'singlemodel/captchabreak.ckpt')
        # builder = tf.saved_model.builder.SavedModelBuilder("alipay/buildmodel")
        # builder.add_meta_graph_and_variables(
        #     sess,
        #     [tf.saved_model.tag_constants.SERVING]
        # )
        # builder.save()
        # return;
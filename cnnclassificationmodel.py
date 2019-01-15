import tensorflow as tf
from datautil import build_all_dataset,generate_num,batch_yield,gettestdata,getdevdata
import matplotlib.pyplot as plt
import numpy as np

MAX_LEN=1000
MAX_EPISODES=2001
BATCH_SIZE=50
INPUT_SIZE=20

value_dict={"G":0,"A":1,"V":2,"L":3,"I":4,"P":5,"F":6,"Y":7,"W":8,"S":9,"T":10,"C":11,
           "M":12,"N":13,"Q":14,"D":15,"E":16,"K":17,"R":18,"H":19}

def Seq2Mat(seq,max_len):
    nums=[]
    usable=True
    for alpha in seq[:-1]:
        num = [0] * INPUT_SIZE
        if alpha in value_dict.keys():
            num[value_dict[alpha]] = 1
        else:
            usable=False
            break
        nums.append(num)
    if usable:
        if len(nums)<max_len:
            nums.extend([[0]*INPUT_SIZE]*(max_len-len(nums)))#pad with zero matrix
        elif len(nums)>max_len:
            nums=nums[:max_len]
    return nums,usable

class CNNnet:
    def __init__(self,input_len,begin,input_size):
        self.input_len=input_len
        self.sess=tf.Session()
        self.input_size=input_size
        self.buildnet()
        if begin:
            self.init()
        else:
            self.restore()

    def buildnet(self):
        self.tf_x=tf.placeholder(tf.float32,[None,self.input_len,self.input_size])
        conv1=tf.layers.conv1d(inputs=self.tf_x,filters=16,kernel_size=4,strides=1,padding='same',activation=tf.nn.relu)
        pool1=tf.layers.max_pooling1d(conv1,pool_size=2,strides=2)
        conv2=tf.layers.conv1d(pool1,32,8,1,'same',activation=tf.nn.relu)
        pool2=tf.layers.max_pooling1d(conv2,5,5)#
        conv3 = tf.layers.conv1d(pool2, 64, 12, 1, 'same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(conv3, 3, 3)  #
        shape=pool3.shape
        flat=tf.reshape(pool3,[-1,int(shape[-1]*shape[-2])])#?
        FClayer=tf.layers.dense(flat,20)
        self.output=tf.layers.dense(FClayer,6)
        self.tf_y = tf.placeholder(tf.int32, [None, ])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tf_y, logits=self.output))  # ?
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def predict(self,inputseq):
        out=self.sess.run(tf.nn.softmax(self.output),feed_dict={self.tf_x:inputseq})
        return out

    def init(self):
        init_op = tf.group(tf.global_variables_initializer())
        self.sess.run(init_op)

    def trainnet(self):
        print("trainbegin")
        inputs,labels=build_all_dataset()
        self.losses=[]
        self.accuracies=[]
        for step in range(MAX_EPISODES):
            nums=generate_num(BATCH_SIZE,len(inputs)-1)
            batch_inputseqs,batch_labels=batch_yield(nums,inputs,labels)
            batch_inputs=[]
            batch_labels_=[]
            Seq2Mat(batch_inputseqs[49], MAX_LEN)
            for i in range(len(batch_inputseqs)):
                seq_input,seq_usable=Seq2Mat(batch_inputseqs[i],MAX_LEN)
                if seq_usable:
                    batch_inputs.append(seq_input)
                    batch_labels_.append(batch_labels[i])
            batch_inputs=np.array(batch_inputs)
            batch_labels_=np.array(batch_labels_)
            loss,_=self.sess.run([self.loss,self.optimizer], feed_dict={self.tf_x: batch_inputs, self.tf_y: batch_labels_})
            if step%50==0:
                test_inputseqs,test_labels=gettestdata()
                test_inputs=[]
                test_labels_=[]
                for i in range(len(test_inputseqs)):
                    seq_input,seq_usable=Seq2Mat(test_inputseqs[i],MAX_LEN)
                    if seq_usable:
                        test_inputs.append(seq_input)
                        test_labels_.append(test_labels[i])
                test_outputs=self.sess.run(tf.argmax(self.output,axis=1),feed_dict={self.tf_x:test_inputs})
                error_num=0
                for i in range(len(test_outputs)):
                    if(test_outputs[i]!=test_labels_[i]):
                        error_num+=1
                accuracy=(len(test_outputs)-error_num)/len(test_outputs)
                self.accuracies.append(accuracy)
                self.losses.append(loss)
                print("test: accuracy=",accuracy)
                self.save()

    def plot(self):
        x1=np.arange(1,len(self.losses)+1)
        y1=self.losses
        x2 = np.arange(1, len(self.accuracies) + 1)
        y2 = self.accuracies
        for i in range(len(x1)):
            x1[i]=x1[i]*50
            x2[i] = x2[i] * 50
        plt.xlabel("times")
        plt.ylabel("loss")
        plt.plot(x1, y1, marker='o')
        plt.show()
        plt.xlabel("times")
        plt.ylabel("accuracy")
        plt.plot(x2,y2,marker='o')
        plt.show()

    def save(self):
        saver=tf.train.Saver()
        saver.save(self.sess,r'./model/model.ckpt')

    def restore(self):
        saver=tf.train.Saver()
        saver.restore(self.sess,r'./model/model.ckpt')

    def dev(self):
        dev_inputseqs, dev_labels = getdevdata()
        dev_inputs = []
        dev_labels_=[]
        for i in range(len(dev_inputseqs)):
            seqinput,seq_usable=Seq2Mat(dev_inputseqs[i],MAX_LEN)
            if seq_usable:
                dev_inputs.append(seqinput)
                dev_labels_.append(dev_labels[i])
        dev_outputs = self.sess.run(tf.argmax(self.output, axis=1), feed_dict={self.tf_x: dev_inputs})
        error_num = 0
        for i in range(len(dev_inputs)):
            if (dev_outputs[i] != dev_labels_[i]):
                error_num += 1
        accuracy = (len(dev_inputs) - error_num) / len(dev_inputs)
        print("dev: accuracy=", accuracy)

def main():
    net=CNNnet(MAX_LEN,True,INPUT_SIZE)
    net.trainnet()
    net.dev()
    net.plot()

if __name__=="__main__":
    main()
import tensorflow as tf
from datautil import build_all_dataset,generate_num,batch_yield,gettestdata,getdevdata
import matplotlib.pyplot as plt
import numpy as np

MAX_LEN=1000
MAX_EPISODES=2001
BATCH_SIZE=50
INPUT_SIZE=21

valuedict={"G":0,"A":1,"V":2,"L":3,"I":4,"P":5,"F":6,"Y":7,"W":8,"S":9,"T":10,"C":11,
           "M":12,"N":13,"Q":14,"D":15,"E":16,"K":17,"R":18,"H":19}

def Getmat(alpha):
    num=[0]*21
    if alpha in valuedict.keys():
        num[valuedict[alpha]]=1
    else:
        num[20]=1
    return num

def Getseqnum2(seq,maxlen):
    nums=[]
    for alpha in seq:
        nums.append(Getmat(alpha))
    if len(nums)<maxlen:
        nums.extend([[0]*21]*(maxlen-len(nums)))#pad with zero matrix
    elif len(nums)>maxlen:
        nums=nums[:maxlen]
    return nums

class CNNnet:
    def __init__(self,inputlen,begin,inputsize):
        self.inputlen=inputlen
        self.sess=tf.Session()
        self.inputsize=inputsize
        self.buildnet()
        if begin:
            self.init()
        else:
            self.restore()

    def buildnet(self):
        self.tf_x=tf.placeholder(tf.float32,[None,self.inputlen,self.inputsize])
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
            print(step)
            nums=generate_num(BATCH_SIZE,len(inputs)-1)
            batch_inputseqs,batch_labels=batch_yield(nums,inputs,labels)
            batch_inputs=[]
            for seq in batch_inputseqs:
                batch_inputs.append(Getseqnum2(seq, MAX_LEN))
            loss,_=self.sess.run([self.loss,self.optimizer], feed_dict={self.tf_x: batch_inputs, self.tf_y: batch_labels})
            if step%50==0:
                testinputseqs,testlabels=gettestdata()
                testinputs=[]
                for seq in testinputseqs:
                    testinputs.append(Getseqnum2(seq, MAX_LEN))
                testoutputs=self.sess.run(tf.argmax(self.output,dimension=1),feed_dict={self.tf_x:testinputs})
                wrong_num=0
                for i in range(len(testinputs)):
                    if(testoutputs[i]!=testlabels[i]):
                        wrong_num+=1
                accuracy=(len(testinputs)-wrong_num)/len(testinputs)
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
        devinputseqs, devlabels = getdevdata()
        devinputs = []
        for seq in devinputseqs:
            devinputs.append(Getseqnum2(seq, MAX_LEN))
        testoutputs = self.sess.run(tf.argmax(self.output, dimension=1), feed_dict={self.tf_x: devinputs})
        wrong_num = 0
        for i in range(len(devinputs)):
            if (testoutputs[i] != devlabels[i]):
                wrong_num += 1
        accuracy = (len(devinputs) - wrong_num) / len(devinputs)
        print("dev: accuracy=", accuracy)

def main():
    net=CNNnet(MAX_LEN,True,INPUT_SIZE)
    net.trainnet()
    net.dev()
    net.plot()

if __name__=="__main__":
    main()
from Bio import SeqIO
import random

BATCH_SIZE=50

paths=['PVA','PU','PHB','PHA','Phthalate','Others']

#This part of code is used to divide raw data into different data sets.
def process_all_data():
    for path in paths:
        data=read_original_data(path)
        write_dataset(data,path)

def read_original_data(path):
    data=[]
    for seq_record in SeqIO.parse(r'data/original/'+path+'.fasta',"fasta"):
        data.append(seq_record.seq)
    return data

def write_dataset(data,path):
    trainfile=open(r"data/train/"+path,"w")
    for i in range(int(len(data)*0.8)):
        trainfile.write(str(data[i]))
        trainfile.write('\n')
    trainfile.close()
    testfile=open(r"data/test/"+path,"w")
    for i in range(int(len(data)*0.8),int(len(data)*0.9)):
        testfile.write(str(data[i]))
        testfile.write('\n')
    testfile.close()
    devfile=open(r"data/dev/"+path,"w")
    for i in range(int(len(data)*0.9),len(data)):
        devfile.write(str(data[i]))
        devfile.write('\n')
    devfile.close()

#This part of code is used to read data from datasets
def build_all_dataset():
    traininputs=[]
    trainlabels=[]
    k=0
    for i in range(5):
        build_dataset(paths[i], r'data/train/', i, traininputs, trainlabels)
    return traininputs,trainlabels

def build_dataset(path,basepath,label,inputs,labels):
    with open(basepath+path,'r') as f:
        line=f.readline()
        while line:
            inputs.append(line)
            labels.append(label)
            line=f.readline()

def generate_num(batch_size,range):
    nums=[]
    while len(nums)<batch_size:
        num=random.randint(0,range)
        if num not in nums:
            nums.append(num)
    return nums

def batch_yield(nums,inputs,labels):
    batch_inputs=[]
    batch_labels=[]
    for num in nums:
        batch_inputs.append(inputs[num])
        batch_labels.append(labels[num])
    return batch_inputs,batch_labels

def gettestdata():
    testinputs = []
    testlabels = []
    for i in range(5):
        build_dataset(paths[i], r'data/test/', i, testinputs, testlabels)
    return testinputs, testlabels

def getdevdata():
    devinputs = []
    devlabels = []
    for i in range(5):
        build_dataset(paths[i], r'data/dev/', i, devinputs, devlabels)
    return devinputs, devlabels

#def main():
#    process_all_data()

#if __name__=="__main__":
#    main()
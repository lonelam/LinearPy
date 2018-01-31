'''
 @Time    : 2017/12/19 下午12:47
 @Author  : Eric Wang
 @File    : naivebayes.py
 @license : Copyright(C), Eric Wang
 @Contact : eric.w385@gmail.com
'''
import numpy as np
from math import log,exp,pow
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import defaultdict,Counter

def read_data(model = None):
    dataset = []
    boy = open('data/boy.txt')
    girl = open('data/girl.txt')
    labels = []
    test_set = []
    boy_date = []
    girl_date = []
    boy_labels = []
    girl_labels = []
    for item in boy.readlines():
        item = item.strip().split()
        boy_labels.append(1)
        boy_date.append(item[:model])
    for item in girl.readlines():
        item = item.strip().split()
        girl_labels.append(0)
        girl_date.append(item[:model])
    dataset = boy_date[:int(len(boy_date)*0.8)] + girl_date[:int(len(girl_date)*0.8)]
    labels = boy_labels[:int(len(boy_date)*0.8)] + girl_labels[:int(len(girl_date)*0.8)]
    test_set = boy_date[int(len(boy_date)*0.8):] + girl_date[int(len(girl_date)*0.8):]
    test_label = boy_labels[int(len(boy_date)*0.8):] + girl_labels[int(len(girl_date)*0.8):]
    return dataset, labels, test_set, test_label

def train(dataset, labels, lambd=1):
    '''label:样本标签
    feat_dic_list:身高体重鞋码:出现次数
    feat_value:特征取值
    '''
    feat_num = len(dataset[0]) 
    label_count = Counter(labels)
    feat_value = [set() for i in range(feat_num)] 
    model = {label:[defaultdict(int) for i in range(feat_num)] for label in set(labels)}
    for example,label in zip(dataset,labels):
        for i in range(feat_num):
            feat_value[i].add(example[i]) 
            feat_dic = model[label][i]
            feat_dic[example[i]] += 1 
    feat_value_num = [len(value) for value in feat_value] # 每个特征可能的取值数目
    for label,feat_dic_list in model.items():
        for i in range(feat_num):
            for value in feat_value[i]: # 贝叶斯估计
                if value in feat_dic_list[i]:
                    feat_dic_list[i][value] = log((feat_dic_list[i][value]+lambd)/(label_count[label]+feat_value_num[i]*lambd))
                else:
                    feat_dic_list[i][value] = log(1/(label_count[label]+feat_value_num[i]*lambd))
    label_log_prob = label_count
    for label,value in label_log_prob.items(): 
        label_log_prob[label] = (value)/(len(labels))
    return model, label_log_prob

def classfication(dataset, model, label_log_prob):
    feat_num = len(dataset)
    class_log_prob = {}
    for label,feat_dic_list in model.items():
        log_prob = 0.0
        for i in range(feat_num):
            log_prob += feat_dic_list[i][dataset[i]]
        log_prob += label_log_prob[label]
        class_log_prob[label] = log_prob
        best_class, max_log_prob = Counter(class_log_prob).most_common(1)[0]
    return exp(class_log_prob[1]),exp(class_log_prob[0])

def plot_prob(dataset):
    x = np.arange(145,190,0.1)
    y = np.linspace(0.00,1.00,450)
    dataset = list(map(lambda x: x[0],dataset))
    mu_boy = np.array(dataset[:181]).astype(int).mean()
    mu_girl = np.array(dataset[181:]).astype(int).mean()
    sigma_boy = np.array(dataset[:181]).astype(int).std()
    sigma_girl = np.array(dataset[181:]).astype(int).std()
    pdf_boy = np.exp(-((x - mu_boy)**2)/(2*sigma_boy**2)) / (sigma_boy * np.sqrt(2*np.pi))
    pdf_girl = np.exp(-((x - mu_girl)**2)/(2*sigma_girl**2)) / (sigma_girl * np.sqrt(2*np.pi))
    plt.plot(x,pdf_boy)
    plt.plot(x,pdf_girl)
    
    plt.hist(np.array(dataset[:181]).astype(int), bins=10, rwidth=0.9, normed=True)
    plt.hist(np.array(dataset[181:]).astype(int), bins=10, rwidth=0.9, normed=True)
    plt.show()
    
def plot_roc_crove(test_set, test_label, model, label_log_prob):
    total_girl = len(list(filter(lambda x:x==0 ,test_label)))
    total_boy = len(list(filter(lambda x:x==1 ,test_label)))
    predict = []
    TP_rate = []
    FP_rate = []
    for date in test_set:
            boy_rate,girl_rate = classfication(date,model,label_log_prob)
            if boy_rate>=girl_rate:
                predict.append(1)
            else:
                predict.append(0)
    fpr, tpr, threshold = roc_curve(test_label, predict)
    roc_auc = auc(fpr, tpr)*100
    for t in np.linspace(-1,1,1000):
        TP = 0
        FP = 0
        for date,label in zip(test_set,test_label):
            boy_rate,girl_rate = classfication(date,model,label_log_prob)
            #print(boy_rate)
            if boy_rate+t>girl_rate:
                #print(label)
                if label==1:
                    TP += 1
                else:
                    FP += 1
        TP_rate.append(TP/total_boy)
        FP_rate.append(FP/total_girl)
    plt.plot(FP_rate,TP_rate)
    print('accuracy for program 2: %0.2f%%' %roc_auc)
  
def parzen_window(h,dataset):
    num=len(dataset)
    result = []
    for i in dataset:
        sum=0.0
        for j in dataset:
            if(abs(int(i[0])-int(j[0]))<=h/2 ):     
                sum+=1
        result.append(sum/(h*num))
    return result

def plot_parzen(dataset,h):
    dataset.sort()                                                                             
    m_Data=parzen_window(h,dataset)
    lines=plt.plot(dataset,m_Data,'x',linestyle="-")
    plt.setp(lines, color='r', linewidth=2.0)
    plt.grid(True)
    plt.show() 
    
def program1():
    dataset, labels, test_set, test_label = read_data(1) 
    model,label_log_prob = train(dataset, labels)
    predict = []
    for date in test_set:
            boy_rate,girl_rate = classfication(date,model,label_log_prob)
            if boy_rate>=girl_rate:
                predict.append(1)
            else:
                predict.append(0)
    fpr, tpr, threshold = roc_curve(test_label, predict)
    roc_auc = auc(fpr, tpr)*100
    print('accuracy for program 1: %0.2f%%' %roc_auc)
    plot_prob(dataset)
    plt.show()
    
def program2():
    dataset, labels, test_set, test_label = read_data(2) 
    model,label_log_prob = train(dataset, labels)
    plot_roc_crove(test_set, test_label, model, label_log_prob)
    plt.show()
    #print(list(map(lambda x: x[1], dataset[:181])),list(map(lambda x: x[0],dataset[:181])))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(list(map(lambda x: x[1],dataset[:181])),
                list(map(lambda x: x[0],dataset[:181])),
                marker='.')
    ax.scatter(list(map(lambda x: x[1],dataset[181:])),
                list(map(lambda x: x[0],dataset[181:])),
                marker='x')
    plt.show()
    print(list(map(lambda x: x[0],dataset[181:])))
def program3():
    dataset, labels, test_set, test_label = read_data(1) 
    model,label_log_prob = train(dataset, labels)
    predict = []
    for date in test_set:
            boy_rate,girl_rate = classfication(date,model,label_log_prob)
            if boy_rate>=girl_rate:
                predict.append(1)
            else:
                predict.append(0)
    fpr, tpr, threshold = roc_curve(test_label, predict)
    roc_auc = auc(fpr, tpr)*100
    print('accuracy for program 3: %0.2f%%' %roc_auc)
    plot_parzen(dataset,20)
    
if __name__ == "__main__":
    '''model = 2
    dataset, labels, test_set, test_label = read_data() 
    model,label_log_prob = train(dataset, labels)
    plot_prob(dataset)
    plt.show()
    plot_roc_crove(test_set, test_label, model, label_log_prob)'''
    program1()
    program2()
    # program3()

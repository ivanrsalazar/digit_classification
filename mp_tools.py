import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations
import math



def import_data():
    '''
    imported and converted mnist data into numpy arrays
    these array consist of two test sets and training sets
    trainX has shape 60,000 x 784
    trainY has shape 60,000 x 1
    testX has shape 10,000 x 784
    testY has shape 10,000 x 1

    returns a tuple (testX,testY,trainX, trainY)
    with the normalized and formatted data
    '''
    try:
        mat = sio.loadmat('mnist.mat')
    except:
        print("please place mnist.mat into the same directory as this file...thanks")
    testX = mat['testX']/255            # normalizing
    testY = mat['testY']

    trainX = mat['trainX']/255          # normalizing
    trainY = mat['trainY']

    train_ones = np.ones((trainX.shape[0],1))                               
    test_ones = np.ones((testX.shape[0],1))
        
    trainX = np.hstack((trainX,train_ones))                       
    testX = np.hstack((testX,test_ones))

    return testX, testY, trainX, trainY


def set_true(labels, k):
    '''
    input: numpy column array (nx1), int
    output: numpy nx1 array
    function: sets all the elements equal to k to true (1) and all else to false (-1)
    '''
    assert(isinstance(labels, np.ndarray))
    assert(isinstance(k,int))
    labels = labels.astype(int)
    
    for i,el in enumerate(labels[0]):              
        labels[0][i] = 1 if (el == k) else -1
    
    return labels


def filter_data(data_set,labels, k, j):
    '''
    input: numpy column array (nx1), int, int
    output: numpy nx1 array
    function: sets all the elements equal to k to true (1) and all else to false (-1)
    '''
    assert(isinstance(data_set, np.ndarray))
    assert(isinstance(labels, np.ndarray))
    assert(isinstance(k,int))
    assert(isinstance(j,int))

    idk = np.where(labels == k)[1]                             # returns the indices of images which have the label of k
    idj = np.where(labels == j)[1]                             # returns the indices of images which have the label of j
    k_arr = np.zeros((idk.shape[0],data_set.shape[1]))
    j_arr = np.zeros((idj.shape[0],data_set.shape[1]))

    for i, el in enumerate(idk):
        k_arr[i] = data_set[el]                                 # grab the images with label k and store into the k_arr
    for i, el in enumerate(idj):                                
        j_arr[i] = data_set[el]                                 # grab the images with label j and store into the j_arr

    k_labels = np.ones((1,len(idk)))                            # creating +1 labels for images with k label
    j_labels = np.ones((1,len(idj))) * -1                       # creating -1 labels for images with j label

    return np.vstack((k_arr,j_arr)), np.hstack((k_labels, j_labels))
    
    


def reshape_data():
    '''
    returns void, does work on dataset so it can be displayed
    '''
    testX = testX.reshape((10000, 28, 28))
    trainX = trainX.reshape((60000,28,28))

    return testX


# iterating thru each image found in trainX and plotting
# used this to verify correct reshaping

def show_img(data,show_one=0):
    '''
    input: int (optional), set to 1 for true
    returns void, plots all images
    if the user only wants to show one, set show_one to 1 
    '''
    trainX = data[0]
    for img in trainX:
        plt.imshow(img)
        plt.show()
        if show_one == 1: break
    

def dump_lines(data,file_name='dump_lines.txt'):
    '''
    This function dumps matrices into file
    function arg file_name is set to "dump_lines.txt" by default
    '''
    
    outF = open(file_name,'w')
    for y in data:
        outF.write(str(y))
        outF.write('\n')
    outF.close()
    print("finished dumping lines") 

def train_OVA(train_data, labels, k):
    '''
    input: numpy 2d array, column numpy array, int 
    output: column numpy array
    '''
    assert(isinstance(train_data,np.ndarray))
    assert(isinstance(labels,np.ndarray))
    assert(isinstance(k,int))

    theta = np.zeros((785,1))                                               
    true_set = set_true(labels,k)
    theta = np.matmul(np.linalg.pinv(train_data),true_set.transpose())
    
    return theta, train_data, true_set

def train_OVO(train_data, labels, i, j):
    '''
    input: numpy 2d array, column numpy array, int, int 
    output: column numpy array
    '''
    assert(isinstance(train_data,np.ndarray))
    assert(isinstance(labels,np.ndarray))
    assert(isinstance(i,int))
    assert(isinstance(j,int))

    filtered_data, filtered_labels = filter_data(train_data,labels,i,j)
    return np.matmul(np.linalg.pinv(filtered_data), filtered_labels.transpose()),filtered_data, filtered_labels
  

def test(theta, test_data, true_labels):
    '''
    input: column numpy array, 2d numpy arry, column numpy array
    return is void
    '''
    assert(isinstance(theta,np.ndarray))
    assert(isinstance(test_data,np.ndarray))
    assert(isinstance(true_labels,np.ndarray))
    pred_temp = np.matmul(test_data, theta)
    pred = np.sign(pred_temp)
    e_count = 0
    for i,e in enumerate(pred):
        if true_labels.transpose()[i] != e:
            e_count = e_count + 1
    #print("errors: " +str(e_count))
    #print("error%: " + str("{:.4f}".format(e_count/test_data.shape[0])) + '%')
    #print('\n')



def get_ova_weights(train_data,train_labels, test_data, test_labels):
    """
    input: 2D numpy array, 1D numpy array, 2D numpy array, 1D numpy array
    output: 1D numpy array
    function: takes in data and labels and returns the trained models/weights for all ten classifiers
    """
    assert(isinstance(train_data, np.ndarray))
    assert(isinstance(train_labels,np.ndarray))
    assert(isinstance(test_data, np.ndarray))
    assert(isinstance(test_labels, np.ndarray))
    lst = list(range(10))
    ova_weights= []
    for i in lst:
        theta, ftrain_data, ftrain_labels = train_OVA(train_data,train_labels, i)
        #print("testing %d vs all classifier (training data)" % (i))
        test(theta, ftrain_data, ftrain_labels)
        true_labels = set_true(test_labels,i)
        #print("testing %d vs all classifier (testing data)" % (i))
        test(theta, test_data, true_labels)
        ova_weights.append(theta)
    return ova_weights



def get_ovo_weights(train_data,train_labels, test_data, test_labels):
    """
    input: 2D numpy array, 1D numpy array, 2D numpy array, 1D numpy array
    output: 1D numpy array
    function: takes in data and labels and returns the trained models/weights for all 45 classifiers
    """
    assert(isinstance(train_data, np.ndarray))
    assert(isinstance(train_labels,np.ndarray))
    assert(isinstance(test_data, np.ndarray))
    assert(isinstance(test_labels, np.ndarray))
    lst = list(range(10))
    ovo_weights = []
    combs = list(combinations(lst,2))
    for i in range(45):
        i, j = combs[i][0],combs[i][1]
        theta, ftrain_data, ftrain_labels = train_OVO(train_data,train_labels, i , j)
        #print("testing %d vs %d classifier (training data)" % (i, j))
        test(theta, ftrain_data, ftrain_labels)
        ftest_data, ftest_labels = filter_data(test_data,test_labels,i,j)
        #print("testing %d vs %d classifier (testing data)" % (i, j))
        test(theta,ftest_data,ftest_labels)
        ovo_weights.append(theta)
    return ovo_weights

def get_ova_bin_preds(theta_arr, test_data):
    '''
    input: numpy array, numpy array
    output: numpy array
    function: returns the predictions of each of the 10 binary classifiers 
    '''
    assert(isinstance(theta_arr,list))
    assert(isinstance(test_data,np.ndarray))
    bin_preds = []
    for i, theta in enumerate(theta_arr):
        pred_temp = np.matmul(test_data, theta)
        bin_preds.append(pred_temp)
    return bin_preds

def get_ovo_bin_preds(theta_arr, test_data):
    '''
    input: numpy array, numpy array
    output: numpy array
    function: returns the predictions of each of the 45 binary classifiers 
    '''
    assert(isinstance(theta_arr,list))
    assert(isinstance(test_data,np.ndarray))
    bin_preds = []
    for i, theta in enumerate(theta_arr):
        pred_temp = np.matmul(test_data, theta)
        pred = np.sign(pred_temp)
        bin_preds.append(pred)
    return bin_preds


def OVA_classifier(ova_preds, test_labels, train=False):
    '''
    input: list consisting of 10 different trained binary classifier weights, test labels
    type: list[10], np.ndarray[10,000]
    output: One vs. One classifier output labels, One vs. One confusion matrix
    '''
    assert(isinstance(ova_preds,list))
    assert(isinstance(test_labels,np.ndarray))
    assert(len(ova_preds) == 10)
    ova_labels = []
    ova_confusion_matrix = np.zeros((10,10), dtype = int)
    N = test_labels.shape[1]
    e_count = 0
    for i in range(N):
        max_pred = -1.5
        pred_label = None
        for j, pred in enumerate(ova_preds):
            if(pred[i] > max_pred):
                max_pred = pred[i]
                pred_label = j
        act_label = test_labels.transpose()[i]
        if act_label != pred_label:
            e_count = e_count + 1
            #print("pred: %d , actual: %d, errors: %d" % (pred_label, act_label, e_count))
        #else:
            #print("pred: %d , actual: %d" % (pred_label, act_label))
        ova_confusion_matrix[pred_label,act_label] = ova_confusion_matrix[pred_label,act_label] + 1
        ova_labels.append(pred_label)
    if(train==False):
        print("TESTING DATA RESULTS")
    else: print("TRAINING DATA RESULTS") 
    print("One vs. All Classifier error count:")
    print(e_count)
    print("One vs. All error percentage:")
    print("%f"% (e_count*100/N) + '%')
    print("printing out One vs All Confusion matrix")
    print(ova_confusion_matrix)
    print('\n')
    return e_count*100/N
    #plt.imshow(ova_confusion_matrix)
    #plt.show()
    

def OVO_classifier(ovo_preds, test_labels, train=False):
    '''
    input: list consisting of 45 different trained binary classifier weights, test labels
    i-type: list, np.ndarray
    '''
    assert(isinstance(ovo_preds, list))
    assert(isinstance(test_labels, (np.ndarray, np.generic) ))
    assert(len(ovo_preds) == 45)
    ovo_labels = []
    ovo_votes = [0] * 10
    ovo_confusion_matrix = np.zeros((10,10), dtype = int)
    lst = list(range(10))
    combs = list(combinations(lst,2))                                       # returns the combinations needed for the 45 binary classifiers, ((0,1),(0,2),...,(1,2),(1,3),...,(2,3),...,(8,9))
    
    e_count = 0
    N = test_labels.shape[1]
    for i in range(N):                                                      # looping over the number of images (for each image do this work)
        for j,pred in enumerate(ovo_preds):                                 # looping over each ovo binary classifier, see what it votes based on the image
            if(pred[i] == True):
                ovo_votes[combs[j][0]] = ovo_votes[combs[j][0]] + 1         
            else:
                ovo_votes[combs[j][1]] = ovo_votes[combs[j][1]] + 1         

        max_votes = 0
        plabel = None
        for k, votes in enumerate(ovo_votes):                               # for each digit, find the one with the most votes
            if(votes > max_votes):
                max_votes = votes
                plabel = k                                                  # set the OVO prediction for the image here
        ovo_votes = [0] * 10

        act_label = test_labels.transpose()[i]                                          # get the real label for the image
        if act_label != plabel:                                                         # see if the OVO classifier got the label right for this image
            e_count = e_count + 1
            #print("pred: %d , actual: %d, errors: %d" % (plabel, act_label, e_count))   # logs
        #else:print("pred: %d , actual: %d" % (plabel, act_label))                       # logs
            
        ovo_labels.append(plabel)
        ovo_confusion_matrix[plabel,act_label] = ovo_confusion_matrix[plabel,act_label] + 1
    if(train==False):
        print("TESTING DATA RESULTS")
    else: print("TRAINING DATA RESULTS")
    print("One vs. One error count:")
    print(e_count)
    print("One vs. One error percentenge:")                                                                              
    print("%f" % (e_count*100/N) + '%')
    print("One vs One Confusion matrix:")
    print(ovo_confusion_matrix)
    print('\n')
    #plt.imshow(ovo_confusion_matrix)
    #plt.show()

    return (e_count*100)/N

    

def randomize_data(data_set, w, b):
    L = w.shape[0]
    new_input = np.zeros((L,1))
    temp_h = np.zeros((data_set.shape[0],L), float)
    for j,x in enumerate(data_set):
        for i in range(L):
            new_input[i] = np.matmul(w[i].transpose(),x) + b[i]
        temp_h[j] = new_input.transpose()

    return temp_h

def sig_data(data_set, w, b):
    L = w.shape[0]
    new_input = np.zeros((L,1))
    temp_h = np.zeros((data_set.shape[0],L), float)
    for j,x in enumerate(data_set):
        for i in range(L):
            temp = np.matmul(w[i].transpose(),x) + b[i]
            new_input[i] = 1 / (1+ (math.exp(-temp)))
        temp_h[j] = new_input.transpose()

    return temp_h

def sin_data(data_set, w, b):
    L = w.shape[0]
    new_input = np.zeros((L,1))
    temp_h = np.zeros((data_set.shape[0],L), float)
    for j,x in enumerate(data_set):
        for i in range(L):
            temp = np.matmul(w[i].transpose(),x) + b[i]
            new_input[i] = math.sin(temp)
        temp_h[j] = new_input.transpose()

    return temp_h

def max_data(data_set, w, b):                                           # when L = 2000
    L = w.shape[0]                                                      # OVO e% = 2.54% | OVA e% = 4.22%
    new_input = np.zeros((L,1))
    temp_h = np.zeros((data_set.shape[0],L), float)
    for j,x in enumerate(data_set):
        for i in range(L):
            temp = np.matmul(w[i].transpose(),x) + b[i]
            new_input[i] = max(temp,0)
        temp_h[j] = new_input.transpose()

    return temp_h

def run_all(trainX,trainY,testX,testY,L):
    w = np.random.normal(size=(L,785))
    b = np.random.normal(size=(L,1))

    id_train = randomize_data(trainX,w,b)
    id_test = randomize_data(testX,w,b)
    id_res = []
    print("g(x) = x")
    ovo_weights = get_ovo_weights(id_train,trainY,id_test, testY)
    trovo_bin_preds = get_ovo_bin_preds(ovo_weights,id_train)
    id_res.append(OVO_classifier(trovo_bin_preds,trainY,train=True))
    ovo_bin_preds = get_ovo_bin_preds(ovo_weights,id_test)
    id_res.append(OVO_classifier(ovo_bin_preds, testY))
    ova_weights = get_ova_weights(id_train,trainY,id_test, testY)
    trova_bin_preds = get_ova_bin_preds(ova_weights,id_train)
    id_res.append(OVA_classifier(trova_bin_preds, trainY,train=True))
    ova_bin_preds = get_ova_bin_preds(ova_weights, id_test)
    id_res.append(OVA_classifier(ova_bin_preds, testY))

    sig_train = sig_data(trainX,w,b)
    sig_test = sig_data(testX,w,b)
    sig_res = []
    print("g(x) = sigmoid function")
    ovo_weights = get_ovo_weights(sig_train,trainY,sig_test, testY)
    trovo_bin_preds = get_ovo_bin_preds(ovo_weights,sig_train)
    sig_res.append(OVO_classifier(trovo_bin_preds,trainY,train=True))
    ovo_bin_preds = get_ovo_bin_preds(ovo_weights,sig_test)
    sig_res.append(OVO_classifier(ovo_bin_preds, testY))
    ova_weights = get_ova_weights(sig_train,trainY,sig_test, testY)
    trova_bin_preds = get_ova_bin_preds(ova_weights,sig_train)
    sig_res.append(OVA_classifier(trova_bin_preds, trainY))
    ova_bin_preds = get_ova_bin_preds(ova_weights, sig_test,train=True)
    sig_res.append(OVA_classifier(ova_bin_preds, testY))

    sin_train = sin_data(trainX,w,b)
    sin_test = sin_data(testX,w,b)
    sin_res = []
    print("g(x) = sin(x)")
    ovo_weights = get_ovo_weights(sin_train,trainY,sin_test, testY)
    trovo_bin_preds = get_ovo_bin_preds(ovo_weights,sin_train)
    sin_res.append(OVO_classifier(trovo_bin_preds,trainY,train=True))
    ovo_bin_preds = get_ovo_bin_preds(ovo_weights,sin_test)
    sin_res.append(OVO_classifier(ovo_bin_preds, testY))
    ova_weights = get_ova_weights(sin_train,trainY,sin_test, testY)
    trova_bin_preds = get_ova_bin_preds(ova_weights,sin_train,)
    sin_res.append(OVA_classifier(trova_bin_preds, trainY,train=True))
    ova_bin_preds = get_ova_bin_preds(ova_weights, sin_test)
    sin_res.append(OVA_classifier(ova_bin_preds, testY))

    max_train = max_data(trainX,w,b)
    max_test = max_data(testX,w,b)
    max_res = []
    print("g(x) = Rectified Linear Unit Function")
    ovo_weights = get_ovo_weights(max_train,trainY,max_test, testY)
    trovo_bin_preds = get_ovo_bin_preds(ovo_weights,max_train)
    max_res.append(OVO_classifier(trovo_bin_preds,trainY,train=True))
    ovo_bin_preds = get_ovo_bin_preds(ovo_weights,max_test)
    max_res.append(OVO_classifier(ovo_bin_preds, testY))
    ova_weights = get_ova_weights(max_train,trainY,max_test, testY)
    trova_bin_preds = get_ova_bin_preds(ova_weights,max_train)
    max_res.append(OVA_classifier(trova_bin_preds, trainY, train=True))
    ova_bin_preds = get_ova_bin_preds(ova_weights, max_test)
    max_res.append(OVA_classifier(ova_bin_preds, testY))

    return id_res,sig_res,sin_res,max_res

    

    '''
    ova_weights = get_ova_weights(trainX,trainY,testX, testY)
ova_bin_preds = get_ova_bin_preds(ova_weights, testX)
ova_preds, ova_confusion_matrix = OVA_classifier(ova_bin_preds, testY)

L = 1000
w = np.random.normal(size=(L,785))
b = np.random.normal(size=(L,1))

new_train = mpt.max_data(trainX,w,b)
new_test = mpt.max_data(testX,w,b)

print('starting training')
ovo_weights = mpt.get_ovo_weights(new_train,trainY,new_test, testY)
trovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,new_train)
mpt.OVO_classifier(trovo_bin_preds,trainY,train=True)
ovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,new_test)
ovo_preds, ovo_confusion_matrix = mpt.OVO_classifier(ovo_bin_preds, testY)

ova_weights = mpt.get_ova_weights(new_train,trainY,new_test, testY)
ova_bin_preds = mpt.get_ova_bin_preds(ova_weights, new_test)
ova_preds, ova_confusion_matrix = mpt.OVA_classifier(ova_bin_preds, testY)
'''


'''
L = 500

id_train = mpt.randomize_data(trainX,w,b)
id_test = mpt.randomize_data(testX,w,b)
print("g(x) = x")
ovo_weights = mpt.get_ovo_weights(id_train,trainY,id_test, testY)
trovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,id_train)
mpt.OVO_classifier(trovo_bin_preds,trainY,train=True)
ovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,id_test)
mpt.OVO_classifier(ovo_bin_preds, testY)
ova_weights = mpt.get_ova_weights(id_train,trainY,id_test, testY)
trova_bin_preds = mpt.get_ova_bin_preds(ova_weights,id_train)
mpt.OVA_classifier(trova_bin_preds, trainY)
ova_bin_preds = mpt.get_ova_bin_preds(ova_weights, id_test)
mpt.OVA_classifier(ova_bin_preds, testY)

sig_train = mpt.sig_data(trainX,w,b)
sig_test = mpt.sig_data(testX,w,b)
print("g(x) = sigmoid function")
ovo_weights = mpt.get_ovo_weights(sig_train,trainY,sig_test, testY)
trovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,sig_train)
mpt.OVO_classifier(trovo_bin_preds,trainY,train=True)
ovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,sig_test)
mpt.OVO_classifier(ovo_bin_preds, testY)
ova_weights = mpt.get_ova_weights(sig_train,trainY,sig_test, testY)
trova_bin_preds = mpt.get_ova_bin_preds(ova_weights,sig_train)
mpt.OVA_classifier(trova_bin_preds, trainY)
ova_bin_preds = mpt.get_ova_bin_preds(ova_weights, sig_test)
mpt.OVA_classifier(ova_bin_preds, testY)

sin_train = mpt.sin_data(trainX,w,b)
sin_test = mpt.sin_data(testX,w,b)
print("g(x) = sin(x)")
ovo_weights = mpt.get_ovo_weights(sin_train,trainY,sin_test, testY)
trovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,sin_train)
mpt.OVO_classifier(trovo_bin_preds,trainY,train=True)
ovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,sin_test)
mpt.OVO_classifier(ovo_bin_preds, testY)
ova_weights = mpt.get_ova_weights(sin_train,trainY,sin_test, testY)
trova_bin_preds = mpt.get_ova_bin_preds(ova_weights,sin_train)
mpt.OVA_classifier(trova_bin_preds, trainY)
ova_bin_preds = mpt.get_ova_bin_preds(ova_weights, sin_test)
mpt.OVA_classifier(ova_bin_preds, testY)

max_train = mpt.max_data(trainX,w,b)
max_test = mpt.max_data(testX,w,b)
print("g(x) = Rectified Linear Unit Function")
ovo_weights = mpt.get_ovo_weights(max_train,trainY,max_test, testY)
trovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,max_train)
mpt.OVO_classifier(trovo_bin_preds,trainY,train=True)
ovo_bin_preds = mpt.get_ovo_bin_preds(ovo_weights,max_test)
mpt.OVO_classifier(ovo_bin_preds, testY)
ova_weights = mpt.get_ova_weights(max_train,trainY,max_test, testY)
trova_bin_preds = mpt.get_ova_bin_preds(ova_weights,max_train)
mpt.OVA_classifier(trova_bin_preds, trainY, train=True)
ova_bin_preds = mpt.get_ova_bin_preds(ova_weights, max_test)
mpt.OVA_classifier(ova_bin_preds, testY)
'''
    



    







'CS-smote'
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score,confusion_matrix, recall_score, matthews_corrcoef
#from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import warnings
import time
import math
from sklearn.utils import shuffle
from sklearn.cluster import AgglomerativeClustering
import random
from kneed import KneeLocator
warnings.filterwarnings("ignore")
def loadDataSet(fileName, splitChar=','):
    dataSet = []  # 存放数据
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    # np.power（x，2） 是对应元素的2次方


start = time.perf_counter()
dataset = loadDataSet(r'C:/Users/Stephen117/Desktop/shuju/yeast1.csv', splitChar=',')
dataset = np.mat(dataset)
UPPER_BOUND, LOWER_BOUND = 0.9, 0.1
data_x = np.array(dataset[:, :-1], dtype=np.float32)
data_y = np.array(dataset[:, -1], dtype=np.int32)
x_min = np.min(data_x, axis=0)
x_max = np.max(data_x, axis=0)
data_x = (data_x - x_min) / x_max * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND
data_num = data_x.shape[0]
feature_num = data_x.shape[1]
rnd_index = np.arange(data_num)
np.random.shuffle(rnd_index)
data_x = data_x[rnd_index]
data_y = data_y[rnd_index]
dataset = np.append(data_x, data_y, axis=1)
dataset1 = np.mat(dataset)
dataset2 = shuffle(dataset1)

F1s1 = []
recall_s1 = []
Gms1 = []
roc_auc_score1 = []
Mcc1 = []
F1s2 = []
recall_s2 = []
Gms2 = []
roc_auc_score2 = []
F1s3 = []
recall_s3 = []
Gms3 = []
roc_auc_score3 = []
k = 5
Xminor = dataset1[np.nonzero(dataset1[:, -1].A == 0)[0]]
Xmajor = dataset1[np.nonzero(dataset1[:, -1].A == 1)[0]]
#w = FindEps(Xminor)
#print(w)
neigh = NearestNeighbors(n_neighbors=k).fit(dataset1[:, 0:-1])
_, neigh_idx = neigh.kneighbors(Xminor[:, 0:-1])
X1 = dataset1[neigh_idx]
# print(dataset[neigh_idx])
x2 = X1[:, :, -1]
# print(x2)
cishu = np.sum(x2, axis=1)
# print(cishu)
Xminor1 = np.append(Xminor, cishu, axis=1)
Xminor2 = Xminor1[np.nonzero(Xminor1[:, -1].A != 4)[0]]
Xminor_clean = Xminor2[:, 0:-1]
# print(Xminor_clean)
X_cof = []
Xgenmin = []
Xminorf = Xminor_clean[:, 0:-1]

#clustering = AgglomerativeClustering(linkage='average', n_clusters=2).fit(Xminor_clean)
#print(clustering.labels_)
k_means = KMeans(n_clusters=2, random_state=0).fit(Xminor_clean)
#print(k_means.labels_)

L1_center, L2_center = k_means.cluster_centers_
#print(L1_center,L2_center)
k_means_labels = np.mat(k_means.labels_)
#k_means_labels = np.mat(clustering.labels_)
arr1 = k_means_labels.tolist()
C11 = np.mat(arr1)
C21 = C11.tolist()
transs = []
for i in range(len(C21[0])):
    a = []
    for j in range(len(C21)):
        a.append(C21[j][i])
    transs.append(a)
trans11 = np.mat(transs)


Xminor_clean_labels = np.append(Xminor_clean, trans11, axis=1)
Xminor_L1 = Xminor_clean_labels[np.nonzero(Xminor_clean_labels[:, -1].A == 0)[0]]
Xminor_L2 = Xminor_clean_labels[np.nonzero(Xminor_clean_labels[:, -1].A == 1)[0]]


for xi in Xminor_clean_labels:
    xi_neigh_center = []
    Xi_co = []
    dist_center = []

    for i in range(1, k):
        neigh1 = NearestNeighbors(n_neighbors=i).fit(Xminor_clean_labels[:, 0:-2])
        __, neigh1_idx = neigh1.kneighbors(xi[:, 0:-2])
        neighboor = Xminor_clean_labels[:, 0:-2][neigh1_idx]
        neighboor = np.squeeze(neighboor, axis=(0))
        xi_neigh_center.append((np.sum(neighboor, axis=0)) / i)
    xi_neigh_center1 = np.squeeze(xi_neigh_center)
    for j in range(1, k - 1):
        dist_center.append(distEclud(xi_neigh_center1[j - 1], xi_neigh_center1[j]))
    for q in range(1, len(dist_center)):
        Xi_co.append(sum(abs(xi_neigh_center1[q + 1] - xi_neigh_center1[q])))
    Xi_cof = sum(Xi_co)
    X_cof.append(Xi_cof)
X_cof = np.squeeze(X_cof)
#print(X_cof)
X_cof = sorted(X_cof)
#print(T)
list1 = []
for i in range(len(X_cof)):
    list1.append(i)
#print(list1)
kn = KneeLocator(list1, X_cof, S=1.0, curve="convex", direction="increasing")
#print(kn)
kn.plot_knee(figsize=(8, 6))
Eps = kn.knee_y

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.grid(linestyle="--", alpha=0.3)
plt.title('', fontsize=16)
plt.xlabel('Data points sorted by COF value in ascending order', fontsize=16)
plt.ylabel('COF value', fontsize=16)
#plt.show()

for i in tqdm(range(5)):
    X_cof = []

    for xi in Xminor_clean_labels:
        xi1_neigh_center = []
        Xi1_co = []
        dist_center1 = []

        for i in range(1, k):
            neigh1 = NearestNeighbors(n_neighbors=i).fit(Xminor_clean_labels[:, 0:-2])
            __, neigh1_idx = neigh1.kneighbors(xi[:, 0:-2])
            neighboor = Xminor_clean_labels[:, 0:-2][neigh1_idx]
            neighboor = np.squeeze(neighboor, axis=(0))
            xi1_neigh_center.append((np.sum(neighboor, axis=0)) / i)
        xi_neigh_center2 = np.squeeze(xi1_neigh_center)
        for j in range(1, k - 1):
            dist_center1.append(distEclud(xi_neigh_center2[j - 1], xi_neigh_center2[j]))
        for q in range(1, len(dist_center1)):
            Xi1_co.append(sum(abs(xi_neigh_center2[q + 1] - xi_neigh_center2[q])))
        Xi_cof = sum(Xi1_co)

        if Xi_cof > Eps:
            Xgenmin.append(xi)
    Xgenmin1 = np.squeeze(Xgenmin)
    Xgenmin1 = np.mat(Xgenmin1)
    Xminor_L11 = Xgenmin1[np.nonzero(Xgenmin1[:, -1].A == 0)[0]]
    Xminor_L22 = Xgenmin1[np.nonzero(Xgenmin1[:, -1].A == 1)[0]]


    Clean_date = np.append(Xmajor, Xminor_clean, axis=0)

    Gen_min = []
    N1 = len(Xmajor) - len(Xminor_clean)
    for i in range(N1):
        x_samples = Xgenmin1
        rand_arr = np.arange(x_samples.shape[0])
        rand_arrl1 = np.arange(Xminor_L1.shape[0])
        rand_arrl2 = np.arange(Xminor_L2.shape[0])
        np.random.shuffle(rand_arr)
        np.random.shuffle(rand_arrl1)
        np.random.shuffle(rand_arrl2)
        x_samples1 = x_samples[rand_arr[0]]
        y_samples1 = Xminor_L1[rand_arrl1[0]]
        y_samples2 = Xminor_L2[rand_arrl2[0]]
        gap = random.uniform(0, 1)
        gap1 = random.uniform(0, 1)
        if x_samples1[:,-1] == 0 :
            tem = (1-gap)*x_samples1[:, 0:-1] + gap*y_samples1[:, 0:-1]
            s = (1-gap1)*L2_center + gap1*tem
        else:
            tem = (1-gap)*x_samples1[:, 0:-1] + gap*y_samples2[:, 0:-1]
            s = (1 - gap1) * L1_center + gap1 * tem
        Gen_min.append(s)
        Gen_min1 = np.squeeze(Gen_min)

    Xgenmin_maj = np.append(Gen_min1, Clean_date, axis=0)
    end = time.perf_counter()
    #X11 = Clean_date[:, 0:-1]
    #y11 = Clean_date[:, -1]
    X12 = Xgenmin_maj[:, 0:-1]
    y12 = Xgenmin_maj[:, -1]

    train_indexs = []
    test_indexs = []
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in folds.split(Xgenmin_maj):  # 调用split方法切分数据
        #print('train_index:%s , test_index: %s ' % (train_index, test_index))
        train_indexs.append(train_index), test_indexs.append(test_index)

    for i in range(5):
        X2_train, y2_train = X12[train_indexs[i]], y12[train_indexs[i]]
        X2_test, y2_test = X12[test_indexs[i]], y12[test_indexs[i]]

        # print(fold1_train_data)

        model1 = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
        model2 = KNeighborsClassifier(n_neighbors=5)
        model3 = DecisionTreeClassifier(criterion='entropy')

        model1.fit(X2_train, y2_train)
        y2_pre1 = model1.predict(X2_test)
        confusion1 = confusion_matrix(y2_test, y2_pre1)

        TP1, FP1, FN1, TN1 = confusion1[0, 0], confusion1[1, 0], confusion1[0, 1], confusion1[1, 1]
        TPR1 = TP1 / (TP1 + FN1)
        TNR1 = TN1 / (TN1 + FP1)

        F1s1.append(f1_score(y2_test, y2_pre1, pos_label = 0))
        recall_s1.append(recall_score(y2_test, y2_pre1))
        Gms1.append(math.sqrt(TPR1 * TNR1))
        roc_auc_score1.append(roc_auc_score(y2_test, y2_pre1))
        Mcc1.append(matthews_corrcoef(y2_test, y2_pre1))
      
F1ss_svm = sum(F1s1)/25
print('RF F1-score is %s' % F1ss_svm)
recall_svm = sum(recall_s1)/25
#print('recall_score is %s' % F1ss)
Gms_svm = sum(Gms1)/25
print('RF Gmean is %s' % Gms_svm)
roc_auc_score_svm = sum(roc_auc_score1)/25
print('RF AUC is %s' % roc_auc_score_svm)
MCC_ADA = sum(Mcc1)/25
print('RF MCC is %s' % MCC_ADA)
df = pd.DataFrame(results)
df.to_excel('algorithm_results.xlsx', index=False)

print('finish all in %s' % str(end - start))



from imblearn.over_sampling._smote.base import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import Lasso, SGDClassifier
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import RandomUnderSampler,TomekLinks
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from category_encoders import BaseNEncoder
from sklearn.svm import OneClassSVM


#----------------------------------------Outlier ----------------------------------------------
def topModels(models,X_Train,Y_Train,X_Test,Y_Test):
    fig,axs = plt.subplots(ncols=3, nrows=1)
    (ax1,ax2,ax3) = axs
    fig.suptitle('Top 3 comparisson', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')
    listOfAxs2 = [[ax1,'Naive Bayes'],[ax2,'Random Forest'],[ax3,'ADA Boost']]
    i = 0
    # print(X_Test,'=============')
    for model in models:
        model.fit(X_Train,Y_Train)
        nbpred = model.predict(X_Test)
        ACC = evaluate_model(nbpred,model,X_Test,Y_Test)
        listOfAxs2[i][0].plot(ACC['fpr'],ACC['tpr'],label=str(listOfAxs2[i][1])+', = {:0.5f}'.format(ACC['auc']))
        #X,Y axis config
        listOfAxs2[i][0].set_xlabel('False Positive Rate', fontweight='bold')
        listOfAxs2[i][0].set_ylabel('True Positive Rate', fontweight='bold')
        # Create legend & title
        listOfAxs2[i][0].set_title('ROC Curve'+str(listOfAxs2[i][1]), fontsize=5, fontweight='bold')
        listOfAxs2[i][0].legend(loc=4)
        i +=1
    # plt.show()

#---------------------------------------Outlier END --------------------------------------------

#----------------------------------------Outlier ----------------------------------------------
def outlierRemoval(dat,model,sizes):
    X_Train = dat[:, :-1][sizes[0]]
    Y_Train = dat[:,-1][sizes[0]]
    X_Test = dat[:, :-1][sizes[1]]
    Y_Test = dat[:,-1][sizes[1]]

    print('Size before removal',X_Train.shape)
    # X_Train,X_Test,Y_Train,Y_Test = train_test_split(features,dat[:, -1])
    out = model.fit_predict(X_Train,Y_Train)
    X_Train, Y_Train = X_Train[(out != -1), :], Y_Train[(out != -1)]   

    print('Size after removal',X_Train.shape)
    return X_Train, Y_Train,X_Test,Y_Test
#---------------------------------------Outlier END --------------------------------------------

#-----------------------------------MODEL EVALUATION START--------------------------------------
def evaluate_model(y_pred,model, x_test, y_test):

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred,normalize = False)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    if isinstance(model,Lasso) or isinstance(model,SGDClassifier):
        y_pred_proba = y_pred
    else:
        y_pred_proba = model.predict_proba(x_test)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}
#-----------------------------------MODEL EVALUATION END--------------------------------------

#------------------------------------Model Implementations---------------------------------
def modelBuilds(X_Train,Y_Train,X_Test,Y_Test,ax,i):
    acc = []
    # Naive model
    model = GaussianNB()
    model.fit(X_Train,Y_Train)
    nbpred = model.predict(X_Test)
    NBACC = evaluate_model(nbpred,model,X_Test,Y_Test)
    acc.append([NBACC['auc'],NBACC,model])
    
    #KNN
    knn = KNeighborsClassifier()
    knn.fit(X_Train,Y_Train)
    knpred = knn.predict(X_Test)
    KNNACC = evaluate_model(knpred,knn,X_Test,Y_Test)
    acc.append([KNNACC['auc'],KNNACC,knn])

    #Random Forest  BEST---------------
    rf = RandomForestClassifier()#{'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 200}
    rf.fit(X_Train,Y_Train)
    rfpred = rf.predict(X_Test)
    RFACC = evaluate_model(rfpred,rf,X_Test,Y_Test)
    acc.append([RFACC['auc'],RFACC,rf])

    #Decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_Train,Y_Train)
    dtpred = tree.predict(X_Test)
    DTACC = evaluate_model(dtpred,tree,X_Test,Y_Test)
    acc.append([DTACC['auc'],DTACC,tree])
    
    #Ada boost
    adb = AdaBoostClassifier()
    adb.fit(X_Train,Y_Train)
    adpred = adb.predict(X_Test)
    ADACC = evaluate_model(adpred,adb,X_Test,Y_Test)
    acc.append([ADACC['auc'],ADACC,adb])
        
    # Lasso
    ls = Lasso()
    ls.fit(X_Train,Y_Train)
    lspred = knn.predict(X_Test)
    LSACC = evaluate_model(lspred,ls,X_Test,Y_Test)
    acc.append([LSACC['auc'],LSACC,ls])
    
    #Scholastic
    sc = SGDClassifier()
    sc.fit(X_Train,Y_Train)
    scpred = sc.predict(X_Test)
    SCACC = evaluate_model(scpred,sc,X_Test,Y_Test)
    acc.append([SCACC['auc'],SCACC,sc])

    #Printing all accuracies
    print('Accuracy of model:',NBACC['acc'])
    print('Accuracy of KNN:',KNNACC['acc'])
    print('Accuracy of Regresion Forest:',RFACC['acc'])
    print('Accuracy of Decision Tree:',DTACC['acc'])
    print('Accuracy of ADA boost:',ADACC['acc'])
    print('Accuracy of Lasso:',LSACC['acc'])
    print('Accuracy of Scholastic:',SCACC['acc'])
    
    # Make the plot
    ax.plot(NBACC['fpr'],NBACC['tpr'],label='Naive model, = {:0.5f}'.format(NBACC['auc']))
    ax.plot(KNNACC['fpr'],KNNACC['tpr'],label='KNN, = {:0.5f}'.format(KNNACC['auc']))
    ax.plot(RFACC['fpr'],RFACC['tpr'],label='Random Forest, = {:0.5f}'.format(RFACC['auc']))
    ax.plot(DTACC['fpr'],DTACC['tpr'],label='Decision Tree, = {:0.5f}'.format(DTACC['auc']))
    ax.plot(ADACC['fpr'],ADACC['tpr'],label='Ada boost, = {:0.5f}'.format(ADACC['auc']))
    ax.plot(LSACC['fpr'],LSACC['tpr'],label='Lasso, = {:0.5f}'.format(LSACC['auc']))
    ax.plot(SCACC['fpr'],SCACC['tpr'],label='Scholastic, = {:0.5f}'.format(SCACC['auc']))
    #X,Y axis config
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    # Create legend & title
    ax.set_title('ROC Curve'+str(i+1), fontsize=5, fontweight='bold')
    ax.legend(loc=4)
    return acc
#------------------------------------Model Implementations END---------------------------------

#------------------------------------DATASET IMBALANCE ---------------------------------
def dataBalanceModels(X_Train, Y_Train,X_Test,Y_Test):  
    print('-----------Random Under Sampling--------------')  
    sm = RandomOverSampler()#For tackeling data imbalance
    X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)
    regRunModel(X_Train, Y_Train,X_Test,Y_Test)

    print('-----------Random Over Sampling--------------')  
    sm = RandomUnderSampler()#For tackeling data imbalance
    X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)
    regRunModel(X_Train, Y_Train,X_Test,Y_Test)

    print('-----------SMOTE--------------')  
    sm = SMOTE()#For tackeling data imbalance
    X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)
    regRunModel(X_Train, Y_Train,X_Test,Y_Test)

    print('-----------TomekLinks--------------')  
    sm = TomekLinks()#For tackeling data imbalance
    X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)
    regRunModel(X_Train, Y_Train,X_Test,Y_Test)   

#------------------------------------DATASET IMBALANCE END---------------------------------

#------------------------------------Cross validation START-------------------------------------
def crossVal():
    #Model created
    kf = StratifiedKFold(shuffle=True, n_splits=6, random_state=1)
    i = 0
    models = []
    for train_index,val_index in kf.split(dat[:, :-1],bankCopy['y'],):
        # evaluate model
        acc = modelBuilds(dat[:, :-1][train_index],bankCopy['y'][train_index],dat[:, :-1][val_index],bankCopy['y'][val_index],listOfAxs[i],i)
        acc.sort(key=lambda x : x[0])
        # print(train_index)
        models.append([acc,[train_index,val_index]])
        print("------------------Itteration------------------",i+1)
        i+=1
    plt.show()
    # print([[i[0][]] for i in models])
    return models
#------------------------------------Cross validation END-------------------------------------

#------------------------------------Req Model Run Start---------------------------------------
def hypModelRun(model,param_grid,X_Train,Y_Train):
    gd = GridSearchCV(model,param_grid=param_grid, scoring='accuracy', cv=5)
    gd.fit(X_Train,Y_Train)   
    print(gd.best_params_) 
    print(gd.best_score_)
    return gd
#------------------------------------Req Model Run END---------------------------------------
    
def regRunModel(X_Train,Y_Train,X_Test,Y_Test): #Hyper tuened just for regresiion forest
    fig,ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle('Top 3 comparisson', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')
    model = RandomForestClassifier(criterion='gini',max_depth=4,max_features='log2',n_estimators=200)
    model.fit(X_Train,Y_Train)
    nbpred = model.predict(X_Test)
    ACC = evaluate_model(nbpred,model,X_Test,Y_Test)
    print('Accuracy of Regresion Forest:',ACC['acc'])
    # Make the plot
    ax.plot(ACC['fpr'],ACC['tpr'],label='Regression Forest model, = {:0.5f}'.format(ACC['auc']))
    #X,Y axis config
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    # Create legend & title
    ax.set_title('ROC Curve ', fontsize=5, fontweight='bold')
    ax.legend(loc=4)
    plt.show()

#----------------------------------------1.1-------------------------------------------------

dataBankDf = pd.read_csv('archive/bank-additional-full.csv', delimiter=';') #Data loading 
col = ['age', 'campaign','pdays','previous','emp.var.rate','cons.price.idx','euribor3m','nr.employed','duration']

bankCopy = dataBankDf.copy() #Making a copy to make change


#String data nrmalization
enc = OneHotEncoder(sparse=False)
strCol = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
strTrans = enc.fit_transform(dataBankDf[strCol])
strTrans = pd.DataFrame(strTrans ,columns=enc.get_feature_names(strCol))

# Droping string data from copy and concat new normalized data
bankCopy = dataBankDf.drop(strCol,axis=1)
bankCopy = pd.concat([strTrans,bankCopy],axis=1)

#converting target data in 0 and 1 
bankCopy['y'] = bankCopy['y'].apply(lambda x :1 if x=='yes' else 0)
col = dataBankDf.columns
dat = bankCopy.values

#----------------Visualization------------------------
fig, axs = plt.subplots(ncols=3, nrows=2)
(ax1,ax2,ax3),(ax4,ax5,ax6) = axs
listOfAxs = [ax1,ax2,ax3,ax4,ax5,ax6]
fig.suptitle('Imbalance Tecnhique Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

#---------------ALL MODELS CROSS VALL-------------------
allModels1 = crossVal()

#---------------OUTLER REMOVAL---------------------------
model = IsolationForest()
X_Train, Y_Train,X_Test,Y_Test = outlierRemoval(bankCopy.values,model,allModels1[2][1])

#Numerical data transformation
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)

#-----------------RANDOM UNDER SAMPLING-------------------
sm = RandomOverSampler()#For tackeling data imbalance
X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)  #Undersampled


# ------------------ TOP 3 MODELS------------------------

randFor = RandomForestClassifier(criterion='gini',max_depth=4,max_features='log2',n_estimators=200)
# Hyperparameter tuning code for Random Forest
# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
# mod1 = hypModelRun(randFor,param_grid)

nB = GaussianNB(var_smoothing=0.2848035868435802)
# Hyperparameter tuning code for GaussianNB
# param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
# mod2 = hypModelRun(nB,param_grid)

ad = AdaBoostClassifier(learning_rate=0.001,n_estimators=500)
# Hyperparameter tuning code for ADA boost
# param_grid = {'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}
# mod3 = hypModelRun(ad,param_grid)

models=[nB,randFor,ad]
topModels(models,X_Train,Y_Train,X_Test,Y_Test) #TOP 3 MODEL RUN



#----------------------------------------1.2-----------------------------------------
dataBankDf = pd.read_csv('archive/bank-additional-full.csv', delimiter=';') #Data loading
dataBankDf = dataBankDf.drop('duration', axis=1) #No need for any caluclations

# ------------------Feature Selection START---------------------------
col = ['age', 'campaign','pdays','previous','emp.var.rate','cons.price.idx','euribor3m','nr.employed']
corrTmp = dataBankDf[col]
corrData = corrTmp.corr()
fig = plt.figure(figsize=(24, 24))
sns.heatmap(corrData, annot=True, fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.show()
highly_correlated = abs(corrData[corrData > 0.95])
print(highly_correlated[highly_correlated < 1.0].stack().to_string())
dataBankDf = dataBankDf.drop('euribor3m', axis=1) #Feature removed
# ------------------Feature Selection END---------------------------

strCol = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
sum = BaseNEncoder(cols=strCol, base=3)
endcoded = sum.fit_transform(dataBankDf[strCol])
encPd = pd.DataFrame(endcoded, columns=sum.get_feature_names())

# Droping string data from copy and concat new normalized data
dataBankDf = dataBankDf.drop(strCol,axis=1)
dataBankDf = pd.concat([encPd,dataBankDf],axis=1)

#converting target data in 0 and 1 
dataBankDf['y'] = dataBankDf['y'].apply(lambda x :1 if x=='yes' else 0)
col = dataBankDf.columns
dat = dataBankDf.values
bankCopy = dataBankDf.copy()

#---------------ALL MODELS CROSS VALL-------------------
allModels2 = crossVal()

# -----------Outlier removal--------------
model = OneClassSVM(nu=0.01)
X_TrainSVM, Y_TrainSVM,X_TestSVM,Y_TestSVM = outlierRemoval(dataBankDf.values,model,allModels2[2][1]) #Outlier removal

# #Numerical data transformation
scaler = StandardScaler()
X_TrainSVM = scaler.fit_transform(X_TrainSVM)

# -----------Random Under Sampling--------------  
sm = RandomOverSampler()#For tackeling data imbalance
X_TrainSVM, Y_TrainSVM = sm.fit_resample(X_TrainSVM, Y_TrainSVM)  #Undersampled

regRunModel(X_TrainSVM, Y_TrainSVM,X_TestSVM,Y_TestSVM)

#----------------------Experimental tuned Isolation Forest outlier removal------------------
model = IsolationForest(random_state=47)
X_Train, Y_Train,X_Test,Y_Test = outlierRemoval(dataBankDf.values,model,allModels2[2][1]) #Outlier removal

# #Numerical data transformation
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)

# -----------Random Under Sampling--------------  
sm = RandomOverSampler()#For tackeling data imbalance
X_Train, Y_Train = sm.fit_resample(X_Train, Y_Train)  #Undersampled

regRunModel(X_Train, Y_Train,X_Test,Y_Test)


#----------------------------------------1.3-----------------------------------------

model = IsolationForest()
X_Train, Y_Train,X_Test,Y_Test = outlierRemoval(bankCopy.values,model,allModels1[2][1]) #Outlier removal

# #Numerical data transformation
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)

dataBalanceModels(X_Train, Y_Train,X_Test,Y_Test)
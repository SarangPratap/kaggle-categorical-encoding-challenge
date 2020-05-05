import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


""" Data Preprocessing """
                            
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

t = train[train.target == 1]
train = pd.concat([train,t],axis = 0)

''' Missing Values '''

for i in train.columns:
    train[i].fillna(train[i].mode().values[0],inplace=True)

for i in test.columns:
    test[i].fillna(test[i].mode().values[0],inplace=True)
"""Feature Engineering """
                       
""" NOMINAL FEATURES """
''' Mean Encoding 1 '''

'''def mean_encode(train, test, col, target, reg_method=None,
                alpha=5, add_random=False, rmean=0, rstd=0.1, folds=4):
    
    target_mean_global = train[col].mean()
    
        # Getting means for test data
    nrows = train.groupby(col)[target].count()
    target_mean = train.groupby(col)[target].mean()
    target_mean_adj = (target_mean*nrows + target_mean_global*alpha)/(nrows+alpha)
    
        # Mapping means to test data
    test[:,"{}_mean_encode".format(col)] = test[col].map(target_means_cats_adj)
    test.drop([col])
    
        # Getting a train encodings
    if reg_method == 'expanding_mean':
        train_shuffled = train.sample(frac=1, random_state=1)
        cumsum = train_shuffled.groupby(col)[target].cumsum() - train_shuffled[target]
        cumcnt = train_shuffled.groupby(col).cumcount()
        encoded_col_train = cumsum/(cumcnt)
        encoded_col_train.fillna(target_mean_global, inplace=True)
        if add_random:
               encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        train[:,"{}_mean_encode".format(col)] = encoded_col_train
        train.drop([col])
        
    elif (reg_method == 'k_fold') and (folds > 1):
        kfold = StratifiedKFold(n_split = folds, shuffle=True, random_state=1)
                                      
        for tr_in, val_ind in kfold.split(train[col],train[target]):
                # divide data
            df_for_estimation, df_estimated = train.iloc[tr_in], train.iloc[val_ind]
            # getting means on data for estimation (all folds except estimated)
            nrow = df_for_estimation.groupby(col)[target].count()
            target_mean = df_for_estimation.groupby(col)[target_col].mean()
            target_mean_adj = (target_mean *nrow + 
                                         target_mean_global*alpha)/(nrow+alpha)
                # Mapping means to estimated fold
            encoded_col_train = df_estimated[col].map(target_mean_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train.shape[0]))
           
            
        train[:,"{}_mean_encode".format(col)] = encoded_col_train '''        

''' Mean Encoding 2 '''

def Mean_encode(col):
    mean = train['target'].mean()
    
    agg = train.groupby(col)['target'].agg(['count','mean'])
    counts = agg['count']
    means = agg['mean']
    weight = 100
    
    smooth = ((counts * means) + (weight * mean)) / (counts + weight)
    
    train.loc[:,"{}_mean_encode".format(col)] = train[col].map(smooth)
    test.loc[:,"{}_mean_encode".format(col)] = test[col].map(smooth)


Mean_encode('nom_0')
Mean_encode('nom_1')
Mean_encode('nom_2')
Mean_encode('nom_3')
Mean_encode('nom_4')
Mean_encode('nom_5')
Mean_encode('nom_6')
Mean_encode('nom_7')
Mean_encode('nom_8')
Mean_encode('nom_9')

''' Mean Encoding Ordinal and Cyclic Features '''

Mean_encode('ord_0')
Mean_encode('ord_1')
Mean_encode('ord_2')
Mean_encode('ord_3')
Mean_encode('ord_4')
Mean_encode('ord_5')
Mean_encode('day')
Mean_encode('month')

train = train.drop(['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5','day','month'],axis=1)
test = test.drop(['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5','day','month'],axis=1)

''' '''

train = train.drop(['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)
test = test.drop(['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1)
 
''' Merging train and test '''
Train = pd.concat([train,test])


''' BINARY FEATURES '''
         
Train["bin_3"] = Train["bin_3"].apply(lambda x: 1 if x=='T' else (0 if x=='F' else None))
Train["bin_4"] = Train["bin_4"].apply(lambda x: 1 if x=='Y' else (0 if x=='N' else None))
  



""" CYCLIC FEATURES"""
"""
Train['day_sin'] = np.sin(2 * np.pi * Train['day']/7)
Train['day_cos'] = np.cos(2 * np.pi * Train['day']/7)

Train['month_sin'] = np.sin(2 * np.pi * Train['month']/12)
Train['month_cos'] = np.cos(2 * np.pi * Train['month']/12)


Train = Train.drop(['day','month'],axis=1)
Train = Train.drop(['bin_0','bin_2'],axis=1)
"""
''' Data Splitting '''
                            
n_train = Train.head(712323)
trainlabel = n_train['target']
n_train = n_train.drop(['target','id'],axis = 1)

Test = Train.tail(400000)
Id_test=Test['id'] 
Test = Test.drop(['target','id'],axis =1)

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(n_train, trainlabel, test_size=0.2)

''' Models '''

''' CatBoost '''

#categorical_features_indices = np.where(n_train.dtypes != np.float)[0]

from catboost import CatBoostClassifier
SEED=1
param={'loss_function':'Logloss',
       'eval_metric' : 'AUC',
       'verbose': 200,      
       'random_seed':SEED}


param_rad={
            'learning_rate': [0.03,0.1],
        'depth': [4,5] ,
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'grow_policy':['SymmetricTree','Lossguide'],
        'bagging_temperature':[0.1,0.2,0.5,0.4,0.7,1]       }

model=CatBoostClassifier(task_type='GPU')
result=model.randomized_search(param_rad,X_n_train,y_n_train,cv=5)

cat = CatBoostClassifier(**param,grow_policy='Lossguide')

cat.fit(X_n_train,y_n_train,
        eval_set=(X_n_test,y_n_test),
        #cat_features =  categorical_features_indices,
        use_best_model=True,plot=True)

prediction = cat.predict_proba(Test,
                                ntree_start=0,
                                ntree_end=0,
                                thread_count=1,
                                verbose=None)[:,1]




'''CatBoost with Cross Validation'''
'''
from catboost import Pool
from sklearn.model_selection import StratifiedKFold
#cat_features = np.where(n_train.dtypes != np.float)[0]

n_fold = 4 # amount of data folds
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)

params = {'loss_function':'Logloss',
          'eval_metric':'AUC',
          'verbose': 200,
          'random_seed': SEED
         }

test_data = Pool(data=Test
                 #cat_features=cat_features
                 )
                
scores = []
prediction = np.zeros(Test.shape[0])

for fold_n, (train_index, valid_index) in enumerate(folds.split(n_train, trainlabel)):
    
    X_train, X_valid = n_train.iloc[train_index], n_train.iloc[valid_index] # train and validation data splits
    y_train, y_valid = trainlabel.iloc[train_index], trainlabel.iloc[valid_index]
    
    train_data = Pool(data=X_train, 
                      label=y_train
                      #cat_features=cat_features
                      )
    valid_data = Pool(data=X_valid, 
                      label=y_valid
                      #cat_features=cat_features
                      )
    
    model = CatBoostClassifier(**params)
    model.fit(train_data,
              eval_set=valid_data, 
              use_best_model=True,plot=True
             )
    
    score = model.get_best_score()['validation_0']['AUC']
    scores.append(score)

    y_pred = model.predict_proba(Test)[:, 1]
    prediction += y_pred

prediction /= n_fold
print('CV mean: {:.4f}, CV std: {:.4f}'.format(np.mean(scores), np.std(scores)))
'''

''' k-fold Cross Vaidation '''
'''
from sklearn.model_selection import KFold

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = KFold(n_splits=10)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/10')
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores}
    return results

def runLR(train_X, train_y, test_X, test_y, test_X2, params):
    print('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2


lr_params = {'solver': 'lbfgs', 'C':  0.1}
results = run_cv_model(n_train, Test, trainlabel, runLR, lr_params, auc, 'lr')
'''

                                ''' Submission'''

submission = pd.DataFrame(
    {'id': Id_test, 'target': prediction,})
submission.to_csv('submission6.csv', index=False)
from sklearn.metrics import roc_auc_score,accuracy_score
y_pred=cat.predict(X_n_test)
ac=accuracy_score(y_n_test,y_pred)
print(ac)
print(roc_auc_score(y_n_test,y_pred,average="macro"))



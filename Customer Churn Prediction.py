import numpy as np
import pandas as pd
from datetime import datetime
import time
from datetime import date
import datetime as dt
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn import tree
from numpy import loadtxt


######################################################
# Jupyter view adjustmets
from IPython.core.display import display, HTML #Increase the code window width
display(HTML("<style>.container { width:100% !important; }</style>")) #Increase the code window width
pd.set_option('expand_frame_repr', False) #keep columns inline withouth wrapping to the next line
######################################################

history = pd.read_csv('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/history.csv')

# Make customerID an index column
history.set_index('customerID', inplace=True)

# Data Overview
print ('***************** DATA OVERVIEW *****************')

print('\n \nChurn values count')
churn_count = (history['Churn'].value_counts()/history['Churn'].count()*100)
churn_count.plot.bar()
plt.title('How many customers churn?')
plt.xlabel('No - Active, Yes - Churn')
plt.ylabel('%')
plt.savefig('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/churn_counts.png')
plt.show()
print (churn_count)


print ("\nDataset dimensions")
print ("Rows     : " ,history.shape[0])
print ("Columns  : " ,history.shape[1])
#print ("\n Features : \n" ,history.columns.tolist())
print ("\nCount missing values : ", history.isnull().sum().values.sum())
print ("\nUnique values : ", history.nunique())
print('\nHistory info : ')
print(history.info())
#print(history.head())


#Replacing spaces with null values in total charges column
history['TotalCharges'] = history["TotalCharges"].replace(" ", 0)
# Convert to float type
history["TotalCharges"] = history["TotalCharges"].astype(float)


charges_avg = history['TotalCharges'].mean()
tenure_avg = history['tenure'].mean()
monthly_charges_avg = charges_avg / tenure_avg
treatment_cost = monthly_charges_avg*12*0.5
print('\nAverage Revenue per Customer to Date: ', '${:,.2f}'.format(charges_avg))
print('\nAverage Customer Tenure: ', '{:,.2f}'.format(tenure_avg), ' months')
print('\nAverage Monthly Revenue per Customer: ', '${:,.2f}'.format(monthly_charges_avg))
print('\nSugested cost of Treatment per Churning Customer: ', '${:,.2f}'.format(treatment_cost))


"""
# Review unique values of some variables
print('\n Gender Values')
print(pd.unique(history['gender'].values))
print('\n Senior Citizen Values')
print(pd.unique(history['SeniorCitizen'].values))
"""
# Convert Yes/No into 1/0
history['Churn'] = history['Churn'].map(lambda x: 1 if x =='Yes' else 0)
history['PaperlessBilling'] = history['PaperlessBilling'].map(lambda x: 1 if x =='Yes' else 0)
history['PhoneService'] = history['PhoneService'].map(lambda x: 1 if x =='Yes' else 0)
history['Dependents'] = history['Dependents'].map(lambda x: 1 if x =='Yes' else 0)
history['Partner'] = history['Partner'].map(lambda x: 1 if x =='Yes' else 0)
history['gender'] = history['gender'].map(lambda x: 1 if x =='Male' else 0)


# Convert categorical to dummy values
history = pd.get_dummies(data = history, columns =
                       ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
                      , prefix=['MultLns_', 'IntServ_', 'OnlineSec_', 'OnlineBckp_', 'DevProtect_','TSupp_',
                                'StrTV_', 'StrMov_', 'Contract_', 'PmnttMeth_'])


# Show columns with any NaNs
#history_copy = history.loc[:, history.isnull().any()]
#print("\nhistory_copy - expected empty :")
#print(history_copy)

print('\n \nHistorical data shape after preprocessing', history.shape)

print("\nCorrelation between features and target")
corr=np.array(history.corr())
# After feature manipulaiton 'Churn' is feature #9
corr=np.around(corr[9],decimals=2)
corr=pd.DataFrame(corr,index=history.columns)
corr_sorted = corr.sort_values(by=0)
print(corr_sorted)


print ('\n \n***************** Train and Test datasets *****************')
# Creating feature and target arrays
y = history['Churn'].values
history_copy = history.drop('Churn', 1)
X = history_copy.values
#print("\n y")
#print(y)
#print("\n X")
#print(X)

# Convert to numpy arrays
y = np.array(y)
X = np.array(X)


#np.seterr(divide='ignore', invalid='ignore')


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Print shapes
print("\n TRAIN AND TEST SHAPES :")
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

############################ Predict with Various Models ############################
### Linear Regression 
# Create the regressor
lnr_reg = LinearRegression()
# Fit the regressor to the training data
lnr_reg.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred_LnrReg = lnr_reg.predict(X_test)
y_pred_LnrReg_Rnd = [round(value) for value in y_pred_LnrReg]

### Logistics Regression 
# Create the regressor:
log_reg = LogisticRegression(solver='lbfgs')
# Fit the regressor to the training data
log_reg.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred_LogReg = log_reg.predict(X_test)
y_pred_LogReg_Rnd = [round(value) for value in y_pred_LogReg]

### Naive Bayes 
# Create the regressor:
nb = GaussianNB()
# Fit the regressor to the training data
nb.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred_NB = nb.predict(X_test)
y_pred_NB_Rnd = [round(value) for value in y_pred_NB]

### Random Forest 
# Train random forest
rf = RandomForestClassifier(n_estimators= 200, random_state=42)
# Fit the regressor to the training data
rf.fit(X_train, y_train)
# Make RF predictions for test data
y_pred = rf.predict(X_test)
y_pred_Rnd = [round(value) for value in y_pred]


# Print confusion matrices
print('\n \nConfusion Matrix: Linear Regression')
print(confusion_matrix(y_test, y_pred_LnrReg_Rnd, labels=[1, 0]))
print('\nClassification Report: Linear Regression')
print(classification_report(y_test, y_pred_LnrReg_Rnd))
#extracting TP, FN, FP, TN
tn_LnrReg_Rnd, fp_LnrReg_Rnd, fn_LnrReg_Rnd, tp_LnrReg_Rnd = confusion_matrix(y_test, y_pred_LnrReg_Rnd).ravel()
#print("True Positives: ",tp_LnrReg_Rnd)
#print("True Negatives: ",tn_LnrReg_Rnd)
#print("False Positives: ",fp_LnrReg_Rnd)
#print("False Negatives: ",fn_LnrReg_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_LnrReg_Rnd/(tp_LnrReg_Rnd+fn_LnrReg_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_LnrReg_Rnd/(tp_LnrReg_Rnd+fp_LnrReg_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_LnrReg_Rnd)))
revenue_retained_LnrReg_Rnd = tp_LnrReg_Rnd*(charges_avg-treatment_cost) - fp_LnrReg_Rnd*treatment_cost
potential_loss = (fn_LnrReg_Rnd+tp_LnrReg_Rnd)*(-1*charges_avg)
potential_loss_after_LnrReg_Rnd = potential_loss + revenue_retained_LnrReg_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling:', '${:,.2f}'.format(potential_loss_after_LnrReg_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_LnrReg_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_LnrReg_Rnd*-100/(potential_loss)))



print('\n \nConfusion Matrix: Logistics Regression')
print(confusion_matrix(y_test, y_pred_LogReg_Rnd, labels=[1, 0]))
print('\nClassification Report: Logistics Regression')
print(classification_report(y_test, y_pred_LogReg_Rnd))
#extracting TP, FN, FP, TN
tn_LogReg_Rnd, fp_LogReg_Rnd, fn_LogReg_Rnd, tp_LogReg_Rnd = confusion_matrix(y_test, y_pred_LogReg_Rnd).ravel()
#print("True Positives: ",tp_LogReg_Rnd)
#print("True Negatives: ",tn_LogReg_Rnd)
#print("False Positives: ",fp_LogReg_Rnd)
#print("False Negatives: ",fn_LogReg_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_LogReg_Rnd/(tp_LogReg_Rnd+fn_LogReg_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_LogReg_Rnd/(tp_LogReg_Rnd+fp_LogReg_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_LogReg_Rnd)))
revenue_retained_LogReg_Rnd = tp_LogReg_Rnd*(charges_avg-treatment_cost) - fp_LogReg_Rnd*treatment_cost
potential_loss_after_LogReg_Rnd = potential_loss + revenue_retained_LogReg_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling:', '${:,.2f}'.format(potential_loss_after_LogReg_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_LogReg_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_LogReg_Rnd*-100/(potential_loss)))


print('\n \nConfusion Matrix: Naive Bayes')
print(confusion_matrix(y_test, y_pred_NB_Rnd, labels=[1, 0]))
print('\nClassification Report: Naive Bayes')
print(classification_report(y_test, y_pred_NB_Rnd))
tn_NB_Rnd, fp_NB_Rnd, fn_NB_Rnd, tp_NB_Rnd = confusion_matrix(y_test, y_pred_NB_Rnd).ravel()
#print("True Positives: ",tp_NB_Rnd)
#print("True Negatives: ",tn_NB_Rnd)
#print("False Positives: ",fp_NB_Rnd)
#print("False Negatives: ",fn_NB_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_NB_Rnd/(tp_NB_Rnd+fn_NB_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_NB_Rnd/(tp_NB_Rnd+fp_NB_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_NB_Rnd)))
revenue_retained_NB_Rnd = tp_NB_Rnd*(charges_avg-treatment_cost) - fp_NB_Rnd*treatment_cost
potential_loss_after_NB_Rnd = potential_loss + revenue_retained_NB_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling:', '${:,.2f}'.format(potential_loss_after_NB_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_NB_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_NB_Rnd*-100/(potential_loss)))


print('\n \nConfusion Matrix: Random Forest')
print(confusion_matrix(y_test, y_pred_Rnd, labels=[1, 0]))
print('\nClassification Report: Random Forest')
print(classification_report(y_test, y_pred_Rnd))
#extracting TP, FN, FP, TN
tn_Rnd, fp_Rnd, fn_Rnd, tp_Rnd = confusion_matrix(y_test, y_pred_Rnd).ravel()
#print("True Positives: ",tp_Rnd)
#print("True Negatives: ",tn_Rnd)
#print("False Positives: ",fp_Rnd)
#print("False Negatives: ",fn_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_Rnd/(tp_Rnd+fn_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_Rnd/(tp_Rnd+fp_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_Rnd)))
revenue_retained_Rnd = tp_Rnd*(charges_avg-treatment_cost) - fp_Rnd*treatment_cost
potential_loss_after_Rnd = potential_loss + revenue_retained_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling:', '${:,.2f}'.format(potential_loss_after_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_Rnd*-100/(potential_loss)))


############################ Tuning Random Forest ############################
print ('\n \n \n***************** Tuning Random Forest *****************')
n_estimators_options = [2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Recall_t = []
NPV_t = []
Accuracy_t = []
for estimator in n_estimators_options:
    rf_tuned = RandomForestRegressor(n_estimators=estimator, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    y_pred_tuned = rf_tuned.predict(X_test)
    y_pred_tuned_Rnd = [round(value) for value in y_pred_tuned]           
    #extracting TP, FN, FP, TN
    tn_Rnd, fp_Rnd, fn_Rnd, tp_Rnd = confusion_matrix(y_test, y_pred_tuned_Rnd).ravel()
    Recall_ti = tp_Rnd/(tp_Rnd+fn_Rnd)
    NPV_ti = tn_Rnd/(tn_Rnd+fn_Rnd)
    Accuracy_ti = accuracy_score(y_test, y_pred_tuned_Rnd)
    Recall_t.append(Recall_ti)
    NPV_t.append(NPV_ti)
    Accuracy_t.append(Accuracy_ti)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators_options, Recall_t, 'blue', label='Recall')
line2, = plt.plot(n_estimators_options, NPV_t, 'red', label='Negative Predictive Value')
line3, = plt.plot(n_estimators_options, Accuracy_t, 'green', label='Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Rate')
plt.xlabel('N_estimators')
plt.title('How tuning the number of estimators affects the model?')
plt.savefig('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/RF_n_estimators_tuning_.png')
plt.show()


sample_leaf_options = [1,5,10,50,100,200,500]
Recall_t = []
NPV_t = []
Accuracy_t = []
for leaf_size in sample_leaf_options:
    rf_tuned = RandomForestRegressor(n_estimators = 100, n_jobs=-1,  min_samples_leaf = leaf_size)
    rf_tuned.fit(X_train, y_train)
    y_pred_tuned = rf_tuned.predict(X_test)
    y_pred_tuned_Rnd = [round(value) for value in y_pred_tuned]           
    #extracting TP, FN, FP, TN
    tn_Rnd, fp_Rnd, fn_Rnd, tp_Rnd = confusion_matrix(y_test, y_pred_tuned_Rnd).ravel()
    Recall_ti = tp_Rnd/(tp_Rnd+fn_Rnd)
    NPV_ti = tn_Rnd/(tn_Rnd+fn_Rnd)
    Accuracy_ti = accuracy_score(y_test, y_pred_tuned_Rnd)
    Recall_t.append(Recall_ti)
    NPV_t.append(NPV_ti)
    Accuracy_t.append(Accuracy_ti)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(sample_leaf_options, Recall_t, 'blue', label='Recall')
line2, = plt.plot(sample_leaf_options, NPV_t, 'red', label='Negative Predictive Value')
line3, = plt.plot(sample_leaf_options, Accuracy_t, 'green', label='Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Rate')
plt.xlabel('Min number of samples')
plt.title('How tuning the min number of samples affects the model?')
plt.savefig('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/RF_sample_leaf_tuning_.png')
print('\n')
plt.show()


max_depth_options = np.linspace(1, 32, 32, endpoint=True)
Recall_t = []
NPV_t = []
Accuracy_t = []
for max_depth in max_depth_options:
    rf_tuned = RandomForestRegressor(n_estimators = 100, n_jobs=-1, max_depth=max_depth)
    rf_tuned.fit(X_train, y_train)
    y_pred_tuned = rf_tuned.predict(X_test)
    y_pred_tuned_Rnd = [round(value) for value in y_pred_tuned]           
    #extracting TP, FN, FP, TN
    tn_Rnd, fp_Rnd, fn_Rnd, tp_Rnd = confusion_matrix(y_test, y_pred_tuned_Rnd).ravel()
    Recall_ti = tp_Rnd/(tp_Rnd+fn_Rnd)
    NPV_ti = tn_Rnd/(tn_Rnd+fn_Rnd)
    Accuracy_ti = accuracy_score(y_test, y_pred_tuned_Rnd)
    Recall_t.append(Recall_ti)
    NPV_t.append(NPV_ti)
    Accuracy_t.append(Accuracy_ti)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depth_options, Recall_t, 'blue', label='Recall')
line2, = plt.plot(max_depth_options, NPV_t, 'red', label='Negative Predictive Value')
line3, = plt.plot(max_depth_options, Accuracy_t, 'green', label='Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Rate')
plt.xlabel('Max Depth Options')
plt.title('How tuning the max depth affects the model?')
plt.savefig('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/RF_max_depth_tuning_.png')
print('\n')
plt.show()


max_features_options = list(range(1,history_copy.shape[1]))
Recall_t = []
NPV_t = []
Accuracy_t = []
for max_feature in max_features_options:
    rf_tuned = RandomForestRegressor(n_estimators = 100, n_jobs=-1, max_features=max_feature)
    rf_tuned.fit(X_train, y_train)
    y_pred_tuned = rf_tuned.predict(X_test)
    y_pred_tuned_Rnd = [round(value) for value in y_pred_tuned]           
    #extracting TP, FN, FP, TN
    tn_Rnd, fp_Rnd, fn_Rnd, tp_Rnd = confusion_matrix(y_test, y_pred_tuned_Rnd).ravel()
    Recall_ti = tp_Rnd/(tp_Rnd+fn_Rnd)
    NPV_ti = tn_Rnd/(tn_Rnd+fn_Rnd)
    Accuracy_ti = accuracy_score(y_test, y_pred_tuned_Rnd)
    Recall_t.append(Recall_ti)
    NPV_t.append(NPV_ti)
    Accuracy_t.append(Accuracy_ti)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features_options, Recall_t, 'blue', label='Recall')
line2, = plt.plot(max_features_options, NPV_t, 'red', label='Negative Predictive Value')
line3, = plt.plot(max_features_options, Accuracy_t, 'green', label='Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Rate')
plt.xlabel('Max Features')
plt.title('How tuning the max features affects the model?')
plt.savefig('/Users/HF/Documents/Education/Springboard/0_Capstone_1/Telco/RF_max_features_tuning_.png')
print('\n')
plt.show()



############################ Up- and Down- sampling ############################
print ('\n \n \n***************** Up- and Down- sampling Random Forrest *****************')

print('X_train.shape, X_test.shape, y_train.shape, y_test.shape')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Re-create the train dataset including the target value
history_ud = np.column_stack((X_train, y_train))

# pass in array and columns
history_ud_df = pd.DataFrame(history_ud)

# Review number of class values in the dataset
print('\n \n \nCHURN VALUES COUNT')
print(history_ud_df[40].value_counts())


# Separate majority and minority classes
hist_maj = history_ud_df[history_ud_df[40] ==0]
hist_min = history_ud_df[history_ud_df[40] ==1]

############ Up-sampling ########################################################################
# Upsample minority class
hist_min_ups = resample(hist_min, 
                                 replace=True,     # sample with replacement
                                 n_samples=3635,    # to match majority class
                                 random_state=123) # reproducible results

# Combine majority class with upsampled minority class
hist_ups = pd.concat([hist_maj, hist_min_ups])
 
# Display new class counts
print('\nUpsampled dataset value counts')
print(hist_ups[40].value_counts())

## Random Forest - Upsampled Dataset
# Creating feature and target arrays
y_u = hist_ups[40].values
hist_ups = hist_ups.drop(40, 1)
X_u = hist_ups.values

# Convert to numpy arrays
y_u = np.array(y_u)
X_u = np.array(X_u)

# Create training and test sets
X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_u, y_u, test_size = 0.3, random_state = 42)

# Define model parameters
rf_u = RandomForestRegressor(n_estimators = 100, random_state=42)
# Train the model on the upsampled data
rf_u.fit(X_train_u, y_train_u)
# Make predictions for test data
y_pred_u = rf_u.predict(X_test)
y_pred_u_Rnd = [round(value) for value in y_pred_u]

print('\n \nConfusion Matrix: Random Forest Upsampled applied to test data')
print(confusion_matrix(y_test, y_pred_u_Rnd, labels=[1, 0]))
print('\nClassification Report: Random Forest')
print(classification_report(y_test, y_pred_u_Rnd))
#extracting TP, FN, FP, TN
tn_u_Rnd, fp_u_Rnd, fn_u_Rnd, tp_u_Rnd = confusion_matrix(y_test, y_pred_u_Rnd).ravel()
#print("True Positives: ",tp_u_Rnd)
#print("True Negatives: ",tn_u_Rnd)
#print("False Positives: ",fp_u_Rnd)
#print("False Negatives: ",fn_u_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_u_Rnd/(tp_u_Rnd+fn_u_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_u_Rnd/(tp_u_Rnd+fp_u_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_u_Rnd)))
revenue_retained_u_Rnd = tp_u_Rnd*(charges_avg-treatment_cost) - fp_u_Rnd*treatment_cost
potential_loss_after_u_Rnd = potential_loss + revenue_retained_u_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling: ', '${:,.2f}'.format(potential_loss_after_u_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_u_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_u_Rnd*-100/(potential_loss)))



############ Down-sampling ########################################################################

# Downsample majority class
hist_maj_dwn = resample(hist_maj, 
                                 replace=False,     # without replacement
                                 n_samples=1295,    # to match majority class
                                 random_state=123) # reproducible results

# Combine downsampled majority class with minority class
hist_dwn = pd.concat([hist_maj_dwn, hist_min])
 
# Display new class counts
print('\nDownsampled dataset value counts')
print(hist_dwn[40].value_counts())

## Random Forest - Upsampled Dataset
# Creating feature and target arrays
y_d = hist_dwn[40].values
hist_dwn = hist_dwn.drop(40, 1)
X_d = hist_dwn.values

# Convert to numpy arrays
y_d = np.array(y_d)
X_d = np.array(X_d)

# Create training and test sets
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size = 0.3, random_state = 42)

# Define model parameters
rf_d = RandomForestRegressor(n_estimators = 100, random_state=42)
# Train the model on the downsampled data
rf_d.fit(X_train_d, y_train_d)
# Make RF predictions for test data
y_pred_d = rf_d.predict(X_test)
y_pred_d_Rnd = [round(value) for value in y_pred_d]

print('\n \n Confusion Matrix: Random Forest Downsampled applied to test data')
print(confusion_matrix(y_test, y_pred_d_Rnd, labels=[1, 0]))
print('\n Classification Report: Random Forest')
print(classification_report(y_test, y_pred_d_Rnd))
#extracting TP, FN, FP, TN
tn_d_Rnd, fp_d_Rnd, fn_d_Rnd, tp_d_Rnd = confusion_matrix(y_test, y_pred_d_Rnd).ravel()
#print("True Positives: ",tp_d_Rnd)
#print("True Negatives: ",tn_d_Rnd)
#print("False Positives: ",fp_d_Rnd)
#print("False Negatives: ",fn_d_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_d_Rnd/(tp_d_Rnd+fn_d_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_d_Rnd/(tp_d_Rnd+fp_d_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, y_pred_d_Rnd)))
revenue_retained_d_Rnd = tp_d_Rnd*(charges_avg-treatment_cost) - fp_d_Rnd*treatment_cost
potential_loss_after_d_Rnd = potential_loss + revenue_retained_d_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling: ', '${:,.2f}'.format(potential_loss_after_d_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_d_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_d_Rnd*-100/(potential_loss)))



############################ Model Stacking ############################
print('\n\n***************** Model Stacking *****************')
# Using the existing train and test datasets
# Spilt the train dataset into train and validation datasets
print('\nTest dataset shape: ', X_test.shape, y_test.shape)
print('\nTrain dataset shape: ', X_train.shape, y_train.shape)


# Create 1 and 2 level datasets from the train dataset
X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, test_size = 0.5, random_state = 42)
print('\nX_1 dataset shape: ', X_1.shape, y_1.shape)
print('\nX_2 dataset shape: ', X_2.shape, y_2.shape,)

# Build and train models on level 1 dataset
model_1 = RandomForestRegressor(n_estimators= 75, random_state=42)
model_1.fit(X_1, y_1)

model_2 = RandomForestRegressor(n_estimators= 100, random_state=42)
model_2.fit(X_1, y_1)

model_3 = RandomForestRegressor(n_estimators= 125, random_state=42)
model_3.fit(X_1, y_1)

model_4 = RandomForestRegressor(n_estimators= 50, random_state=42)
model_4.fit(X_1, y_1)

model_5 = GaussianNB()
model_5.fit(X_1, y_1)


# Predict the first level learners on the level 2 dataset
model_1_pred = model_1.predict(X_2)
model_2_pred = model_2.predict(X_2)
model_3_pred = model_3.predict(X_2)
model_4_pred = model_4.predict(X_2)
model_5_pred = model_5.predict(X_2)


# Concatenate the 1st level classifiers output to form final training data
X_2_meta = np.concatenate((X_2, np.vstack((model_1_pred, model_2_pred, model_3_pred, model_4_pred, model_5_pred)).T), axis=1)
# Tried to use the meta dataset without the initial features (below) and the accuracy is lower
# Therefor useing the initial features plus the predicted values for the 2nd level prediction
#X_2_meta = np.column_stack((model_1_pred, model_2_pred, model_3_pred, model_4_pred, model_5_pred))
print("\nX_2 meta dataset shape: ", X_2_meta.shape, y_2.shape)


# Define the 2nd level classifier on the newly born meta data set
model_level_2 = LogisticRegression(solver='lbfgs', max_iter = 10000)
# Fit the 2nd level classifier to the newly born meta data set
model_level_2.fit(X_2_meta, y_2)


# Apply the models to the test dataset
model_1_pred = model_1.predict(X_test)
model_2_pred = model_2.predict(X_test)
model_3_pred = model_3.predict(X_test)
model_4_pred = model_4.predict(X_test)
model_5_pred = model_5.predict(X_test)

# Concatenate the classifiers output to form final test data
X_test_meta = np.concatenate((X_test, np.vstack((model_1_pred, model_2_pred, model_3_pred, model_4_pred, model_5_pred)).T), axis=1)

# Fit the 2nd level classifier to the newly born test meta data set
model_level_2_pred = model_level_2.predict(X_test_meta)


print('\n \n Confusion Matrix: Stacked Model applied to test data')
print(confusion_matrix(y_test, model_level_2_pred, labels=[1, 0]))
#print("\nAccuracy Score : ", model_level_2.score(X_test_meta, y_test))
print('\n Classification Report: Stacked Model applied to test data')
print(classification_report(y_test, model_level_2_pred))
tn_s_Rnd, fp_s_Rnd, fn_s_Rnd, tp_s_Rnd = confusion_matrix(y_test, model_level_2_pred).ravel()
#print("True Positives: ",tp_d_Rnd)
#print("True Negatives: ",tn_d_Rnd)
#print("False Positives: ",fp_d_Rnd)
#print("False Negatives: ",fn_d_Rnd)
print('Recall (Churn): ', '{:,.2f}'.format(tp_s_Rnd/(tp_s_Rnd+fn_s_Rnd)))
print('Precision (Churn): ', '{:,.2f}'.format(tp_s_Rnd/(tp_s_Rnd+fp_s_Rnd)))
print("Accuracy Score : ", '{:,.2f}'.format(accuracy_score(y_test, model_level_2_pred)))
revenue_retained_s_Rnd = tp_s_Rnd*(charges_avg-treatment_cost) - fp_s_Rnd*treatment_cost
potential_loss_after_s_Rnd = potential_loss + revenue_retained_s_Rnd
print('Forgone revenue before churn modelling: ', '${:,.2f}'.format(potential_loss))
print('Forgone revenue after churn modelling: ', '${:,.2f}'.format(potential_loss_after_s_Rnd))
print('Revenue retained due to churn modelling: ', '${:,.2f}'.format(revenue_retained_s_Rnd))
print('Revenue retained due to churn modelling, %: ', '{:,.0f}%'.format(revenue_retained_s_Rnd*-100/(potential_loss)))

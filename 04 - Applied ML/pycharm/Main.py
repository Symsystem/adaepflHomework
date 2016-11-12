import pandas as pd                                     # for dealing with csv import
import numpy as np                                      # arrays and other matlab like manipulation
import matplotlib.pyplot as plt                         # Matplotlib's pyplot: MATLAB-like syntax
from sklearn.ensemble import RandomForestClassifier
import scipy # for the randomclassifier

from sklearn.grid_search import GridSearchCV
from numpy._distributor_init import NUMPY_MKL


# import the datas
df = pd.read_csv('../CrowdstormingDataJuly1st.csv')# 14200 28

# clean the datas where there is a nan
df.dropna(axis=0,inplace=True)

# define the parameters we want to keep "only integers ? ":
parameters = ['height', 'weight', 'games','victories','defeats','goals','meanIAT','nIAT','seIAT','meanExp','nExp','seExp']

# color average :
df['color_average'] = (df['rater2'] + df['rater1'])/2

# datas
X_train = np.asarray(df[parameters], dtype="|S6")
Y_train = np.asarray(df['color_average'], dtype="|S6")


## run the test
clf = RandomForestClassifier(n_estimators=10,
criterion='gini', max_depth=None,
min_samples_split=2, min_samples_leaf=1,
max_features='auto',
bootstrap=True, oob_score=False)

clf = clf.fit(X_train,Y_train)
# Error on the train set
goodPrediction =clf.score(X_train,Y_train)
print(goodPrediction)



# error rate for the train
error_rate = []
# # perform the test
# max_estimators = 30
# for i in range(10, max_estimators):
#     clf = RandomForestClassifier(n_estimators=i,
# criterion='gini', max_depth=None,
# min_samples_split=2, min_samples_leaf=1,
# max_features='auto',
# bootstrap=True, oob_score=False)
#     clf = clf.fit(X_train,Y_train)
# # Error on the train set
#     oob_error =1-clf.score(X_train,Y_train)
#     error_rate.append((i, oob_error))
#
#
# # Generate the "train error" vs. "n_estimators" plot.
# plt.plot(error_rate, range(10, max_estimators))
#
# plt.xlim(0, max_estimators)
# plt.xlabel("n_estimators")
# plt.ylabel("train error ")
# plt.legend(loc="upper right")
# plt.show()



# # cross validation ?
#
# param=[{"max_features":list(range(4,64,4))}]
# digit_rf= GridSearchCV(RandomForestClassifier(
# n_estimators=100),param,cv=5,n_jobs=-1)
# digit_rf=digit_rf.fit(X_train, Y_train)
# # paramètre optimal
# print(digit_rf.best_params_)




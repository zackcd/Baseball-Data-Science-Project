import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

"""
Loading in the csv file as a Pandas dataframe
"""
df = pd.read_csv('Data_Project_Data.csv')
seed = 1

"""
Gathering info on the dataset and determining where missing values need to be dealt with
"""
print(df.head())
print(df.info())
print(df.describe())
pitch_types = df.pitch_type.unique()
print(pitch_types)

print()
print("NaN counts:")
print(df.isna().sum())

pitch_mapping = {'Fastball': 0, 'Slider': 1, 'Curveball': 2, 'Changeup': 3}
inv_pitch_mapping = {v: k for v, k in pitch_mapping.items()}

print(pitch_mapping)

"""
For each of break_z, break_x, pitch_initial_speed, since we trust system B more, we are starting with the
values measured by system B and only filling in the missing values with those measured by system A
"""
break_z = df['break_z_b'].fillna(df['break_z_a'])
break_x = df['break_x_b'].fillna(df['break_x_a'])
pitch_initial_speed = df['pitch_initial_speed_b'].fillna(df['pitch_initial_speed_a'])

df['break_x'] = break_x
df['break_z'] = break_z
df['pitch_initial_speed'] = pitch_initial_speed
df = df.rename({'spinrate_b': 'spinrate'}, axis = 1)

"""
Transform pitcher_side into a binary feature where 1 means the pitcher is right handed, 0 mean left handed
"""
df['pitcher_side'] = (df['pitcher_side'] == 'R').astype(int)

"""
Map pitch types from strings to values 0, 1, 2, 3 using the dictionary pitch_mapping
"""
df['pitch_type'] = df['pitch_type'].map(pitch_mapping)

"""
Features that will ultimately be used for prediction, as well as the prediction target pitch_type.
I originally considering methods of imputing spinrate since it has a large number of missing values.
I considered predicting spinrate and using that in the training dataset; however, I believe there are
too many misisng values to make this a viable option and I chose to omit spinrate as a feature.
I first omited pitcher_id but found that the model performed better with it included. I belive this is because
a pitch of a particular type by a certain pitcher will be consistently similar.
"""
features = ['pitcher_id', 'pitcher_side', 'pitch_initial_speed', 'break_x', 'break_z']
#features = ['pitcher_id', 'pitcher_side', 'pitch_initial_speed', 'break_z']
target = 'pitch_type'

data = df[[target] + features]

# colors = ("red", "green", "blue", "black")
# groups = (0, 1, 2, 3)
# gb = data.groupby('pitch_type')
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# for color, group in zip(colors, groups):
#     x = gb.get_group(group)['break_x']
#     y = gb.get_group(group)['break_z']
#     ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', label=group)

# plt.title('Scatter plot of pitch break_x vs break_z, colored by pitch type')
# plt.xlabel('break_x')
# plt.ylabel('break_z')
# plt.legend()
# #plt.show()

# colormap = plt.cm.RdBu
# plt.title('Pearson Correlation of Features')
# sns.heatmap(data.astype(float).corr(), cmap=colormap, linecolor='white', annot=True)
# #plt.show()

"""
Separate train and test data. Train data is the labeled data while test data in the unlabeled data, and as such has
a missing value in the pitch_type feature
"""
train = data[data['pitch_type'].isnull() == False]
test = data[data['pitch_type'].isnull()]

"""
Imputing missing training training values based on the average of training data with the same label as the missing data
"""
train.pitch_initial_speed = train['pitch_initial_speed'].fillna(train.groupby('pitch_type')['pitch_initial_speed'].transform('mean'))
train.break_x = train['break_x'].fillna(train.groupby('pitch_type')['break_x'].transform('mean'))
train.break_z = train['break_z'].fillna(train.groupby('pitch_type')['break_z'].transform('mean'))
#train = train.dropna(axis = 0)

"""
In the case of missing test data, I found it is most accurate by just using the average of the entire feature in the training set
"""
test.pitch_initial_speed = test['pitch_initial_speed'].fillna(train['pitch_initial_speed'].mean())
test.break_x = test['break_x'].fillna(train['break_x'].mean())
test.break_z = test['break_z'].fillna(train['break_z'].mean())

X_train = train[features]
y_train = train[target]

X_test = test[features]

#test_portion = 0.3
# X_train, X_test, y_train, y_test = train_test_split(train[features], train[target], test_size=test_portion, random_state=seed, stratify = train[target])

"""
Normalizing (not standardizing) because each feature has a very different scale
"""
sc = MinMaxScaler()
sc.fit(X_train)
X_train_norm = sc.transform(X_train)
X_test_norm = sc.transform(X_test)

"""
First model I considered in KNN. The data likely cannot be made to be linearly separable, and many pitches of the same type are alike, so this seems like
a strong application for this supervised learning algorithm.
"""

"""
We will be using K-fold cross-validation for hyperparameter tuning in order to find our final model.
Note that when p=2 and metric is set to minkowski, this is using L2 norm distance measure.
"""
print("KNN:")
# for numNeighbors in range(1, 10):
#     # minkowski metrix with p=2 is euclidean distance
#     knn = KNeighborsClassifier(n_neighbors=numNeighbors, p=2, metric='minkowski')
#     knn.fit(X_train, y_train)

#     pred = knn.predict(X_test)
#     accuracy = accuracy_score(y_test, pred)
#     print(accuracy)

neighbors = [x for x in range(1, 51)]
knn_accuracy = []
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    scores = cross_val_score(knn, X_train_norm, y_train, cv=10, scoring='accuracy')
    knn_accuracy.append(scores.mean())

optimal_k_idx = knn_accuracy.index(max(knn_accuracy))
optimal_k = neighbors[optimal_k_idx]
print('Best k value:', optimal_k, '\n')
print('Best accuracy:', knn_accuracy[optimal_k_idx], '\n')

fig = plt.figure()
plt.plot(neighbors, knn_accuracy)
plt.title('Accuracy vs Number of Neighbors k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

"""
I next considered a random forest. The features of this problem are quite simple, but this model is much more complex than KNN, so it may yield more accurate
predictions.
"""
print("Random Forest:")

# numEst = [x for x in range(400, 601, 25)]
# criterion = ['gini', 'entropy']
# cv_scores = []
# best_score = 0
# for k in numEst:
#     print(k)
#     for crit in criterion:
#         forest = RandomForestClassifier(n_estimators=k, random_state=seed, criterion=crit)
#         scores = cross_val_score(forest, X_train_norm, y_train, cv=10, scoring='accuracy')
#         cv_scores.append(scores.mean())
#         if scores.mean() > best_score:
#             best_params = (k, crit)
#             best_score = scores.mean()

# print(best_params)
# print(best_score)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 8, 15, 25, 30, 40],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 10, 15, 50, 75, 100, 140],
    'n_estimators': [200, 400, 600, 800]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 1)

# grid_search.fit(X_train_norm, y_train)
# print(grid_search.best_params_)
"""
Result:
old:
{'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 100, 'n_estimators': 600}
new:
{'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 5, 'min_samples_split': 140, 'n_estimators': 200}
"""

# #opt_forest = RandomForestClassifier(random_state = seed, max_depth = 30, n_estimators = 600, min_samples_split = 100, min_samples_leaf = 1)
# opt_forest = RandomForestClassifier(random_state=seed, max_depth=30, min_samples_leaf=5, min_samples_split=140, n_estimators=200)
# scores = cross_val_score(opt_forest, X_train_norm, y_train, cv=10, scoring='accuracy')
# best_forest_acc = scores.mean()
# print(best_forest_acc)

"""
I also considered SVM and logistic regression, but neither yielded results nearly as accurate as the previous two models, I believe because
the data is relatively inseparable.
"""

# print("Support Vector Machine:")
# params_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# }

# param_range = [0.0001, 0.001, 0.01, 0.1,
# 1.0, 10.0, 100.0, 1000.0]
# params_grid = [{'svc__C': param_range,
#     'svc__kernel': ['linear']}, {'svc__C': param_range,
#     'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

# svm = SVC()

# svm_grid_search = GridSearchCV(estimator = svm, param_grid = params_grid, cv=3, n_jobs = -1, verbose = 1)
# svm_grid_search.fit(X_train_norm, y_train)
# print(svm_grid_search.best_params_)
# """
# {'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 5, 'min_samples_split': 140, 'n_estimators': 200}
# """

# opt_svm = SVC
# scores = cross_val_score(opt_svm, X_train_norm, y_train, cv=10, scoring='accuracy')
# best_svm_acc = scores.mean()
# print(best_svm_acc)



# print('Best score for training data:', svm_model.best_score_,"\n") 
# print('Best C:',svm_model.best_estimator_.C,"\n") 
# print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
# print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")



"""
Finally using out model to predict the values of the test dataset
"""


"""
The result was a very accurate model. Additionally, based on the results of this excercise, I believe that with more
data and potentially a more complex model, this can be extended to acurately predcit even more pitch types, which would
be subtypes of these pitches, such as four-seam fastballs, cutters, etc.
"""
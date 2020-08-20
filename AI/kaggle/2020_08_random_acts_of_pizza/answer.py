import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seab
import json

# to make decision tree
# https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

# for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# open and show plt data
jf = open('train.json', 'r')
json_loaded = json.load(jf)
json_data = pd.DataFrame(json_loaded)

# True to 1, False to 0, and decimal to 0
# https://stackoverflow.com/questions/29960733/how-to-convert-true-false-values-in-dataframe-as-1-for-true-and-0-for-false
tfCols = ['post_was_edited', 'requester_received_pizza']
tfColsIndex = [-1, -1] # column indices of tfCols

# initialize tfColsIndex
for i in range(len(json_data.columns)):
    for j in range(len(tfCols)):
        if json_data.columns[i] == tfCols[j]: tfColsIndex[j] = i

# extract column name before change into np.array
dataCols = np.array(json_data.columns)

# change json_data into np.array
json_data = json_data.to_numpy()

# modify values: True to 1, False to 0, and decimal to 0
for x in range(2):

    # True to 1, False to 0, and decimal to 0
    for i in range(len(json_data)):
        if str(json_data[i][tfColsIndex[x]]) == 'True':
            json_data[i][tfColsIndex[x]] = 1
        else:
            json_data[i][tfColsIndex[x]] = 0

json_data = pd.DataFrame(json_data, columns=dataCols)
print('json_data.shape = ' + str(json_data.shape))

# create data
# .data and .target
targetCol = 'requester_received_pizza' # target column name
textCols = ['giver_username_if_known', 'request_id', 'request_text', 'request_text_edit_aware',
            'request_title', 'request_subreddits_at_request', 'requester_user_flair',
            'requester_username']
print(dataCols)

dataPart = [] # data columns
targetPart = [] # target column

for col in dataCols:
    
    # not in targetCol and all of values are numeric -> dataPart
    if col != targetCol and not col in textCols: dataPart.append(col)

    # in targetCol and not all values are the same (then not meaningful)
    elif col == targetCol and max(json_data[col]) > min(json_data[col]): targetPart.append(col)

print(dataPart)
print(targetPart)

# bind the data and target
dataSet = {'data':json_data[dataPart], 'target':json_data[targetPart]}

# display correlation
# https://seaborn.pydata.org/generated/seaborn.clustermap.html
print('<<< [0] json_data >>>')
json_data = dataSet['data']
print(json_data[:50])

df = json_data.corr()
seab.clustermap(df,
                annot=True,
                cmap='RdYlBu_r',
                vmin=-1, vmax=1)
plt.show()

# PCA
# https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
scaled = StandardScaler().fit_transform(json_data) # to standard normal distribution
pca = PCA(n_components=8)

# show plt data

# get PCA transformed data
pca.fit(scaled)
scaledPCA = pca.transform(scaled)

print('scaledPCA.shape = ' + str(scaledPCA.shape))
print('scaledPCA.data.shape = ' + str(scaledPCA.data.shape))

print('<<< [1] scaledPCA >>>')
print(scaledPCA)

# convert to numpy array
json_data = np.array(json_data)

# name each column for PCA transformed data
pca_cols = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7']
df_pca = pd.DataFrame(scaledPCA.data, columns=pca_cols)
df_pca['target'] = scaledPCA['requester_received_pizza']
df_pca.head(3)

# set markers
markers = ['^', 's', 'o']

# scatter plot
for i, marker in enumerate(markers):
    x_axis_data = df_pca[df_pca['target'] == i]['pca_component_1']
    y_axis_data = df_pca[df_pca['target'] == i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=scaledPCA.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()

result = ''

x = json_data["request_id"]
y = json_data["requester_number_of_posts_on_raop_at_request"]

print(x, y)

for i in range(len(json_data)):
    if y[i] <= 1:
        result += str(x[i]) + ',' + str(0) + '\n'
    else:
        result += str(x[i]) + ',' + str(1) + '\n'

f = open('result.csv', 'w')
f.write(result)
f.close()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("train1.csv")

# #Train describe
# train.describe()
#
# #is there missing?
# train[train.isna()].sum()

#lets figure
# plt.figure(figsize=[12,10])
# plt.subplot(221)
# banner_pos = train.groupby('banner_pos').sum().reset_index()
# sns.barplot('banner_pos', 'click',data = banner_pos)
#
# plt.subplot(222)
# device_type = train.groupby('device_type').sum().reset_index()
# sns.barplot('device_type', 'click', data = device_type)
#
# plt.subplot(223)
# C15 = train.groupby('C15').sum().reset_index()
# sns.barplot('C15', 'click', data = C15)
#
# plt.subplot(224)
# site_category = train.groupby('site_category').sum().reset_index()
# sns.barplot('site_category', 'click', data  = site_category)
# plt.show()

#Relation between attributes
plt.figure(figsize=(14,12))
foo = sns.heatmap(train.corr(), vmax=0.6,square=True, annot=True)
plt.show()

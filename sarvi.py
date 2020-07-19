
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("train1.csv")

#Train describe
train.describe()

#is there missing?
train[train.isna()].sum()

# So really what is the rate of click?
click = train[train['click']==1]
no_click = train[train['click']==0]

print("Click: %i (%.1f percent), Not Click: %i (%.1f percent), Total: %i"\
      %(len(click), 1.*len(click)/len(train)*100.0,\
        len(no_click), 1.*len(no_click)/len(train)*100.0, len(train)))

# In terms of Median
print("Median hour Click: %.1f, Median hour non-Click: %.1f"\
      %(np.median(click['hour'].dropna()), np.median(no_click['hour'].dropna())))

#lets get some barplot figure
plt.figure(figsize=[12,10])
plt.subplot(221)
banner_pos = train.groupby('banner_pos').sum().reset_index()
sns.barplot('banner_pos', 'click',data = banner_pos)

plt.subplot(222)
device_type = train.groupby('device_type').sum().reset_index()
sns.barplot('device_type', 'click', data = device_type)

plt.subplot(223)
C15 = train.groupby('C15').sum().reset_index()
sns.barplot('C15', 'click', data = C15)

plt.subplot(224)
site_category = train.groupby('site_category').sum().reset_index()
sns.barplot('site_category', 'click', data  = site_category)
plt.show()

#Relation between attributes
plt.figure(figsize=(14,12))
foo = sns.heatmap(train.corr(), vmax=0.6,square=True, annot=True)
plt.show()

'''
Seems that 'Unnamed: 0', 'click', 'hour', 'C14', 'C17' are highly correlated to click rate.
In the next section we will plot their inter connections
'''

#Pairplot
warnings.filterwarnings(action="ignore")
cols = ['Unnamed: 0', 'click', 'hour', 'C14', 'C17']
g = sns.pairplot(data = train, vars = cols, size =1.5, hue = 'click')
g.set(xticklabels = [])
plt.show()
plt.tight_layout()

#jointplot of correlated
sns.jointplot("C14", "click", data=train1, kind="reg",
                  xlim=(0, 60), ylim=(0, 5), color="r", size=7)
plt.show()

# I will drop the object(string) object and get the dist plot for the remaining variables.
train1.info()

# Select the string objects :
train1 = train.drop(['Unnamed: 0', 'id', 'site_id','site_domain','site_category',
                     'app_id', 'app_domain', 'app_category', 'device_id',
                     'device_ip', 'device_model'], axis= 1)
col_names = list(train1.columns)
for col in range(len(train1.columns)+1):
    print(col)
    plt.subplot(15,1, int(col)+1)
    sns.distplot(train1.iloc[:, int(col)].values, axlabel= col_names[col])

'''
Unfortunatly the dist plot is not clear. For e.g. the click chart is not clean becasue it is bolean or other features
are somehow have limited levels.
'''

# I am going to check for outliers :






    # here's the parameters of your distribution, scale, location

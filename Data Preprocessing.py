
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## Data Processing

# In[2]:


# read training.csv and test.csv
df = pd.read_csv('./data/training.csv', sep=',')
print(df[:1])
print(df[:1].shape)
print('\n\n')


# In[3]:


df.head()


# In[4]:


# check if the dataframe have some missing facial keypoints 
# if yes replce them with zeros
temp_df = df.drop('Image', axis=1)
# check for null values
print(temp_df.isnull().values.any())
# print the count of null/NaNs in each column of dataframe
print("\n\nMISSING VALUES FOR KEYPOINTS BY NAME:\n")
print(temp_df.isnull().sum())
print("\n\nTOTAL MISSING VALUES IN THE DATASET:\n")
print(temp_df.isnull().sum().sum())
# fill the missing values using forward filling rather than a 0 or some constant 
# value as it is likely that the next keypoint lies closer to previous one on a face
df = df.fillna(method="ffill")
print("Missing values after filling:", df.isnull().sum().sum())


# In[5]:


"""Splitting data into train and validation set 80/20"""
df_shape = df.shape
print(df_shape)
train_split = int(0.8 * df_shape[0])
print(train_split)
train_df = df.iloc[:train_split, :]
val_df = df.iloc[train_split:, :]
print(train_df.shape)
print(val_df.shape)


# ### Prepare training data

# In[6]:


"""Prepare X_train"""
# convert image pixels(str) to numpy(float32) arrays
train_images = train_df['Image']
flat_image_size = len(train_images[0].split())
X_train = np.zeros((train_df.shape[0],flat_image_size), dtype=np.float32)
for index in range(len(train_images)):
    pixels_str = train_images[index].split()
    # Using map function to convert all string values to float32
    pixels = list(map(np.float32, pixels_str))
    X_train[index] = np.asarray(pixels)
print(X_train.shape)


# In[7]:


X_train[0]


# In[8]:


# save X_train as .npy file for fast access
np.save('./processed_data/x_train.npy',X_train)


# In[9]:


"""Prepare Y_train"""
# access the facial keypoints as rows of dataframe, remove the 'Image' column and convert the point to floats
column_names = list(train_df.columns.values)
# remove the name of last column as it holds Image pixel values
column_names = column_names[:len(column_names)-1]

Y_train = np.zeros((train_df.shape[0],len(column_names)), dtype=np.float32)
print(Y_train.shape)
# iterate through the rows of dataframe
for index, row in train_df.iterrows():
    keypoints = []
    for name in column_names:
        keypoints.append(row[name])
    Y_train[index] = np.asarray(keypoints)


# In[10]:


# save Y_train as .npy file for fast access
np.save('./processed_data/y_train.npy',Y_train)


# In[11]:


#rough
lis = train_df.iterrows()
print(type(lis))


# ### Prepare validation data

# In[14]:


val_df.head()['Image']


# In[17]:


"""Prepare X_val"""
# convert image pixels(str) to numpy(float32) arrays
val_images = val_df['Image']
flat_image_size = len(val_images[train_split].split())
X_val = np.zeros((val_df.shape[0],flat_image_size), dtype=np.float32)
index = 0
for image in val_images:
    pixels_str = image.split()
    # Using map function to convert all string values to float32
    pixels = list(map(np.float32, pixels_str))
    X_val[index] = np.asarray(pixels)
    index += 1
print(X_val.shape)


# In[18]:


# save X_val as .npy file for fast access
np.save('./processed_data/x_val.npy',X_val)


# In[30]:


"""Prepare Y_val"""
# access the facial keypoints as rows of dataframe, remove the 'Image' column and convert the point to floats
column_names = list(val_df.columns.values)
# remove the name of last column as it holds Image pixel values
column_names = column_names[:len(column_names)-1]

Y_val = np.zeros((val_df.shape[0],len(column_names)), dtype=np.float32)

# iterate through the rows of dataframe
index = 0
for ind, row in val_df.iterrows():
    keypoints = []
    for name in column_names:
        keypoints.append(row[name])
    Y_val[index] = np.asarray(keypoints)
    index += 1
print(Y_val.shape)


# In[31]:


# save Y_val as .npy file for fast access
np.save('./processed_data/y_val.npy',Y_val)


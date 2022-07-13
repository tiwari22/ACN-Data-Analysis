#!/usr/bin/env python
# coding: utf-8

# In this notebook I have used Adaptive Charging Network data (ACN) provided by California institute of technology to understand the user behaviour of EV owners. The dataset can be found at : https://ev.caltech.edu/dataset
# 
# To prove my coding skills I have tried to code a part of the paper : ACN-Data : Analysis and Applications of an Open EV Charging Dataset which can be found at : https://ev.caltech.edu/assets/pub/ACN_Data_Analysis_and_Applications.pdf
# 
# 

# In[1]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dateutil import tz
from datetime import datetime, timedelta

import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Write a function that takes in a json file and converts it into a pandas DataFrame..
def json_to_DataFrame(file):
    with open(file) as data_file:
        data = json.load(data_file)
        df = pd.DataFrame(data["_items"])
        return df


# In[3]:


caltech_df = json_to_DataFrame(file="acndata_sessions_caltech.json")
jpl_df = json_to_DataFrame(file="acndata_sessions_jpl.json")


# In[4]:


caltech_df.head()


# In[5]:


jpl_df.head()


# In[6]:


# Checking if all the column names are same in both the DataFrames..
caltech_df.columns == jpl_df.columns


# In[7]:


# Now we will compare the number of missing values and number of instances in each of the DataFrames..
df_info_dic = {"caltech_df":([caltech_df.shape[0]]+list(caltech_df.isnull().sum())),
               "jpl_df":([jpl_df.shape[0]]+list(jpl_df.isnull().sum()))}

df_info = pd.DataFrame(df_info_dic, index=["total_instances"]+list(caltech_df.columns))

df_info


# ***Why there are so much missing values in caltech dataframe ?***
# 
# Those drivers who use mobile application to input their Energy Demand and Estimated Departure Time are "claimed" drivers and those drivers who do not use the mobile application are "unclaimed" drivers. For unclaimed drivers Energy demand and estimated departure time are default values.
# 
# Caltech's EV Charger is open to both staff as well as public whereas JPL's EV Charger is open for staff only. It may happen, most of the public drivers are not aware of the mobile application, hence we see alot of missing values in userID column.
# 
# unclaimed drivers are charging their vehicles free of cost. Whereas claimed drivers are charging at $0.12 per KWh.
# 

# In[8]:


# Let us fill in all the missing values in userID column as unclaimed.
caltech_df["userID"] = caltech_df["userID"].fillna("unclaimed")
jpl_df["userID"] = jpl_df["userID"].fillna("unclaimed")


# In[9]:


# Cross verify if all the missing values in userID columns are imputed or not..
caltech_df.userID.isnull().sum(), jpl_df.userID.isnull().sum()


# In[10]:


# Let us replace all other instances in userID column as claimed..

caltech_df.loc[caltech_df["userID"]!="unclaimed", "userID"]="claimed"
jpl_df.loc[jpl_df["userID"]!="unclaimed", "userID"]="claimed"


# In[11]:


caltech_df.head()


# In[12]:


caltech_df.userID.value_counts() # Number of unclaimed should be equal to number of missing values in the column.


# In[13]:


jpl_df.userID.value_counts()


# In[14]:


df_info_dic = {"caltech_df":([caltech_df.shape[0]]+list(caltech_df.isnull().sum())), 
               "jpl_df":([jpl_df.shape[0]]+list(jpl_df.isnull().sum()))}

df_info = pd.DataFrame(df_info_dic, index=["total_instances"]+list(caltech_df.columns))
df_info


# **Check clusterID column of each dataframes**

# In[15]:


caltech_df["clusterID"].value_counts()


# In[16]:


jpl_df["clusterID"].value_counts()


# In[17]:


# Let us change the clusterID of each row for caltech_df to "0039"
caltech_df["clusterID"]="0039"


# In[18]:


print(caltech_df["clusterID"].value_counts())
print(jpl_df["clusterID"].value_counts())


# **Check the siteID of each dataframe**

# In[19]:


print(caltech_df["siteID"].value_counts())
print(jpl_df["siteID"].value_counts())


# In[20]:


# Let us change the siteID of each row of caltech_df to "0002"
caltech_df["siteID"]="0002"


# In[21]:


caltech_df["siteID"].value_counts()


# **Check the spaceID of each DataFrame.**

# In[22]:


print("Number of chargers in Caltech  :  ", len(caltech_df["spaceID"].value_counts()))
print("Number of chargers in JPL      :  ", len(jpl_df["spaceID"].value_counts()))


# **Check the timezone column of each dataframe.**

# In[23]:


print("timezone caltech     :    ",caltech_df["timezone"].value_counts())
print("\ntimezone jpl         :    ",jpl_df["timezone"].value_counts())


# ***Note :*** Though the timezone column has America/Los_Angeles is given as TimeZone. But the connection times and disconnect times are given in GMT. So we have to convert them to America/Los_Angeles timezone.

# **Convert connectionTime, disconnectTime and doneChargingTime columns into a datetime**

# - connectionTime : It is that time when the EV was plugged in.
# - disconnectTime : It is that time when the EV was unplugged.
# - doneChargingTime : It is that time when the last non-zero current draw was recorded. It means it is that time when  the EV's battery became fully charged. So let us fill all the missing values of this column with the corresponding values of disconnectTime column.

# In[24]:


caltech_df["doneChargingTime"].isnull().sum()


# In[25]:


jpl_df["doneChargingTime"].isnull().sum()


# In[26]:


caltech_df["doneChargingTime"] = caltech_df["doneChargingTime"].fillna(caltech_df["disconnectTime"])


# In[27]:


caltech_df["doneChargingTime"].isnull().sum()


# In[28]:


jpl_df["doneChargingTime"] = jpl_df["doneChargingTime"].fillna(jpl_df["disconnectTime"])


# In[29]:


jpl_df["doneChargingTime"].isnull().sum()


# In[30]:


# Now we will convert each instances of connectionTime, disconnectTime and doneChargingTime in datetime objects. Also
# we will convert the timezone from UTC to America/Los_Angeles Timezone.
def convert_datetime(df):
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Los_Angeles')
    
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x : datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"))
    df.iloc[:, 3] = df.iloc[:, 3].apply(lambda x : datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"))
    df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x : datetime.strptime(x, "%a, %d %b %Y %H:%M:%S %Z"))
    
    df.iloc[:,2] = df.iloc[:, 2].apply(lambda x : x.replace(tzinfo=from_zone))
    df.iloc[:,3] = df.iloc[:, 3].apply(lambda x : x.replace(tzinfo=from_zone))
    df.iloc[:,4] = df.iloc[:, 4].apply(lambda x : x.replace(tzinfo=from_zone))
    
    df.iloc[:,2] = df.iloc[:, 2].apply(lambda x : x.astimezone(to_zone))
    df.iloc[:,3] = df.iloc[:, 3].apply(lambda x : x.astimezone(to_zone))
    df.iloc[:,4] = df.iloc[:, 4].apply(lambda x : x.astimezone(to_zone))
    
    return df


# In[31]:


caltech_df = convert_datetime(caltech_df)
jpl_df = convert_datetime(jpl_df)


# In[32]:


caltech_df.head()


# In[33]:


jpl_df.head()


# **Adding session_duration column**

# In[34]:


caltech_df["session_duration"] = (caltech_df["disconnectTime"] - caltech_df["connectionTime"])/timedelta(minutes=1)


# In[35]:


jpl_df["session_duration"] = (jpl_df["disconnectTime"] - jpl_df["connectionTime"])/timedelta(minutes=1)


# In[36]:


caltech_df.head()


# In[37]:


jpl_df.head()


# **Adding a Day column to both the DataFrames that signifies whether the EV was charged on a weekDay or a weekEnd**

# In[38]:


caltech_df["Day"] = caltech_df["connectionTime"].apply(lambda x : x.strftime("%a"))
caltech_df["Day"] = caltech_df["Day"].apply(lambda x : "weekEnd" if (x=="Sun" or x=="Sat") else "weekDay")


# In[39]:


jpl_df["Day"] = jpl_df["connectionTime"].apply(lambda x : x.strftime("%a"))
jpl_df["Day"] = jpl_df["Day"].apply(lambda x : "weekEnd" if (x=="Sun" or x=="Sat") else "weekDay")


# In[40]:


# Let us check the number of vehicles charged on weekDays compared to weekEnds..
caltech_df["Day"].value_counts(normalize=True)


# In[41]:


jpl_df["Day"].value_counts(normalize=True)


# **Note :** 
# - EV Charger installed in JPL is for employees only. This is the reason why only 2.6% vehicles are charging on weekEnds.
# - Whereas the Caltech EV Charger is open for outsiders also hence we can see a significant number of vehicles are charging on weekEnds here.

# **TimeSeries analysis of each DataFrame**

# In[42]:


caltech_ts = caltech_df[["kWhDelivered"]]
caltech_ts.index = caltech_df["connectionTime"]


# In[43]:


caltech_ts.head()


# In[44]:


# Now let us make a function to plot time series plots..
def plot_time_series(df, start=0, end=None, font_size=14, title_font_size=16,label=None, color="b"):
    plt.plot(df[start:end], label=label, c=color)
    plt.title("Time Series Plot", fontsize=title_font_size)
    if label:
        plt.legend(fontsize=font_size)
    plt.xlabel("Time", fontsize=font_size)
    plt.ylabel("Energy demand")
    plt.grid()
    plt.show()


# In[45]:


plt.figure(figsize=(10,7))
plot_time_series(caltech_ts[:50])

# The plot doesn't look like a time series plot. 


# In[46]:


# Let us add a column as connectionDate in each dataframe..
caltech_df["connectionDate"] = caltech_df["connectionTime"].apply(lambda x : x.date)
jpl_df["connectionDate"] = jpl_df["connectionTime"].apply(lambda x : x.date)


# In[47]:


caltech_df.head()


# In[48]:


jpl_df.head()


# Now let a make a function that adds two columns in the dataframe and those are total_energy_consumed and total_sessions per day.
# 
# The following steps should be performed.
# - collect all the instances in connectionDate column in a list that is list1.
# - Remove the duplicate elements and store the unique elements in list2 and sort it so as we get the dates in a proper fashion.
# - Find the indices of these unique elements in list2 from list1.
# - Now count the duplicate values in list1. From this we will find the sessions served each day.
# - With the help of list of indices and duplicate values, we will find the energy demand each day.
# - Then we will make a dictionary of connectionDate, energyDemand and sessions served each day.
# - Convert this dictionary into a DataFrame.
# - Make connectionDate as the index of the DataFrame
# - Finally return the DataFrame.
# 

# In[49]:


# Now let us make a function that calculates total energy consumed and total sessions served on a single day.
def make_correct_time_series(df):
    list1 = list(df["connectionDate"])
    list2 = list(set(list1))
    list2.sort()
    
    indices_list = []
    for i in list2:
        indices_list.append(list1.index(i))
    
    sessions_served_each_day = []
    for i in list2:
        sessions_served_each_day.append(list1.count(i))
    
    # Calculate energy demand on a specific day..
    energy_demand_per_day = []
    for i, j in zip(indices_list, sessions_served_each_day):
        energy_demand = []
        for x in df.iloc[i:(i+j), 5]:
            energy_demand.append(x)
        energy_demand_per_day.append(np.round(np.sum(np.array(energy_demand)),2))
        
    # Now make a dict of connectionDate, energyDemand and sessions.
    df_dic = {"connectionDate":list2,
              "energyDemand" : energy_demand_per_day,
              "sessions" : sessions_served_each_day}
    
    # Now convert this dictionary into a DataFrame.
    ts = pd.DataFrame(df_dic)
    # make connectionDate as the index of the DataFrame.
    ts = ts.set_index(["connectionDate"])
    
    return ts
    


# In[50]:


caltech_ts = make_correct_time_series(caltech_df)
jpl_ts = make_correct_time_series(jpl_df)


# In[51]:


caltech_ts.head()


# In[52]:


jpl_ts.head()


# ## Understanding User Behaviour

# In[53]:


plt.figure(figsize=(10,5))
plt.plot(caltech_ts["energyDemand"][130:400], label="Caltech Energy Demand")
plt.title("Time Series Energy Demand")
plt.xlabel("Time")
plt.ylabel("Energy Demand")
plt.legend()


# In[54]:


plt.figure(figsize=(10,5))
plt.plot(jpl_ts["energyDemand"][30:300], label="JPL Energy Demand")
plt.title("Time Series Energy Demand")
plt.xlabel("Time")
plt.ylabel("Energy Demand")
plt.legend()


# In[55]:


plt.figure(figsize=(10,5))
plt.plot(caltech_ts["sessions"][130:400], label="Caltech sessions served per day")
plt.title("Time Series sessions served at Caltech EVSE")
plt.xlabel("Time")
plt.ylabel("Sessions")
plt.legend()


# In[56]:


plt.figure(figsize=(10,5))
plt.plot(jpl_ts["sessions"][30:300], label="JPL sessions served per day")
plt.title("Time Series sessions served at JPL EVSE")
plt.xlabel("Time")
plt.ylabel("Sessions")
plt.legend()


# -  ***For Caltech ACN :*** The data confirms the difference between paid and free charging facilities. During the first 2.5 years of operation the Caltech ACN was free for drivers. However, from November 1, 2018 a fee of 12 cents per kWh was imposed. We can see this date clearly in the above figure. Right after November 1, 2018 there is a sudden drop in energy demand as well as the number of sessions per day.
# 
# 
# - ***For JPL ACN :*** Because of an issue with the site configuration, approximately half of the charging stations at JPL were free prior to November 1, 2018. After this date, the same fee was also imposed here but we do not see any similar fall in charging demand here because the demand for charging here is high. This high demand overshadows any price sensitivity.

# **Analysing connectionTime and disconnectTime columns**

# In[57]:


caltech_new = caltech_df.copy()
jpl_new = jpl_df.copy()


# In[58]:


caltech_new["connectionTime"] = caltech_new["connectionTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
caltech_new["disconnectTime"] = caltech_new["disconnectTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
caltech_new["doneChargingTime"] = caltech_new["doneChargingTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))


# In[59]:


jpl_new["connectionTime"] = jpl_new["connectionTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
jpl_new["disconnectTime"] = jpl_new["disconnectTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))
jpl_new["doneChargingTime"] = jpl_new["doneChargingTime"].apply(lambda x: np.round(x.time().hour + (x.time().minute)/60))


# In[60]:


caltech_new["connectionMonth"] = caltech_new["connectionDate"].apply(lambda x : x.strftime("%b"))


# In[61]:


caltech_new.head()


# In[62]:


jpl_new["connectionMonth"] = jpl_new["connectionDate"].apply(lambda x : x.strftime("%b"))


# In[63]:


# Plotting arrival time for Free and Paid users on weekDays and weekEnds

a1 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekDay")]["connectionTime"]
a2 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekDay")]["connectionTime"]
a3 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekEnd")]["connectionTime"]
a4 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekEnd")]["connectionTime"]


# In[64]:


fig, axes = plt.subplots(2,2, figsize=(10,10))

fig.subplots_adjust(hspace=0.4, top=0.85)
fig.suptitle("Arrival Time Analysis for Paid and Free Users on weekDays and weekEnds", fontsize=16)

a1.hist(bins=20, ax=axes[0][0], label="weekDay")
a2.hist(bins=20, ax=axes[0][1], label="weekDay")
axes[0][0].set_title("Free Charging")
axes[0][1].set_title("Paid Charging")
axes[0][0].legend()
axes[0][1].legend()

a3.hist(bins=20, ax=axes[1][0], label="weekEnd")
a4.hist(bins=20, ax=axes[1][1], label="weekEnd")
axes[1][0].set_title("Free Charging")
axes[1][1].set_title("Paid Charging")
axes[1][0].legend()
axes[1][1].legend()


# In[65]:


# Plotting departure time for Free and Paid users on weekDays and weekEnds for Caltech EVSE

d1 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekDay")]["disconnectTime"]
d2 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekDay")]["disconnectTime"]
d3 = caltech_new.loc[(caltech_new["userID"]=="unclaimed")&(caltech_new["Day"]=="weekEnd")]["disconnectTime"]
d4 = caltech_new.loc[(caltech_new["userID"]=="claimed")&(caltech_new["Day"]=="weekEnd")]["disconnectTime"]


# In[66]:


fig, axes = plt.subplots(2,2, figsize=(10,10))
fig.subplots_adjust(hspace=0.4, top=0.85)
fig.suptitle("Departure Time Analysis for Paid and Free Users on weekDays and weekEnds", fontsize=16)
d1.hist(bins=20, ax=axes[0][0], label="weekDay")
d2.hist(bins=20, ax=axes[0][1], label="weekDay")
axes[0][0].set_title("Free Charging")
axes[0][1].set_title("Paid Charging")
axes[0][0].legend()
axes[0][1].legend()

d3.hist(bins=20, ax=axes[1][0], label="weekEnd")
d4.hist(bins=20, ax=axes[1][1], label="weekEnd")
axes[1][0].set_title("Free Charging")
axes[1][1].set_title("Paid Charging")
axes[1][0].legend()
axes[1][1].legend()


# In[ ]:





# **Effect of Free vs Paid charging in Caltech EVSE**
# 
# ***1. Arrival time analysis***
# 
# ***weekDay distribution***
# 
# - From the figure we can conclude that the shape of the distributions are similar before and after paid charging was implemented.
# - However two key differences have been observed in weekDay charging between free and paid charging.
#   - First : The peak around 6 pm vanishes as paid charging was implemented. This shows there is a decrement in the community usage of the Caltech ACN after its cost became comparable to at-home charging. So people are preferring to charge their EVs at home instead of standing in a queue at caltech charging station.
#   - Second : The peak in arrivals at around 8 am increases. This may happen because those who are not charging thier EVs in the evening may be using the station in the morning (after coming to office they are connecting their EVs to the charging station.
# - On weekDays there is a morning peak.This means people may be queuing to wait for their chance to pluggin their vehicle. This necessitates a larger infrastucture capacity in the future. As the demand is high in the morning, the owners of the EV station can increase the charging fees in the morning to compensate the infrastructure increment cost.
# 
# ***weekEnd distribution***
# 
# - Since the caltech ACN is open to the public and is located on a university campus , it receives the users on the weekEnds too.
# - We can see a peak at noon for both paid and free. But the peak for paid is lower in comparision to free beacause people perhaps prefer to charge their EVs at home.
# 

# ## Model Building
# 
# - Here I want to build a model to predict Energy Delivered based on features such as connectionTime, session Duration etc.
# - Since our target variable is a continuous value hence we have to build a regression model.
# - A model has been built in the past using this dataset where the author has predicted Charging demand at public charging stations  using XGBoost machine learning method and has achieved R-squared value of 0.52, Mean Absolute Error of 4.6 kWh.
# ***The link of the paper is  :**  https://www.researchgate.net/publication/343693033_Data-Driven_Charging_Demand_Prediction_at_Public_Charging_Stations_Using_Supervised_Machine_Learning_Regression_Methods
# 
# - In the paper mentioned above, the author has used this historical data along with season, location type and charging fees.
# - Since I do not have these information with myself, my results my diverge.
# 
# - In here, I will be using Linear Regression, Support Vector Machines, Random Forest and XGBoost machine learning models. I will compare these models.

# In[67]:


df = pd.concat([caltech_df, jpl_df], axis=0)


# In[68]:


df.shape


# In[69]:


df.head()


# ## Building a Simple Linear Regression Model
# 
# Here we are going to predict the energy delivered based on the session length. Session length is the amount of minutes lapsed between connectionTime and doneChargingTime. Here doneChargingTime is taken because it is the time when the battery became fully charged.

# In[70]:


simple_df = df.loc[:, ["connectionTime", "doneChargingTime", "kWhDelivered"]]


# In[71]:


d1 = simple_df.copy()


# In[72]:


simple_df.head()


# In[73]:


simple_df.shape


# In[74]:


# Now add a column, "session_length" in the dataframe.
simple_df["session_length"] = (simple_df["doneChargingTime"] - simple_df["connectionTime"])/timedelta(minutes=1)


# In[75]:


simple_df.head()


# In[76]:


# drop "connectionTime" and "doneChargingTime" columns..
simple_df = simple_df.drop(columns=["connectionTime", "doneChargingTime"])


# In[77]:


simple_df.head()


# In[78]:


# Check the correlation..
correlation = simple_df.corr()
correlation


# In[79]:


# Let us plot a scatter plot to furthur understand the dataset..
plt.figure(figsize=(10,7))
plt.scatter(x=simple_df["session_length"],
            y=simple_df["kWhDelivered"])


# ***Analysis of the scatter plot :***
# - There are many EV's which are charged for very small period of time but have been delivered with huge amount of energy. We should indentify these instances from the dataframe.
# - There are many EV's which have been charged for more than 2 days but they have been delivered with niegligibly small amount of energy.
# - The reason may be : The data that we have been provided has around 4092 missing values in the "doneChargingTime" column. To fill these missing values, we have copied the corresponding values from disconnectTime column. This may be one of the reasons.

# In[80]:


session_length = list(simple_df["session_length"])
session_length[:10]
session_len_copied = session_length.copy()


# In[81]:


# Let us sort the list in ascending order
session_len_copied.sort()


# In[82]:


# Analysing to 10 smallest session lengths in the dataframe.
session_len_copied[:50]


# ***How can be session length be negative ?***
# 
# Done Charging Time should always be greater than the connectionTime. It means there must be some problem with the dataset. Let us see how many session lengths are negative or close to zero. 

# In[83]:


session_length.index(0.8166666666666667)


# In[84]:


d1.iloc[246]


# In[85]:


caltech_df.iloc[246]


# ***Here EV at index number 246 has been charged for around 1 minute but has consumed 0.586 kWh of energy. It seems there is some problem here. The Ev was connected at 11:45 AM and disconnected at 4:22 PM but its battery became fully charged at 11:46 AM.***

# In[86]:


session_length.index(-1.0)


# In[87]:


d1.iloc[494]


# In[88]:


caltech_df.iloc[494]


# ***Here also the EV was connected at 12:23 PM and disconnected at 5:00 PM but the EV was fully charged at 12:22 PM, which seems a bit odd. So let us find these outliers and remove them from our dataframe.***

# In[89]:


plt.figure(figsize=(10,10))
simple_df[["session_length"]].boxplot()


# In[90]:


for x in ['session_length']:
    q75,q25 = np.percentile(simple_df.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    simple_df.loc[simple_df[x] < min,x] = np.nan
    simple_df.loc[simple_df[x] > max,x] = np.nan


# In[91]:


simple_df["session_length"].isnull().sum()


# ***Hence there are 1812 outliers in the session length column of the dataframe. We have to remove these rows.***

# In[92]:


simple_df = simple_df.dropna()


# In[93]:


simple_df["session_length"].isnull().sum()


# In[94]:


simple_df.shape # From 65062, we have been left with 63250 instances only.


# In[95]:


# Now let us find the correlation again.
correlation = simple_df.corr()
correlation


# ***The correlation between kWhDelivered and session_length columns was around 48% before the removal of outliers has been improved to 60% after the removal of the outliers. This increment is significant.***

# In[96]:


plt.figure(figsize=(10,7))
plt.scatter(x=simple_df["session_length"],
            y=simple_df["kWhDelivered"],
            alpha=0.1)
plt.title("Scatter plot Energy consumed vs Session Length")
plt.xlabel("Session Length")
plt.ylabel("Energy Consumed")
plt.show()


# - The figure shows some horizontal lines at 40 kWh, 15 kWh and closer to 0 kWh. It means there must be some kind of capping at these values.

# **Splitting the dataset into a train and test set**

# In[97]:


simple_df.shape


# In[98]:


# Shuffle the dataframe
simple_df = simple_df.sample(frac=1, random_state=42)


# In[99]:


X = simple_df[["session_length"]]
y = simple_df[["kWhDelivered"]]


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[101]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[102]:


simple_df.head()


# **Build a function to calculate the model performance**

# In[103]:


def calculate_performance(y_test, y_pred):
    MAE = np.round(mean_absolute_error(y_test, y_pred),2)
    RMSE = np.round(np.sqrt(mean_squared_error(y_test, y_pred)),2)
    r_sq = np.round(r2_score(y_test, y_pred),2)
    
    performance_dic = {"MAE": MAE, "RMSE" : RMSE, "r2_score" : r_sq}
    return performance_dic


# **Model Building**

# **Model 1 : Linear Regression**

# In[104]:


model_1_lr = LinearRegression()


# In[105]:


# Fit the training data into the model..
model_1_lr.fit(X_train, y_train)


# In[106]:


# Our model has been trained. Let us predict on test datasets.
y_pred = model_1_lr.predict(X_test)


# In[107]:


model_1_performance = calculate_performance(y_test, y_pred)


# In[108]:


model_1_performance


# **Model 2 : Random Forest Regressor**

# In[109]:


model_2_rf = RandomForestRegressor()


# In[110]:


model_2_rf.fit(X_train, y_train)


# In[111]:


y_pred_rf = model_2_rf.predict(X_test)


# In[112]:


model_2_performance = calculate_performance(y_test, y_pred_rf)
model_2_performance


# In[113]:


y_test[:10], y_pred_rf[:10]


# **Using Cross validation to train the Random Forest Model**

# In[114]:


scores = cross_val_score(model_2_rf, X, y, scoring="neg_mean_squared_error", cv=10)


# In[115]:


rmse_scores = np.sqrt(-scores)


# In[116]:


rmse = np.mean(rmse_scores)


# In[117]:


rmse


# **Model 3 : Support Vector Machine**

# In[118]:


model_3_svr = SVR()


# In[119]:


model_3_svr.fit(X_train, y_train)


# In[120]:


y_pred_svr = model_3_svr.predict(X_test)


# In[121]:


model_3_performance = calculate_performance(y_test, y_pred_svr)
model_3_performance


# **Model 4 : XGBoost**

# In[122]:


from xgboost import XGBRegressor


# In[123]:


model_4_xgb = XGBRegressor()


# In[124]:


model_4_xgb.fit(X_train, y_train)


# In[125]:


y_pred_xgb = model_4_xgb.predict(X_test)


# In[126]:


model_4_performance = calculate_performance(y_test, y_pred_xgb)
model_4_performance


# **Comparing the results of all 4 models**

# In[127]:


results_dic = {"Linear Regression" : model_1_performance,
               "Random Forest" : model_2_performance,
               "Support Vector Machines" : model_3_performance,
               "XGBoost" : model_4_performance}


# In[128]:


results_df = pd.DataFrame(results_dic)
results_df


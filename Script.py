#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


customers_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_customers_dataset.csv')
geolocation_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_geolocation_dataset.csv')
items_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_order_items_dataset.csv')
payments_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_order_payments_dataset.csv')
reviews_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_order_reviews_dataset.csv')
orders_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_orders_dataset.csv')
products_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_products_dataset.csv')
sellers_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/olist_sellers_dataset.csv')
category_translation_df= pd.read_csv('C:/Users/Azizb/Desktop/Pyhton Project/archive (1)/product_category_name_translation.csv')


# In[5]:


customers_df.head()


# In[6]:


geolocation_df.head()


# In[7]:


items_df.head()


# In[8]:


payments_df.head()


# In[9]:


reviews_df.head()


# In[10]:


orders_df.head()


# In[11]:


products_df.head()


# In[12]:


sellers_df.head()


# In[13]:


category_translation_df.head()


# In[14]:


##DATA CLEANING


# In[15]:


##Merging All Dataframes
df= pd.merge(customers_df, orders_df, on="customer_id", how='inner')
df= df.merge(reviews_df, on="order_id", how='inner')
df= df.merge(items_df, on="order_id", how='inner')
df= df.merge(products_df, on="product_id", how='inner')
df= df.merge(payments_df, on="order_id", how='inner')
df= df.merge(sellers_df, on='seller_id', how='inner')
df= df.merge(category_translation_df, on='product_category_name', how='inner')
df.shape


# In[16]:


##Show All Features
df.columns


# In[17]:


##Check duplicates
df.duplicated().sum()


# In[18]:


df.describe()


# In[19]:


df.info()


# In[20]:


# Number of Missing Values for the first half of features

df.isna().sum()[:20]


# In[23]:


##Drop All Missing Values in datetime columns


# In[21]:


df.dropna(subset= ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'], inplace=True)


# In[24]:


df.isna().sum()[20:]


# In[25]:


df[['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']][df.product_weight_g.isna()]


# In[26]:


# Since all the missing values are in the same raw, we will drop this raw
df.drop(27352, inplace=True)

# Reset Index
df.reset_index(inplace= True, drop= True)


# In[27]:


def classify_cat(x):

    if x in ['office_furniture', 'furniture_decor', 'furniture_living_room', 'kitchen_dining_laundry_garden_furniture', 'bed_bath_table', 'home_comfort', 'home_comfort_2', 'home_construction', 'garden_tools', 'furniture_bedroom', 'furniture_mattress_and_upholstery']:
        return 'Furniture'
    
    elif x in ['auto', 'computers_accessories', 'musical_instruments', 'consoles_games', 'watches_gifts', 'air_conditioning', 'telephony', 'electronics', 'fixed_telephony', 'tablets_printing_image', 'computers', 'small_appliances_home_oven_and_coffee', 'small_appliances', 'audio', 'signaling_and_security', 'security_and_services']:
        return 'Electronics'
    
    elif x in ['fashio_female_clothing', 'fashion_male_clothing', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_sport', 'fashion_underwear_beach', 'fashion_childrens_clothes', 'baby', 'cool_stuff', ]:
        return 'Fashion'
    
    elif x in ['housewares', 'home_confort', 'home_appliances', 'home_appliances_2', 'flowers', 'costruction_tools_garden', 'garden_tools', 'construction_tools_lights', 'costruction_tools_tools', 'luggage_accessories', 'la_cuisine', 'pet_shop', 'market_place']:
        return 'Home & Garden'
    elif x in ['sports_leisure', 'toys', 'cds_dvds_musicals', 'music', 'dvds_blu_ray', 'cine_photo', 'party_supplies', 'christmas_supplies', 'arts_and_craftmanship', 'art']:
        return 'Entertainment'
    
    elif x in ['health_beauty', 'perfumery', 'diapers_and_hygiene']:
        return 'Beauty & Health'
    
    elif x in ['food_drink', 'drinks', 'food']:
        return 'Food & Drinks'
    
    elif x in ['books_general_interest', 'books_technical', 'books_imported', 'stationery']:
        return 'Books & Stationery'
    
    elif x in ['construction_tools_construction', 'construction_tools_safety', 'industry_commerce_and_business', 'agro_industry_and_commerce']:
        return 'Industry & Construction'

df['product_category'] = df.product_category_name_english.apply(classify_cat)


# In[28]:


df.product_category.value_counts()


# In[29]:


# Create Volume Column
df['product_vol_cm3'] = df.product_length_cm * df.product_width_cm * df.product_height_cm

# Drop Width, Height and Length
df.drop(['product_length_cm', 'product_width_cm', 'product_height_cm'], axis= 1, inplace= True)


# In[30]:


df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])
df['order_delivered_carrier_date'] =pd.to_datetime(df['order_delivered_carrier_date'])


# In[31]:


df['estimated_days'] = (df['order_estimated_delivery_date'].dt.date - df['order_purchase_timestamp'].dt.date).dt.days


# In[32]:


df['arrival_days'] = (df['order_delivered_customer_date'].dt.date - df['order_purchase_timestamp'].dt.date).dt.days


# In[33]:


df['shipping_days'] = (df['order_delivered_customer_date'].dt.date - df['order_delivered_carrier_date'].dt.date).dt.days


# In[34]:


df.drop((df[['order_delivered_carrier_date', 'order_delivered_customer_date']][df.shipping_days < 0]).index, inplace= True)


# In[35]:


# First get seller to carrier duration in days
df['seller_to_carrier_status'] = (df['shipping_limit_date'].dt.date - df['order_delivered_carrier_date'].dt.date).dt.days

# Now calssify the duration into 'OnTime/Early' & 'Late'
df['seller_to_carrier_status'] = df['seller_to_carrier_status'].apply(lambda x : 'OnTime/Early' if x >=0 else 'Late')


# In[36]:


# First get difference between estimated delivery date and actual delivery date in days
df['arrival_status'] = (df['order_estimated_delivery_date'].dt.date - df['order_delivered_customer_date'].dt.date).dt.days

# Now Classify the duration in 'OnTime/Early' & 'Late'
df['arrival_status'] = df['arrival_status'].apply(lambda x : 'OnTime/Early' if x >=0 else 'Late')


# In[37]:


df[['estimated_days', 'arrival_days', 'shipping_days']].describe()


# In[38]:


outlier_indices = df[(df.estimated_days > 60) | (df.arrival_days > 60) | (df.shipping_days > 60)].index

df.drop(outlier_indices, inplace= True)
df.reset_index(inplace= True, drop= True)


# In[39]:


def rates(x):

    if x in range(0, 8):
        return 'Very Fast'
    
    elif x in range(8, 16):
        return 'Fast'
    
    elif x in range(16, 25):
        return 'Neutral'
    
    elif x in range(25, 40):
        return 'Slow'
    
    else:
        return 'Very Slow'

df['estimated_delivery_rate'] = df.estimated_days.apply(rates)

df['arrival_delivery_rate'] = df.arrival_days.apply(rates)

df['shipping_delivery_rate'] = df.shipping_days.apply(rates)


# In[40]:


plt.figure(figsize=[10, 6])
sns.barplot(x = df.customer_city.value_counts().values[:10], y = df.customer_city.value_counts().index[:10], palette= 'crest_r')
plt.title('Top 10 Customers Capacity Cities')
sns.despine()


# In[41]:


plt.figure(figsize=[10, 6])
sns.barplot(x = df.customer_state.value_counts().values[:10], y = df.customer_state.value_counts().index[:10], palette= 'crest_r')
plt.title('Top 10 Customers Capacity States')
sns.despine()


# In[42]:


df.order_status.value_counts()


# In[43]:


df.drop('order_status', axis=1, inplace=True)


# In[44]:


plt.figure(figsize=[15, 8])
review_score_index = [str(i) for i in df.review_score.value_counts().index]
sns.barplot(x = review_score_index, y= df.review_score.value_counts().values, palette= 'crest_r')
plt.title('Review Scores')
sns.despine()


# In[45]:


plt.figure(figsize=[10, 6])
sns.set_palette('crest_r')
sns.distplot(x = df.price)
plt.title('Price Distribution')
sns.despine()


# In[46]:


plt.figure(figsize=[10, 6])
sns.set_palette('crest_r')
sns.distplot(x = df.freight_value)
plt.title('Freight Value Distribution')
sns.despine()


# In[47]:


plt.figure(figsize=[10, 6])
sns.barplot(x = df.product_category.value_counts().values, y = df.product_category.value_counts().index, palette= 'crest_r')
plt.title('Number of orders per each Category')
plt.xticks(rotation = 45)
sns.despine()


# In[48]:


plt.figure(figsize=[10, 6])
sns.distplot(x = df.product_name_lenght)
plt.title('Product Name Length')
df.product_category.value_counts().values


# In[49]:


plt.figure(figsize=[10, 6])
sns.distplot(x = df.product_description_lenght)
plt.title('Product Describtion Length')
sns.despine()


# In[50]:


plt.figure(figsize=[10, 6])
sns.countplot(x = df.product_photos_qty, palette= 'crest_r')
plt.title('Product Photos Quantity')
sns.despine()


# In[51]:


plt.figure(figsize=[10, 6])
sns.distplot(x = df.product_weight_g)
plt.title('Product Weight')
sns.despine()


# In[52]:


plt.figure(figsize=[10, 6])
sns.distplot(x = df.product_vol_cm3)
plt.title('Product Volume')
sns.despine()


# In[53]:


plt.figure(figsize=[10, 10])
plt.pie(df.payment_type.value_counts().values, explode=(0.05, 0.05, 0.05, 0.05), labels= df.payment_type.value_counts().index, autopct='%1.1f%%',shadow=True, startangle=90);


# In[54]:


df.payment_installments.value_counts()


# In[55]:


df[df.payment_installments == 0]


# In[56]:


# Drop indices
df.drop([29113, 29114, 96733], inplace=True)

# Reset Index
df.reset_index(inplace= True, drop= True)


# In[57]:


plt.figure(figsize=[10, 6])
sns.countplot(x = df.payment_installments, palette= 'crest_r')
plt.title('Installments Distribution')
sns.despine()


# In[58]:


plt.figure(figsize=[10, 6])
sns.distplot(x = df.payment_value)
plt.title('Payment Value')
sns.despine()


# In[59]:


plt.figure(figsize=[10, 6])
sns.barplot(x = df.seller_city.value_counts().values[:10], y= df.seller_city.value_counts().index[:10], palette= 'crest_r')
plt.title('Top 10 Sellers Cities')
sns.despine()


# In[60]:


plt.figure(figsize=[15, 8])
sns.barplot(x = df.seller_state.value_counts().values[:10], y= df.seller_state.value_counts().index[:10], palette= 'crest_r')
plt.title('Top 10 Sellers States')
sns.despine()


# In[61]:


# Group customer city by payment value
revenue_per_city = df.groupby('customer_city')[['payment_value']].sum().sort_values(by='payment_value', ascending=False)
revenue_per_city.reset_index(inplace=True)

# plot Top 10 cities with highest revenue
plt.figure(figsize=[15, 8])
sns.barplot(x = revenue_per_city.customer_city[:10], y= revenue_per_city.payment_value[:10], palette= 'crest_r')
plt.title('Top 10 Cities with highest Revenue', fontsize= 15)
plt.xlabel('Customer City', fontsize= 12)
plt.ylabel('Total Payments in Millions',fontsize= 12)
sns.despine()


# In[62]:


# Filter product category with 4.5 or above
review_per_cat = df.groupby('product_category')[['review_score']].mean().sort_values(by='review_score', ascending=False)
review_per_cat.reset_index(inplace=True)

# Plot Product Category vs Review Score
plt.figure(figsize=[15, 8])
sns.barplot(x = review_per_cat.review_score, y= review_per_cat.product_category, palette= 'crest_r')
plt.title('Average Review Score per Product Category', fontsize= 15)
plt.xlabel('Review Score', fontsize=12)
plt.ylabel('Prodcut Category', fontsize= 12)
ax = plt.gca()
ax.set_frame_on(False);


# In[63]:


# Group each payment type by average payment value
payment_methods = df.groupby('payment_type')[['payment_value']].sum().sort_values(by='payment_value', ascending=False)
payment_methods.reset_index(inplace=True)

# plot Average payments per payment method
plt.figure(figsize=[15, 8])
sns.barplot(x = payment_methods.payment_type, y= payment_methods.payment_value, palette= 'crest_r')
plt.title('Total Revenue per payment method', fontsize= 15)
plt.xlabel('Payment Type', fontsize= 12)
plt.ylabel('Revenue per Payment type (Millions)', fontsize= 12)
sns.despine()


# In[65]:


#Group each payment type by average payment value
payment_methods = df.groupby('payment_type')[['payment_value']].mean().sort_values(by='payment_value', ascending=False)
payment_methods.reset_index(inplace=True)

# plot Average payments per payment method
plt.figure(figsize=[15, 8])
sns.barplot(x = payment_methods.payment_type, y= payment_methods.payment_value, palette= 'crest_r')
plt.title('Average payment value per payment method', fontsize= 15)
plt.xlabel('Payment Type', fontsize= 12)
plt.ylabel('Average Payment Value', fontsize= 12)
sns.despine()


# In[66]:


# Group product category by average freight value
freight_per_cat = df.groupby('product_category')[['freight_value']].mean().sort_values(by='freight_value', ascending=False)
freight_per_cat.reset_index(inplace=True)

# plot average freight value per product category
plt.figure(figsize=[15, 8])
sns.barplot(x = freight_per_cat.freight_value, y=  freight_per_cat.product_category, palette= 'crest_r')
plt.title('Average Freight Value per Product Category', fontsize= 15)
plt.xlabel('Average Freight Value',fontsize= 12)
plt.ylabel('Product Category', fontsize= 12)
ax = plt.gca()
ax.set_frame_on(False);


# In[67]:


# Group product category by average arrival time
ship_per_cat = df.groupby('product_category')[['arrival_days']].mean().sort_values(by='arrival_days', ascending=False)
ship_per_cat.reset_index(inplace=True)

# plot average freight value per product category
plt.figure(figsize=[15, 8])
sns.barplot(x = ship_per_cat.arrival_days, y=  ship_per_cat.product_category, palette= 'crest_r')
plt.title('Average arrival Time per Product Category', fontsize= 15)
plt.xlabel('Average arrival time (days)',fontsize= 12)
plt.ylabel('Product Category', fontsize= 12)
ax = plt.gca()
ax.set_frame_on(False);


# In[68]:


plt.figure(figsize=[15, 8])
voucher_trans = df[df.payment_type == 'voucher']
sns.countplot(x= voucher_trans.review_score)


# In[69]:


plt.figure(figsize=[30,8])
Values = df.arrival_status.value_counts().values
Labels = df.arrival_status.value_counts().index
plt.pie(Values, explode=(0.05, 0.05), labels= ['OnTime/Early', 'Late'], autopct='%1.1f%%', shadow=True, colors= ('#0000FF', '#C0C0C0'));


# In[70]:


plt.figure(figsize=[15, 8])
sns.scatterplot(x = geolocation_df.geolocation_lng, y = geolocation_df.geolocation_lat, hue= df.product_category)
plt.title('Distribution Of Categories by location', fontsize= 15)
plt.xlabel('Longitude',fontsize= 12)
plt.ylabel('Latitude', fontsize= 12)
ax = plt.gca()
ax.set_frame_on(False);sns.despine()


# In[71]:


# Create copy of DataFrame
df_2 = df.copy()

# Save sample for EDA Deployment
EDA_df = df_2.drop(['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'order_id', 'review_id', 'order_item_id', 'product_id', 'seller_id', 'seller_zip_code_prefix', 'product_category_name'], axis= 1)
EDA_sample = EDA_df.sample(frac= 1)[:10000]
EDA_sample.to_csv('EDA.csv')


# In[72]:


# Drop all ids, zip codes, datetimes, review comment and title, product length

df.drop(['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state', 'order_id', 'order_purchase_timestamp',
        'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date',
        'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp', 'payment_sequential',
        'order_item_id', 'product_id', 'seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state', 'shipping_limit_date', 'product_category_name',
        'product_category_name_english', 'product_category', 'product_weight_g', 'product_name_lenght',
        'product_vol_cm3'], axis= 1, inplace= True)


# In[73]:


# Show Correlation between Features
plt.figure(figsize= [10, 6])
sns.heatmap(df.corr(), annot= True)


# In[74]:


# Remove features with high correlations
df.drop(['shipping_days', 'price'], axis= 1, inplace= True)


# In[75]:


df.head()


# In[76]:


encoded_class = { 1 : 'Not Satisfied',
                  2 : 'Not Satisfied', 
                  3 : 'Not Satisfied', 
                  4 : 'Satisfied', 
                  5 : 'Satisfied'}

df['review_score'] = df['review_score'].map(encoded_class)


# In[77]:


X = df.drop('review_score', axis=1)
y = df['review_score']


# In[78]:


labels = {'Very Slow' : 1, 
          'Slow' : 2, 
          'Neutral' : 3, 
          'Fast' : 4, 
          'Very Fast' : 5}

X.estimated_delivery_rate = X.estimated_delivery_rate.map(labels)
X.shipping_delivery_rate = X.shipping_delivery_rate.map(labels)
X.arrival_delivery_rate = X.arrival_delivery_rate.map(labels)


# In[79]:


X = pd.get_dummies(X, drop_first=True)


# In[80]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42, stratify= y)


# In[81]:


from sklearn.feature_selection import mutual_info_classif, SelectKBest
fs = SelectKBest(mutual_info_classif, k= 'all')
fs.fit(x_train, y_train)
x_train_fs = fs.transform(x_train)
x_test_fs = fs.transform(x_test)


# In[82]:


# Get the indices sorted by most important to least important
plt.figure(figsize=[15, 8])
indices = np.argsort(fs.scores_)[::-1]

# To get your top 10 feature names
features = []
for i in range(15):
    features.append(fs.feature_names_in_[indices[i]])

# Now plot
sns.barplot(x = fs.scores_[indices[range(15)]], y = features)


# In[83]:


from sklearn.feature_selection import mutual_info_classif, SelectKBest
fs = SelectKBest(mutual_info_classif, k= 9)
fs.fit(x_train, y_train)
x_train_fs = fs.transform(x_train)
x_test_fs = fs.transform(x_test)


# In[84]:


x_train_fs = pd.DataFrame(x_train_fs, columns= fs.get_feature_names_out())
x_test_fs = pd.DataFrame(x_test_fs, columns= fs.get_feature_names_out())


# In[85]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean= False)
x_train_scaled = sc.fit_transform(x_train_fs)
x_test_scaled = sc.transform(x_test_fs)


# In[86]:


x_train_scaled = pd.DataFrame(x_train_scaled, columns= sc.get_feature_names_out())
x_test_scaled = pd.DataFrame(x_test_scaled, columns= sc.get_feature_names_out())


# In[87]:


round((y_train.value_counts() / y_train.shape[0]) * 100, 2)


# In[1]:


get_ipython().system('pip install imblearn')


# In[90]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state= 42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)


# In[91]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix

lr = LogisticRegression()
lr.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, lr.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, lr.predict(x_test_scaled)))

plot_confusion_matrix(lr, x_train_resampled, y_train_resampled)
plot_confusion_matrix(lr, x_test_scaled, y_test)


# In[92]:


#KNN Classifier¶
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, knn.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, knn.predict(x_test_scaled)))

plot_confusion_matrix(knn,x_train_resampled,y_train_resampled)
plot_confusion_matrix(knn, x_test_scaled, y_test)


# In[94]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, dt.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, dt.predict(x_test_scaled)))

plot_confusion_matrix(dt,x_train_resampled,y_train_resampled)
plot_confusion_matrix(dt, x_test_scaled, y_test)


# In[95]:


#Random Forest¶
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, rf.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, rf.predict(x_test_scaled)))

plot_confusion_matrix(rf, x_train_resampled, y_train_resampled)
plot_confusion_matrix(rf, x_test_scaled, y_test)


# In[96]:


#Ada Boost¶
from sklearn.ensemble import AdaBoostClassifier

ad = AdaBoostClassifier()
ad.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, ad.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, ad.predict(x_test_scaled)))

plot_confusion_matrix(ad, x_train_resampled, y_train_resampled)
plot_confusion_matrix(ad, x_test_scaled, y_test)


# In[98]:


pip install xgboost


# In[99]:


#XGboost¶
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_xg = le.fit_transform(y_train_resampled)
y_test_xg = le.fit_transform(y_test)
xg = XGBClassifier()
xg.fit(x_train_resampled, y_train_xg)

print('Evaluation on Training \n', classification_report(y_train_xg, xg.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test_xg, xg.predict(x_test_scaled)))

plot_confusion_matrix(xg, x_train_resampled, y_train_xg)
plot_confusion_matrix(xg, x_test_scaled, y_test_xg)


# In[100]:


#Naive Bayes¶
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, plot_confusion_matrix

nb = GaussianNB()
nb.fit(x_train_resampled, y_train_resampled)
y_pred = nb.predict(x_test_scaled)

print('Evaluation on Training \n', classification_report(y_train_resampled, nb.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, nb.predict(x_test_scaled)))

plot_confusion_matrix(nb, x_train_resampled, y_train_resampled)
plot_confusion_matrix(nb, x_test_scaled, y_test)


# In[102]:


pip install lightgbm


# In[103]:


#LightGBM
import lightgbm as ltb

lg = ltb.LGBMClassifier()
lg.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, lg.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, lg.predict(x_test_scaled)))

plot_confusion_matrix(lg, x_train_resampled, y_train_resampled)
plot_confusion_matrix(lg, x_test_scaled, y_test)


# In[104]:


get_ipython().system(' pip install catboost')


# In[105]:


#CatBoost¶
import catboost as cb

cb = cb.CatBoostClassifier()
cb.fit(x_train_resampled, y_train_resampled)

print('Evaluation on Training \n', classification_report(y_train_resampled, cb.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test, cb.predict(x_test_scaled)))

plot_confusion_matrix(cb, x_train_resampled, y_train_resampled)
plot_confusion_matrix(cb, x_test_scaled, y_test)


# In[106]:


#from sklearn.model_selection import GridSearchCV

#param_grid = {
    #'learning_rate': [0.1, 0.2],
    #'max_depth': [5, 7, 8],
    #'n_estimators': [100, 200] 
    #}
#grid_search = GridSearchCV(xg, param_grid= param_grid, cv= 5, scoring= 'f1_macro')
#grid_search.fit(x_train_resampled, y_train_xg)


# In[107]:


final_xg_model = XGBClassifier(learning_rate= 0.2, max_depth= 8, n_estimators= 200)
final_xg_model.fit(x_train_resampled, y_train_xg)

print('Evaluation on Training \n', classification_report(y_train_xg, final_xg_model.predict(x_train_resampled)))
print('Evaluation on Testing \n', classification_report(y_test_xg, final_xg_model.predict(x_test_scaled)))


# In[108]:


#param_grid = {
    #'learning_rate': [0.1, 0.2],
    #'depth': [5, 7, 8],
    #'iterations': [100, 200]}

#grid_search = GridSearchCV(cb, param_grid= param_grid, cv= 5, scoring= 'f1_macro')
#grid_search.fit(x_train_resampled, y_train_resampled)
#grid_search.best_params_


# In[109]:


import catboost as cb
final_cb_model = cb.CatBoostClassifier(depth= 7, iterations= 200, learning_rate= 0.2)
final_cb_model.fit(x_train_resampled,y_train_resampled)
print('Evaluation on Training \n', classification_report(y_train_resampled, final_cb_model.predict(x_train_resampled)))
print('Evaluation on Training \n', classification_report(y_test, final_cb_model.predict(x_test_scaled)))


# In[110]:


#RandomForest¶
#param_grid = {
    #'max_depth': [8, 9, 10],
    #'n_estimators': [100, 200]
    #}
#grid_search = GridSearchCV(rf, param_grid= param_grid, cv= 5, scoring= 'f1_macro')
#grid_search.fit(x_train_resampled, y_train_resampled)
#grid_search.best_params_


# In[111]:


final_rf_model = RandomForestClassifier(n_estimators= 200, max_depth= 10)
final_rf_model.fit(x_train_resampled,y_train_resampled)
print('Evaluation on Training \n', classification_report(y_train_resampled, final_rf_model.predict(x_train_resampled)))
print('Evaluation on Training \n', classification_report(y_test, final_rf_model.predict(x_test_scaled)))


# In[112]:


#7.0 Pipeline¶
df_pipeline = df.copy()
df_pipeline.head()


# In[113]:


encoded_class = { 'Not Satisfied' : 0,
                  'Satisfied' : 1,
                }

df_pipeline['review_score'] = df_pipeline['review_score'].map(encoded_class)


# In[114]:


X = df_pipeline.drop('review_score', axis=1)
y = df_pipeline['review_score']


# In[115]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify= y)


# In[116]:


numeric_columns = x_train.select_dtypes(exclude = 'object').columns
numeric_columns


# In[117]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numerical_pipeline = Pipeline(steps=[('Handle Missing Values', SimpleImputer(strategy= 'median')), 
                                    ('Feature Scaling', StandardScaler(with_mean=False))])


# In[118]:


cat_columns = x_train.select_dtypes(include = 'object').columns
cat_columns


# In[119]:


cat_pipeline = Pipeline(steps=[('Handle Missing Values', SimpleImputer(strategy= 'most_frequent')),
                                ('OneHot Encoding', OneHotEncoder(drop= 'first')),
                                ('Feature Scaling', StandardScaler(with_mean= False))])


# In[120]:


from sklearn.compose import ColumnTransformer

preprocessing = ColumnTransformer(transformers=[('Numerical Columns', numerical_pipeline, numeric_columns),
                                                ('Cat Columns', cat_pipeline, cat_columns)], remainder= 'passthrough')
preprocessing


# In[121]:


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

final_pipeline = Pipeline(steps=[('Preprocessing', preprocessing), ('Smote', SMOTE()), 
                                ('Model', XGBClassifier(learning_rate= 0.2, max_depth= 8, n_estimators= 200))])
final_pipeline


# In[122]:


final_pipeline.fit(x_train, y_train)


# In[123]:


import joblib
joblib.dump(final_pipeline, 'Brazilian Ecommerce Classification.bkl')


# In[124]:


#8.0 NLP For Customer Satisfaction¶
reviews_df.head()


# In[125]:


# Remove 'review_comment_title' because of high missing values perentage and remove other features for unneccessity
reviews_df = reviews_df[['review_comment_message', 'review_score']]

reviews_df.info()


# In[126]:


# Drop missing values
reviews_df.dropna(inplace= True)

# Rename columns for ease
reviews_df.rename(columns = {'review_comment_message' : 'comment', 'review_score' : 'score'}, inplace= True)

# Reset index
reviews_df.reset_index(inplace= True, drop= True)


# In[127]:


# Encode scores to be Satisfied or Not Satisfied
encoded_class = { 1 : 'Not Satisfied',
                  2 : 'Not Satisfied', 
                  3 : 'Not Satisfied', 
                  4 : 'Satisfied', 
                  5 : 'Satisfied'}

reviews_df['score'] = reviews_df['score'].map(encoded_class)


# In[129]:


import nltk
nltk.download('stopwords')


# In[130]:


#Text Cleaning & Processing¶
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
corpus = []

for i in range(reviews_df.shape[0]):
    # Remove any special characters or number
    comment = re.sub('[^a-zA-Z]', ' ', reviews_df.comment[i])
    # Lower text
    comment = comment.lower()
    # Remove any spaces before or after text
    comment = comment.strip()
    # Split text for stemming
    comment = comment.split()
    # Stemming words in Portugues
    comment = [stemmer.stem(word) for word in comment if word not in set(stopwords.words('portuguese'))]
    # Merge stemmed words to be sentences
    comment = ' '.join(comment)
    
    corpus.append(comment)
    
corpus


# In[131]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Use TFIDF Vectorizer to convert text into numbers
tf = TfidfVectorizer()
df_new = tf.fit_transform(corpus).toarray()


# In[132]:


df_new = pd.DataFrame(df_new, columns= tf.get_feature_names_out())
df_new


# In[133]:


#Split into Input Features & Target Variable¶
X = df_new
y = reviews_df['score']


# In[134]:


#Split data into Train & Test¶
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify= y)


# In[135]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, plot_confusion_matrix

nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

print('Evaluation on Training \n', classification_report(y_train, nb.predict(x_train)))
print('Evaluation on Testing \n', classification_report(y_test, nb.predict(x_test)))

plot_confusion_matrix(nb, x_train, y_train)
plot_confusion_matrix(nb, x_test, y_test)


# In[136]:


#Check Reasons for Non-Satisfaction¶
# Create a separate DataFrame to check reasons of non satisfaction 
non_satisfied = reviews_df[reviews_df.score == 'Not Satisfied']

# Reset index
non_satisfied.reset_index(inplace= True, drop= True)


# In[137]:


#Text Cleaning & Processing¶
stemmer = PorterStemmer()
corpus = []

for i in range(non_satisfied.shape[0]):
    comment = re.sub('[^a-zA-Z]', ' ', non_satisfied.comment[i])
    comment = comment.lower()
    comment = comment.strip()
    comment = comment.split()
    comment = [stemmer.stem(word) for word in comment if word not in set(stopwords.words('portuguese'))]
    comment = ' '.join(comment)
    
    corpus.append(comment)
    
corpus


# In[138]:


#Translate sample of non-satisfied comments for comprehension¶
# First install deep_translator library
get_ipython().system(' pip install deep_translator')


# In[139]:


from deep_translator import GoogleTranslator
import random

non_satisfied_trans = []
random.seed(42)

for sentence in random.sample(corpus, 1000):
    non_satisfied_trans.append(GoogleTranslator(source='portuguese', target='english').translate(sentence))
    
non_satisfied_trans


# In[140]:


# Apply stemming to the translated text
non_satisfied_final = []

for sent in non_satisfied_trans:

    for word in sent.split():

        if word not in set(stopwords.words('english')):

            non_satisfied_final.append(stemmer.stem(word))

non_satisfied_final


# In[142]:


pip install wordcloud


# In[143]:


# Visualize most common words for non-satsifaction
from wordcloud import WordCloud

non_satisfied_final = ' '.join(non_satisfied_final)
non_satisfied_freq = WordCloud(width=1000, height=800, background_color='white').generate(non_satisfied_final)

plt.figure(figsize=(15, 10))
plt.imshow(non_satisfied_freq)
plt.axis("off")


# In[144]:


# Sort the word frequencies in descending order
non_satisfied_freq = non_satisfied_freq.process_text(non_satisfied_final)
sorted_word_frequencies = sorted(non_satisfied_freq.items(), key= lambda x : x[1], reverse=True)


# In[145]:


# Plot Words vs Frequency
plt.figure(figsize= [20, 10])
sns.barplot(x = pd.DataFrame(sorted_word_frequencies)[1:20][0], y= pd.DataFrame(sorted_word_frequencies)[1:20][1], palette= 'crest_r')
plt.title('Top 20 Words and their Frequinces', fontsize= 15)
plt.xlabel('Words', fontsize= 12)
plt.ylabel('Frequency', fontsize= 12)
sns.despine()


# In[146]:


#Customer Segmentation by RFM Analysis¶
df_2.head()


# In[147]:


# Get last transaction date to help calculate Recency
max_trans_date = max(df_2.order_purchase_timestamp).date()
max_trans_date


# In[148]:


#Create Recency, Frequancy and Monetary Features¶
from datetime import datetime

rfm_table = df_2.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda x:(datetime.strptime(str(max_trans_date),'%Y-%m-%d') - x.max()).days,
                                                                'product_id': lambda x:len(x),
                                                             'payment_value': lambda x:sum(x)})
rfm_table


# In[149]:


# Rename columns
rfm_table.rename(columns={'order_purchase_timestamp':'Recency','product_id':'Frequancy','payment_value':'Monetary'}, inplace=True)
rfm_table


# In[150]:


#Create Recency, Frequancy and Monetary scores¶
rfm_table['r_score'] = pd.qcut(rfm_table['Recency'], 4, ['4','3','2','1'])
rfm_table['f_score'] = pd.qcut(rfm_table['Frequancy'].rank(method= 'first'), 4, ['1','2','3','4'])
rfm_table['m_score'] = pd.qcut(rfm_table['Monetary'], 4, ['1','2','3','4'])
rfm_table


# In[151]:


plt.figure(figsize= [10, 6])
sns.scatterplot(x= 'Recency',y= 'Monetary', data=rfm_table)
plt.title('Recency vs Monetary', fontsize= 15)
sns.despine()


# In[152]:


plt.figure(figsize= [10, 6])
sns.scatterplot(x='Frequancy', y='Monetary', data=rfm_table)
plt.title('Frequancy vs Monetary', fontsize= 15)
sns.despine()


# In[153]:


rfm_table['rfm_score'] = 100 * rfm_table['r_score'].astype(int) + 10 * rfm_table['f_score'].astype(int)+ rfm_table['m_score'].astype(int)
rfm_table


# In[155]:


#Cluster customers based on RFM Score
def customer_segmenation(rfm_score):
  
  if rfm_score == 444:
    return 'VIP'
  
  elif  rfm_score >= 433 and rfm_score < 444:
    return 'Very Loyal'
  
  elif   rfm_score >=421 and rfm_score< 433:
    return 'Potential Loyalist'
  
  elif rfm_score>=344 and rfm_score < 421:
    return 'New customers'
  
  elif rfm_score>=323 and rfm_score<344:
    return 'Potential customer'
  
  elif rfm_score>=224 and rfm_score<311:
    return 'High risk to churn' 
  
  else:
    return 'Lost customers'       
  
rfm_table['customer_segmentation'] = rfm_table['rfm_score'].apply(customer_segmenation)

rfm_table


# In[156]:


# Plot frquency of each segment
plt.figure(figsize=[10,6])
sns.barplot(x = rfm_table.customer_segmentation.value_counts().values, y= rfm_table.customer_segmentation.value_counts().index, palette= 'crest_r')
sns.despine()


# In[157]:


#Check Outliers¶
rfm_table.describe()


# In[158]:


#Recency
sns.boxplot(x= rfm_table.Recency)
sns.stripplot(x = rfm_table.Recency, color= 'black')


# In[159]:


#Frequancy¶
sns.boxplot(x= rfm_table.Frequancy)
sns.stripplot(x = rfm_table.Frequancy, color= 'black')


# In[160]:


#Monetary¶
sns.boxplot(x= rfm_table.Monetary)
sns.stripplot(x = rfm_table.Monetary, color= 'black')


# In[161]:


#Remove Extreme 5% of Outliers¶
print('Recency 5% Outliers Limits:', np.percentile(rfm_table.Recency, 5), np.percentile(rfm_table.Recency, 95))
print('Frequancy 5% Outliers Limits:', np.percentile(rfm_table.Frequancy, 5), np.percentile(rfm_table.Frequancy, 95))
print('Monetary 5% Outliers Limits:', np.percentile(rfm_table.Monetary, 5), np.percentile(rfm_table.Monetary, 95))


# In[162]:


#Remove Outliers for Recency & Monetary (Extreme 5%)¶
for i in [0, 2]:

    outlier_indices = []
    col = rfm_table.columns[i]
    percentile_5 = np.percentile(rfm_table[col], 5)
    percentile_95 = np.percentile(rfm_table[col], 95)
    outlier_indices.append(rfm_table[(rfm_table[col] < percentile_5) | (rfm_table[col] > percentile_95)].index)

rfm_table.drop(outlier_indices[0][:], inplace= True)
rfm_table.reset_index(inplace= True, drop= True)


# In[163]:


get_ipython().system(' pip install squarify')


# In[164]:


#Customer Segmentation Grid
import squarify

plt.figure(figsize=[15,8])
plt.rc('font', size=15)

Sizes = rfm_table.groupby('customer_segmentation')[['Monetary']].count()
squarify.plot(sizes= Sizes.values, label= Sizes.index, color=["red", "orange", "blue", "yellow", "fuchsia", "green", "royalblue"], alpha=.55)
plt.suptitle("Customer Segmentation Grid", fontsize=25);


# In[165]:


#Recency & Monetary Plot¶
plt.figure(figsize= [15, 8])
colors = ['purple', 'green', 'red', 'blue', 'orange', 'royalblue', 'yellow']
sns.scatterplot(x= rfm_table.Recency, y= rfm_table.Monetary, hue= rfm_table.customer_segmentation, palette= colors)
plt.legend(prop={'size':10})
sns.despine()


# In[166]:


#Check Skeweness¶
# Recency
sns.distplot(x= rfm_table.Recency)


# In[167]:


# Frequancy
sns.distplot(x= rfm_table.Frequancy)


# In[168]:


# Monetary
sns.distplot(x= rfm_table.Monetary)


# In[169]:


#Apply Log function to handle skeweness for Frequancy & Monetary¶
for i in ['Frequancy', 'Monetary']:
    rfm_table[i] = np.log10(rfm_table[i])


# In[170]:


# Frequancy
sns.distplot(x= rfm_table.Frequancy)


# In[171]:


# Monetary
sns.distplot(x= rfm_table.Monetary)


# In[172]:


#Clustering with K-means¶
df_cluster = df_2[['freight_value', 'price', 'payment_value', 'payment_installments', 'payment_sequential']]
df_cluster


# In[173]:


#Take sample from data (10k)¶
df_sample = df_cluster.sample(frac= 1, random_state= 42)[:10000]


# In[174]:


#Save sample as CSV for deployment¶
df_sample.to_csv('Clustering Sample.csv')


# In[175]:


df_sample.describe()


# In[176]:


#Drop freight values with zeros¶
df_sample.drop(df_sample[df_sample.freight_value == 0].index, inplace= True)
df_sample.reset_index(inplace= True, drop= True)


# In[177]:


#Take copy for Pipeline¶
cluster_pipeline = df_sample.copy()


# In[178]:


for i in ['freight_value', 'price', 'payment_value', 'payment_installments', 'payment_sequential']:
    df_sample[i] = np.log10(df_sample[i])


# In[179]:


#Feature Scaling¶
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean= False)
data_scaled = sc.fit_transform(df_sample)


# In[180]:


#Detecting number of clusters uning Elbow Method¶
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
wcss = []
scores = []

for i in range(2,10):
  kmean = KMeans(n_clusters=i)
  y_pred_kmean = kmean.fit_predict(data_scaled)
  wcss.append(kmean.inertia_)
  scores.append(silhouette_score(data_scaled,y_pred_kmean))

plt.plot(range(2,10),wcss)
plt.title('number of cluster vs WCSS')
plt.xlabel('number of cluster')
plt.ylabel('WCSS')


# In[181]:


#Detecting number of clusters using Silhouete Score¶
plt.plot(range(2,10),scores)
plt.title('number of cluster vs silhouette_score')
plt.xlabel('number of cluster')
plt.ylabel('silhouette_score')


# In[182]:


#Select number of clusters k= 3¶
from sklearn.cluster import KMeans

kmean = KMeans(n_clusters= 3)
y_pred_kmean = kmean.fit_predict(data_scaled)


# In[183]:


# Count of each cluster
len(kmean.labels_[kmean.labels_ == 0]), len(kmean.labels_[kmean.labels_ == 1]), len(kmean.labels_[kmean.labels_ == 2])


# In[184]:


# Take another sample of original cluster dataframe to assign kmeans labels
original_cluster_sample = df_cluster.sample(frac= 1, random_state= 42)[:9966]

# Assign cluster label to original cluster sample
original_cluster_sample['cluster_label'] = y_pred_kmean
original_cluster_sample.head()


# In[185]:


original_cluster_sample.groupby('cluster_label').describe().T


# In[186]:


sns.pairplot(data= original_cluster_sample, hue= 'cluster_label', palette= ['blue', 'orange', 'red'])


# In[187]:


#As we can see from statistics table and pairplot that clusters have high percentage of overlaping, sow RFM would be better in this case to cluster customers.¶
#Show Kmeans Clusters
plt.figure(figsize=[10, 6])
plt.scatter(data_scaled[y_pred_kmean==0,0], data_scaled[y_pred_kmean==0,1], c = 'red',label = 'cluster1')
plt.scatter(data_scaled[y_pred_kmean==1,0], data_scaled[y_pred_kmean==1,1], c = 'green',label = 'cluster2')
plt.scatter(data_scaled[y_pred_kmean==2,0], data_scaled[y_pred_kmean==2,1], c = 'blue',label = 'cluster3')
plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], c='yellow', s= 100,label= 'Centroids')
plt.title('Customers Kmeans Clusters')
plt.legend()
sns.despine()


# In[188]:


#Show Clusters using PCA¶
from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
x_pca = pca.fit_transform(data_scaled)
pca.explained_variance_ratio_


# In[189]:


wcss = []
scores = []
for i in range(2,10):
  kmean = KMeans(n_clusters=i)
  y_pred = kmean.fit_predict(x_pca)
  wcss.append(kmean.inertia_)
  scores.append(silhouette_score(x_pca,y_pred))
plt.plot(range(2,10),wcss)
plt.title('Elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')


# In[190]:


plt.plot(range(2,10),scores)
plt.title('silhouette_score')
plt.xlabel('number of clusters')
plt.ylabel('silhouette_score')


# In[191]:


kmean = KMeans(n_clusters=3)
y_pred_pca = kmean.fit_predict(x_pca)


# In[192]:


plt.figure(figsize=[10, 6])
plt.scatter(x_pca[y_pred_pca==0,0],x_pca[y_pred_pca==0,1],c = 'red',label = 'cluster1')
plt.scatter(x_pca[y_pred_pca==1,0],x_pca[y_pred_pca==1,1],c = 'green',label = 'cluster2')
plt.scatter(x_pca[y_pred_pca==2,0],x_pca[y_pred_pca==2,1],c = 'blue',label = 'cluster3')
plt.scatter(kmean.cluster_centers_[:,0],kmean.cluster_centers_[:,1],c='yellow',s=100,label='Centroids')
plt.title('Customers Clusters with PCA')
plt.legend()
sns.despine()


# In[193]:


#Pipeline¶
#Prepare Features
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline_cluster = Pipeline(steps=[('Feature Scaling', StandardScaler(with_mean=False))])


# In[194]:


from sklearn.compose import ColumnTransformer

preprocessing_cluster = ColumnTransformer(transformers= [('Numerical Columns', numerical_pipeline_cluster, cluster_pipeline.columns)], 
                                          remainder= 'passthrough')
preprocessing_cluster


# In[195]:


final_pipeline_cluster = Pipeline(steps=[('Preprocessing', preprocessing_cluster), ('Log Transformer', FunctionTransformer(np.log10)),
                                ('Model', KMeans(n_clusters= 3))])
final_pipeline_cluster


# In[196]:


# Fit pipeline to Dataframe
final_pipeline_cluster.fit(cluster_pipeline)


# In[197]:


# Save model as bkl file
import joblib
joblib.dump(final_pipeline_cluster, 'Brazilian Ecommerce Clustering.bkl')


# In[198]:


final_pipeline_cluster


# In[202]:


#Model Deployment¶
model_classification = joblib.load('Brazilian Ecommerce Classification.bkl')
model_clustering = joblib.load('Brazilian Ecommerce Clustering.bkl')


# In[203]:


#Test Classification Model¶
model_classification.predict(pd.DataFrame({'freight_value' :[30], 'product_description_lenght' :[1000], 'product_photos_qty' :[4], 'payment_type' :['credit_card'], 'payment_installments' :[6], 'payment_value' :[1000], 'estimated_days' :[7], 'arrival_days' :[6], 'arrival_status' :['OnTime/Early'], 'seller_to_carrier_status' :['OnTime/Early'], 'estimated_delivery_rate' :['Very Fast'], 'arrival_delivery_rate' :['Very Fast'], 'shipping_delivery_rate' :['Very Fast']}))


# In[204]:


#Test Clustering Model
model_clustering.predict(pd.DataFrame({'freight_value' :[10], 'price' :[90], 'payment_value' :[100], 'payment_installments' :[10], 'payment_sequential' :[3]}))


# In[205]:


# Install neccessary libraries for deployment

get_ipython().system(' pip install ydata_profiling')
get_ipython().system(' pip install streamlit_pandas_profiling')


# In[208]:


get_ipython().run_cell_magic('writefile', 'Brazilian_Ecommerce_Project.py', 'import numpy as np\nimport pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nimport joblib\nimport streamlit as st\nfrom sklearn.preprocessing import  StandardScaler\nfrom sklearn.cluster import KMeans\nfrom sklearn.cluster import AgglomerativeClustering\nfrom sklearn.decomposition import PCA\nfrom ydata_profiling import ProfileReport\nfrom streamlit_pandas_profiling import st_profile_report\n\n# Load Classification and Clustering Pipeline models\nmodel_classification = joblib.load(\'Brazilian Ecommerce Classification.bkl\')\nmodel_clustering = joblib.load(\'Brazilian Ecommerce Clustering.bkl\')\n\n# Create Sidebar to navigate between EDA, Classification and Clustering\nsidebar = st.sidebar\nmode = sidebar.radio(\'Mode\', [\'EDA\', \'Classification\', \'Clustering\'])\nst.markdown("<h1 style=\'text-align: center; color: #ff0000;\'></h1>", unsafe_allow_html=True)\n\nif mode == "EDA":\n\n    def main():\n\n        # Header of Customer Satisfaction Prediction\n        html_temp="""\n                    <div style="background-color:#F5F5F5">\n                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>\n                    </div>\n                """\n        # Create sidebar to upload CSV files\n        with st.sidebar.header(\'Upload your CSV data\'):\n            uploaded_file = st.sidebar.file_uploader(\'Upload your input csv file\')\n\n        if uploaded_file is not None:\n            # Read file and Put headers\n            EDA_sample = pd.read_csv(uploaded_file, index_col= 0)\n            pr = ProfileReport(EDA_sample, explorative=True)\n            st.header(\'**Input DataFrame**\')\n            st.write(EDA_sample)\n            st.write(\'---\')\n            st.header(\'**Pandas Profiling Report**\')\n            st_profile_report(pr)\n        \n        else:\n            st.info(\'Awaiting for CSV file to be uploaded.\')\n\n    if __name__ == \'__main__\':\n        main()\n\nif mode == "Classification":\n\n    # Define function to predict classification based on assigned features\n    def predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, \n    estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate):\n\n        prediction_classification = model_classification.predict(pd.DataFrame({\'freight_value\' :[freight_value], \'product_description_lenght\' :[product_description_lenght], \'product_photos_qty\' :[product_photos_qty], \'payment_type\' :[payment_type], \'payment_installments\' :[payment_installments], \'payment_value\' :[payment_value], \'estimated_days\' :[estimated_days], \'arrival_days\' :[arrival_days], \'arrival_status\' :[arrival_status], \'seller_to_carrier_status\' :[seller_to_carrier_status], \'estimated_delivery_rate\' :[estimated_delivery_rate], \'arrival_delivery_rate\' :[arrival_delivery_rate], \'shipping_delivery_rate\' :[shipping_delivery_rate]}))\n        return prediction_classification\n\n    def main():\n\n        # Header of Customer Satisfaction Prediction\n        html_temp="""\n                    <div style="background-color:#F5F5F5">\n                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>\n                    </div>\n                """\n        st.markdown(html_temp,unsafe_allow_html=True)\n        \n        # Assign all features with desired data input method\n        sidebar.title(\'Numerical Features\')\n        product_description_lenght = sidebar.slider(\'product_description_lenght\', 4,3990,100)\n        product_photos_qty = sidebar.slider(\'product_photos_qty\', 1,20,1)\n        payment_installments = sidebar.slider(\'payment_installments\', 1,24,1)\n        estimated_days = sidebar.slider(\'estimated_days\', 3,60,1)\n        arrival_days = sidebar.slider(\'arrival_days\', 0,60,1)\n        payment_type = st.selectbox(\'payment_type\', [\'credit_card\', \'boleto\', \'voucher\', \'debit_card\'])\n        arrival_status = st.selectbox(\'arrival_status\', [\'OnTime/Early\', \'Late\'])\n        seller_to_carrier_status = st.selectbox(\'seller_to_carrier_status\', [\'OnTime/Early\', \'Late\'])\n        estimated_delivery_rate = st.selectbox(\'estimated_delivery_rate\', [\'Very Slow\', \'Slow\', \'Neutral\', \'Fast\', \'Very Fast\'])\n        arrival_delivery_rate = st.selectbox(\'arrival_delivery_rate\', [\'Very Slow\', \'Slow\', \'Neutral\', \'Fast\', \'Very Fast\'])\n        shipping_delivery_rate = st.selectbox(\'shipping_delivery_rate Date\', [\'Very Slow\', \'Slow\', \'Neutral\', \'Fast\', \'Very Fast\'])\n        payment_value = st.text_input(\'payment_value\', \'\')\n        freight_value = st.text_input(\'freight_value\', \'\')\n        result = \'\'\n\n        # Predict Customer Satsifaction\n        if st.button(\'Predict_Satisfaction\'):\n            result = predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, \n                                        estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate)\n                                        \n        if result == 0:\n            result = \'Not Satisfied\'\n            st.success(f\'The Customer is {result}\')\n        else:\n            result = \'Satisfied\'\n            st.success(f\'The Customer is {result}\')\n\n    if __name__ == \'__main__\':\n        main()\n\nif mode == "Clustering":\n\n    def predict_clustering(freight_value, price, payment_value, payment_installments, payment_sequential):\n\n        prediction_clustering = model_clustering.predict(pd.DataFrame({\'freight_value\' :[freight_value], \'price\' :[price], \'payment_installments\' :[payment_installments], \'payment_value\' :[payment_value], \'payment_sequential\' :[payment_sequential]}))\n        return prediction_clustering\n\n    def main():\n\n        # Header of Customer Segmentation\n        html_temp="""\n                <div style="background-color:#F5F5F5">\n                <h1 style="color:#31333F;text-align:center;"> Customer Segmentation </h1>\n                </div>\n            """\n        st.markdown(html_temp,unsafe_allow_html=True)\n\n        # Assign all features with desired data input method\n        payment_installments = st.slider(\'payment_installments\', 1,24,1)\n        payment_sequential = st.slider(\'payment_sequential\', 1,24,1)\n        freight_value = st.text_input(\'freight_value\', \'\')\n        price = st.text_input(\'price\', \'\')\n        payment_value = st.text_input(\'payment_value\', \'\')\n        result_cluster = \'\'\n\n        # Predict Cluster of the customer\n        if st.button(\'Predict_Cluster\'):\n            result_cluster = predict_clustering(freight_value, price, payment_value, payment_installments, payment_sequential)\n                                        \n        st.success(f\'Customer Cluster is {result_cluster}\')\n        \n        # Upload CSV file\n        with st.sidebar.header(\'Upload your CSV data\'):\n            uploaded_file = st.sidebar.file_uploader(\'Upload your input csv file\')\n\n        if uploaded_file is not None:\n\n            # Read dataset\n            sample = pd.read_csv(uploaded_file, index_col= 0)\n            \n            # Define sidebar for clustering algorithm\n            selected_algorithm = sidebar.selectbox(\'Select Clustering Algorithm\', [\'K-Means\', \'Agglomerative\'])\n\n            # Define sidebar for number of clusters\n            selected_clusters = sidebar.slider(\'Select number of clusters\', 2, 10, 1)\n\n            # Define sidebar for PCA\n            use_pca = sidebar.radio(\'Use PCA\', [\'No\', \'Yes\'])\n\n            # Drop freight values with zeros\n            sample.drop(sample[sample.freight_value == 0].index, inplace= True)\n            # Reset Index \n            sample.reset_index(inplace= True, drop= True)\n            # Handle Skeweness in sample data\n            for i in [\'freight_value\', \'price\', \'payment_value\', \'payment_installments\', \'payment_sequential\']:\n                sample[i] = np.log10(sample[i])\n\n            # Apply standard scaler\n            sc = StandardScaler(with_mean= False)\n            data_scaled = sc.fit_transform(sample)\n\n            # Select number of clusters\n            if selected_algorithm == \'Agglomerative\':\n                hc = AgglomerativeClustering(n_clusters= selected_clusters)\n                y_pred_hc = hc.fit_predict(data_scaled)\n\n            else:\n                kmean = KMeans(n_clusters= selected_clusters)\n                y_pred_kmean = kmean.fit_predict(data_scaled)\n\n            # Apply PCA\n            pca = PCA(n_components= 2)\n            data_pca = pca.fit_transform(data_scaled)\n\n            # Select number of clusters for PCA\n            kmean_pca = KMeans(n_clusters= selected_clusters)\n            y_pred_pca = kmean_pca.fit_predict(data_pca)\n\n            def plot_cluster(data, y_pred, num_clusters):\n\n                # Plot Clusters\n                fig, ax = plt.subplots()\n                Colors= [\'red\', \'green\', \'blue\', \'purple\', \'orange\', \'royalblue\', \'brown\', \'grey\', \'chocolate\', \'fuchsia\']\n                for i in range(num_clusters):\n                    ax.scatter(data[y_pred==i,0], data[y_pred==i,1], c= Colors[i], label= \'Cluster \' + str(i+1))\n\n                ax.set_title(\'Customers Clusters\')\n                ax.legend(loc=\'upper left\', prop={\'size\':5})\n                ax.axis(\'off\')\n                st.pyplot(fig)\n\n            # Option to select and plot PCA for clustering\n            if use_pca == \'No\' and selected_algorithm == \'K-Means\':\n                plot_cluster(data_scaled, y_pred_kmean, selected_clusters)\n\n            elif use_pca == \'No\' and selected_algorithm == \'Agglomerative\':\n                plot_cluster(data_scaled, y_pred_hc, selected_clusters)           \n\n            else:\n                plot_cluster(data_pca, y_pred_pca, selected_clusters)    \n        \n        else:\n            st.info(\'Awaiting for CSV file to be uploaded.\')\n\n    if __name__ == \'__main__\':\n        main()')


# In[ ]:


get_ipython().system(' streamlit run Brazilian_Ecommerce_Project.py')


# In[ ]:





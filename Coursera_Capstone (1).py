#!/usr/bin/env python
# coding: utf-8

# # Peer-graded Assignment / Week 3:
# # Segmenting and Clustering Neighborhoods in Toronto
# ### Part 1 - Building Dataframe / 10 Marks

# ##### Import Packages

# In[79]:


import pandas as pd
get_ipython().system('conda install --yes lxml')


# ##### Scrape daraframes and count the number of dataframes

# In[80]:


url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
df=pd.read_html(url)
print(len(df))


# ##### List format to pandas dataframe

# In[81]:


df=pd.DataFrame(df[0])
df.head()


# ##### Drop rows where Borough not assigned

# In[82]:


noBo=df[df['Borough'] == 'Not assigned'].index
df.drop(noBo, inplace=True)
df.head()


# ##### Check and combine if there's duplicate Postal Codes 

# In[83]:


df[df['Postal Code'].duplicated()]


# ##### Any cell that has a borough but a Not assigned neighborhood - Change the neighborhood to = the borough as per instruction

# In[84]:


df[df['Neighbourhood'] == 'Not assigned']


# ##### Dataframe Ready

# In[85]:


df=df.reset_index(drop=True)
df.head(12)


# ##### Dataframe Shape

# In[86]:


df.shape


# ### Part 2 - Get Geographical Coordinates / 2 Marks

# ##### Used csv file instead of Geocoder - Download csv and read it into pd

# In[87]:


get_ipython().system('wget -O geocode.csv https://cocl.us/Geospatial_data')
df_G_Code = pd.read_csv("geocode.csv")
df_G_Code.head()


# ##### Combine dataframe ('df_G_Code') with previous ('df').

# In[88]:


df_merged=pd.merge(df, df_G_Code, on='Postal Code')
df_merged.head()


# ### Part 3 - Explore and Cluster the Neighbourhoods in Toronto / 3 marks

# ##### Clustering the neighborhoods based on categories of common venues only with boroughs that contain the word Toronto

# In[89]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import requests
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
get_ipython().system(' pip install folium==0.5.0')
import folium


# ##### Leave only rows that has the word 'Toronto'

# In[90]:


df_toronto=df_merged[df_merged['Borough'].str.contains('Toronto')]
df_toronto.reset_index(drop=True, inplace=True)
df_toronto


# ##### Foursquare Credentials & Version

# In[91]:


CLIENT_ID='JZPSHB3FTUKUWT15U2X0YXJQZXCCYZFH1Z0QHZAOPL0GF10O'
CLIENT_SECRET='2WBVHEK3YYVMANARI0Z1KI01Q45OD305OXMINXMVOUUKROQU'
VERSION='20200311'


# ##### Extract categories of venues

# In[92]:


def get_category_type(row):
    try:
        categories_list=row['categories']
    except:
        categories_list=row['venue.categories']        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# ##### Explore each neighbourhood and get nearby venues

# In[93]:


LIMIT=100
radius=500

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        url='https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
        
        results=requests.get(url).json()['response']['groups'][0]['items']
        
        venues_list.append([(name, 
                             lat, 
                             lng,
                             v['venue']['name'],
                             v['venue']['location']['lat'], 
                             v['venue']['location']['lng'],
                             v['venue']['categories'][0]['name']) for v in results])
        
    nearby_venues=pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns=['Neighbourhood',
                           'Neighbourhood Latitude',
                           'Neighbourhood Longitude',
                           'Venue',
                           'Venue Latitude',
                           'Venue Longitude',
                           'Venue Category']
    return(nearby_venues)


# ##### Run 'getNearbyVenues' to each neighbourhood

# In[94]:


toronto_venues=getNearbyVenues(names=df_toronto['Neighbourhood'],
                                   latitudes=df_toronto['Latitude'],
                                   longitudes=df_toronto['Longitude'])


# In[95]:


print(toronto_venues.shape)
toronto_venues.head(15)


# ##### How many venues for each neighbourhood

# In[96]:


toronto_venues.groupby('Neighbourhood').count()


# ##### How many unique categories 

# In[97]:


print('{} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ##### Onehot encoding to venue categories

# In[98]:


toronto_onehot=pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.rename(columns={'Neighbourhood':'Neighbourhood (category)'}, inplace=True)
toronto_onehot['Neighbourhood']=toronto_venues['Neighbourhood'] 
fixed_columns=[toronto_onehot.columns[-1]]+list(toronto_onehot.columns[:-1])
toronto_onehot=toronto_onehot[fixed_columns]
toronto_onehot.head()


# ##### Group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[99]:


toronto_grouped=toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped.head()


# ##### Confirm new data

# In[100]:


toronto_grouped.shape


# ##### Here's each neighbourhood along with the top 5 most common venues

# In[101]:


num_top_venues = 5
for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# ##### Sort venues in descending order

# In[102]:


def return_most_common_venues(row, num_top_venues):
    row_categories=row.iloc[1:]
    row_categories_sorted=row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]


# ##### Top 10 venues for each neighbourhood

# In[103]:


num_top_venues=10
indicators=['st', 'nd', 'rd']
columns=['Neighbourhood']

for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood']=toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:]=return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)
neighbourhoods_venues_sorted.head()


# ##### K-means to cluster the neighbourhoods into 5 clusters

# In[104]:


kclusters=5
toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)
kmeans=KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)
kmeans.labels_[0:10]


# ##### Build new dataframe including the cluster & the top 10 venues for each neighbourhood

# In[105]:


neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
toronto_merged=df_toronto
toronto_merged=toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')
toronto_merged.head()


# ##### Visualise resulting clusters

# In[106]:


toronto_coordinates =[43.6532, -79.3832]
map_clusters=folium.Map(location=toronto_coordinates, zoom_start=12)

x=np.arange(kclusters)
ys=[i + x + (i*x)**2 for i in range(kclusters)]
colors_array=cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow=[colors.rgb2hex(i) for i in colors_array]

markers_colors=[]
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label=folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
    
map_clusters


# ### Analysis
# ##### Most of neighbourhoods in Toronto are in cluster 2 (Blue)
# ##### In cluster 2 we have lots of places to enjoy a good coffee and quick meals

# In[107]:


print('Cluster 0')
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[108]:


print('Cluster 1')
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[109]:


print('Cluster 2')
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[110]:


print('Cluster 3')
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[111]:


print('Cluster 4')
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids  
import collections
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

DATA = '../yelp_dataset/'
RAW_DATA = '../yelp_dataset/raw/'
JOIN_TABLE = '../yelp_dataset/join_table/'
RESTAURANT_DATA = '../yelp_dataset/restaurant_data/'
FEATURE_DATA = '../yelp_dataset/features/'
rest_checkin_csv = JOIN_TABLE + 'restaurant_checkin.csv'
RESULT = '../result/'
FILTERED_BIZ_CAT = '../result/root/business_category/'
START = 5
END = 21
STEP = 1
def get_rest_tags():
    rest_checkin = pd.read_csv(rest_checkin_csv,dtype = object,index_col = 0)
    biz_id = rest_checkin['business_id'].to_list()
    rest_df = pd.read_csv(RESTAURANT_DATA+"restaurant.csv")
    print(rest_df)
    rest_df = rest_df.loc[rest_df['business_id'].isin(biz_id)]
    rest_df = rest_df.loc[rest_df['review_count']>10]
    rest_tags = rest_df[['business_id','categories']]
    rest_tags.set_index('business_id',drop= True, inplace = True)
    # print(type(rest_tags.loc['fnZrZlqW1Z8iWgTVDfv_MA']['categories']))
    # 1 or ‘columns’: apply function to each row.
    rest_tags['processed_tags'] = rest_tags.apply(lambda x: x['categories'].split(', '),axis = 1)
    # print(rest_tags)
    # print(rest_tags)
    unique_tags =list(set(tag for r in rest_tags['processed_tags'].to_list() for tag in r))
    unique_tags.remove('restaurants')
    unique_tags.remove('food')
    tags_df = pd.DataFrame(unique_tags)
    tags_df.to_csv(FEATURE_DATA+"business_tags.csv")
    rest_tags.to_csv(RESTAURANT_DATA+"rest_biz_tags.csv")
    return unique_tags, rest_tags

def get_rest_tag_vec(unique_tags, rest_tags):
    rest_id_ls = rest_tags.index.values.tolist()
    rest_tag_df = pd.DataFrame(0,columns = unique_tags, index = rest_id_ls)
    for rest_id in rest_id_ls:
        tags = rest_tags.loc[rest_id]['processed_tags']
        for tag in tags:
            rest_tag_df.loc[rest_id][tag] = 1

    print(rest_tag_df)
    # drop rows that don't have any relevant business tags
    rest_tag_df = rest_tag_df.loc[(rest_tag_df != 0).any(axis=1)]
    print(rest_tag_df)
    return rest_tag_df


def k_means_rest_root(rest_tag_df,NUM_CLUSTERS):
    # initialize 
    pca_rest = pca(rest_tag_df)
    size_of_list = int((END -START)/STEP)
    y_kmeans = np.zeros((rest_tag_df.shape[0],size_of_list))
    sse_res_ls = []
    sil_score_ls = []
    distortions = []
    dbIndex=[]
    # for k in range(START,END,STEP):
    #     kmeans_cluster = KMeans(n_clusters=k,random_state=42).fit(pca_rest)
    #     # kmeans_cluster = KMedoids(n_clusters=k,metric = "cosine",random_state=42).fit(pca_rest)
    #     idx = int((k-START)/STEP)
    #     y_kmeans[:,idx] = kmeans_cluster.predict(pca_rest)
    #     centroids = kmeans_cluster.cluster_centers_
    #     sse_res_ls.append(compute_sse(pca_rest,y_kmeans[:,idx],centroids, k))
    #     sil_score_ls.append(get_sil_score(pca_rest,y_kmeans[:,idx]))
    #     distortions.append(sum(np.min(cdist(pca_rest, centroids, 'euclidean'), axis=1)) / pca_rest.shape[0])
    #     # distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
    #     dbIndex.append(davies_bouldin_score(pca_rest,y_kmeans[:,idx]))

    #     # print(distortions)
    # print('distortions: ')    
    # print(distortions)
    # print('dbIndex: ') 
    # print(dbIndex)
    # print('silhouette score: ') 
    # print(sil_score_ls)
    # plot_dbIndex(dbIndex)
    # plot_elbow(distortions)
    # plot_sse(sse_res_ls)
    # plot_sil_score(sil_score_ls)
   
    


    # kmeans_cluster = KMedoids(n_clusters=28,metric = "cosine",random_state=42).fit(pca_rest)
    kmeans_cluster = KMeans(n_clusters=NUM_CLUSTERS,random_state=42).fit(pca_rest)
    category = kmeans_cluster.predict(pca_rest)
    rest_cat = pd.DataFrame(category, columns=['category'],index = rest_tag_df.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(RESTAURANT_DATA+'restaurant.csv')
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    sil_score =  get_sil_score(pca_rest, category)
    print(sil_score)
    # print(result)
    result.to_csv(FILTERED_BIZ_CAT+'restaurant_root_cat_kmean_filtered_'+str(NUM_CLUSTERS)+'.csv')
    distribution = result.groupby('category').size().to_frame()
    distribution.to_csv(FILTERED_BIZ_CAT+'distribution_root_cat_kmean_filtered_'+str(NUM_CLUSTERS)+'.csv')

def EM(rest_tag_df):
    y_em = np.zeros((rest_tag_df.shape[0],11))
    
    sil_score_ls = []
    sse_res_ls = []
    distortions = []
    dbIndex=[]

    pca_rest = pca(rest_tag_df)

    for k in range(20,42,2):
        step = int((k-20)/2)
        y_em[:,step] = GaussianMixture(n_components=k,random_state=42).fit_predict(pca_rest)
        # compute silhouette coefficient via original data and predicted clusters
        sil_score_ls.append(get_sil_score(pca_rest,y_em[:,step]))
        dbIndex.append(davies_bouldin_score(pca_rest,y_em[:,step]))
        print(sil_score_ls)
    plot_dbIndex(dbIndex)
    category = GaussianMixture(n_components=38,random_state=42).fit_predict(pca_rest)
    rest_cat = pd.DataFrame(category, columns=['category'],index = rest_tag_df.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(RESTAURANT_DATA+'restaurant.csv')
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    # sil_score = silhouette_score(pca_rest, category, metric='euclidean')
    # print(sil_score)
    # print(result)
    result.to_csv(RESULT+'restaurant_root_cat_gmm.csv')
    distribution = result.groupby('category').size().to_frame()
    distribution.to_csv(RESULT+'distribution_root_cat_gmm.csv')
       
def hierarchy_rest_root(rest_tag_df):
    # initialize 
    pca_rest = pca(rest_tag_df)
    y_hierarchy = np.zeros((rest_tag_df.shape[0],16))
    sse_res_ls = []
    sil_score_ls = []
    distortions = []
    dbIndex=[]
    # for k in range(20,52,2):
    #     step = int((k-20)/2)
    #     y_hierarchy[:,step] = AgglomerativeClustering(n_clusters=k,affinity= 'cosine',linkage='average').fit_predict(pca_rest)
    #     sil_score_ls.append(get_sil_score(pca_rest,y_hierarchy[:,step]))
    #     # distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
    #     dbIndex.append(davies_bouldin_score(pca_rest,y_hierarchy[:,step]))
    # print(dbIndex)
    # print(sil_score_ls)
    # plot_dbIndex(dbIndex)
    # plot_sil_score(sil_score_ls)
   

    model = AgglomerativeClustering(n_clusters=38,affinity= 'cosine',linkage='average')
    '''
    plot the dendrogram
    model = model.fit(pca_rest)
    plt.title('Hierarchical Clustering Dendrogram')
    plot the top three levels of the dendrogram
    plot_dendrogram(model, labels=model.labels_)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    '''
    category = model.fit_predict(pca_rest)
    rest_cat = pd.DataFrame(category, columns=['category'],index = rest_tag_df.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(RESTAURANT_DATA+'restaurant.csv')
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    sil_score = silhouette_score(pca_rest, category, metric='cosine')
    print(sil_score)
    # print(result)
    result.to_csv(FILTERED_BIZ_CAT+'restaurant_root_cat_hierarchy_filtered_38.csv')
    distribution = result.groupby('category').size().to_frame()
    distribution.to_csv(FILTERED_BIZ_CAT+'distribution_root_cat_hierarchy_filtered_38.csv')



def pca(data):
    norm_data = StandardScaler().fit_transform(data)

    #Plotting the Cumulative Summation of the Explained Variance
    # plot_pca(norm_data)
    pca = PCA(0.8)
    pca_data = pca.fit_transform(norm_data)
    print(pca.n_components_)
    return pca_data

def compute_sse(test, y, centroids,K):
    distance = np.zeros(test.shape[0])
    for k in range(K):
        distance[y == k] = np.linalg.norm(test[y == k] - centroids[k], axis=1)  
        sse = np.sum(np.square(distance))
    return sse
def get_sil_score(data, predicted):
    
    score = silhouette_score(data, predicted, metric='euclidean')
    return score

def plot_pca(norm_data):
    pca = PCA().fit(norm_data)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Yelp Dataset PCA Explained Variance')
    plt.show()

def plot_sse(sse_res):
    plt.figure()
    filename="sse_root_kmean_filtered"
    x = list(range(START,END,STEP))
    plt.plot(x, sse_res,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, STEP))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig(filename)

def plot_elbow(distortions):
    plt.figure()
    filename="elbow_root_kmean_filtered"
    x = list(range(START,END,STEP))
    plt.plot(x, distortions, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, STEP))
    plt.xlabel('K')
    plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)

def plot_dbIndex(dbIndex):
    plt.figure()
    filename = "dbIndex_root_kmean_filtered"
    x = list(range(START,END,STEP))
    plt.plot(x, dbIndex,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, STEP))
    plt.xlabel("K")
    plt.ylabel("DBIndex")
    plt.savefig(filename)

def plot_sil_score(sil_score):
    plt.figure()
    filename = "sil_score_root_kmean_filtered"
    x = list(range(START,END,STEP))
    plt.plot(x, sil_score,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, STEP))
    plt.xlabel("K")
    plt.ylabel("silhouette score")
    plt.savefig(filename)

def remove_business(removed_categories):
    rest_root_cat = pd.read_csv(RESULT+'root/outlier/restaurant_root_cat_kmean.csv',index_col = 0)
    filtered_rest = rest_root_cat.loc[~rest_root_cat['category'].isin(removed_categories)]
    filtered_rest['processed_tags'] = filtered_rest.apply(lambda x: x['categories'].split(', '),axis = 1)
    filtered_rest.set_index('business_id',drop = True, inplace = True)
    print(filtered_rest) 
    return filtered_rest   

def get_filtered_tags():
    filtered_tags = []
    with open(FEATURE_DATA + 'featured_biz_tags.txt','r') as f:
        for line in f:
            # remove the line reak
            current_tag = line[:-1]
            # add feature to the list
            filtered_tags.append(current_tag)
    # print(filtered_tags)
    return filtered_tags

def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#  
'''
functions for detecting business outliers, where K = 32
'''
# unique_tags, rest_tags = get_rest_tags()
# rest_tag_df = get_rest_tag_vec(unique_tags, rest_tags)
# k_means_rest_root(rest_tag_df, NUM_CLUSTERS = 32)

'''
other models:
# EM(rest_tag_df)
# hierarchy_rest_root(rest_tag_df)
# check the results of hierarchy_rest_root manually to get the categories that are not related to this project
# removed_categories_hierachy = [24,26,11,18,19,27,16,21,3,13,8,22,29,10,7,4,15]
'''

'''
functions for categorizing restaurants based on business tags, where K = 15
'''
removed_categories_kmean = [20, 27, 28, 30, 16, 26, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 21, 22, 24, 31]
filtered_rest = remove_business(removed_categories_kmean)
filtered_tags = get_filtered_tags()
filtered_rest_vec = get_rest_tag_vec(filtered_tags, filtered_rest)
k_means_rest_root(filtered_rest_vec, NUM_CLUSTERS = 15)

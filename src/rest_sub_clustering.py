import pandas as pd
import numpy as np
import scipy
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids  
from spherecluster import SphericalKMeans

RAW_DATA = '../yelp_dataset/raw/'
JOIN_TABLE = '../yelp_dataset/join_table/'
RESTAURANT_DATA = '../yelp_dataset/restaurant_data/'
FEATURE_DATA = '../yelp_dataset/features/'
RESULT = '../result/'

def get_features():
    word_features = []
    with open(FEATURE_DATA + 'features_review.txt','r') as f:
        for line in f:
            # remove the line reak
            current_word = line[:-1]
            # add feature to the list
            word_features.append(current_word)
    # remove duplicate feature
    features = list(set(word_features))
    # print(features)
    # print(len(features))
    return features


def get_restaurant_reviews():
    rest_checkin = pd.read_csv(JOIN_TABLE + 'restaurant_checkin.csv',dtype = object,index_col = 0)
    rest_root_cat_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv')
    rest_root_cat_df = rest_root_cat_df[['business_id','category']]
    # convert processed text to list
    rest_review = pd.read_csv(RESTAURANT_DATA + 'processed_review.csv', index_col = 0, converters={'text': lambda n: n[1:-1].replace("'","").split(', ')}) 
    print(rest_review.shape[0])
    # get restaurants that have check-in data avaliable
    biz_id_checkin = rest_checkin['business_id'].to_list()
    rest_review = rest_review.loc[rest_review['business_id'].isin(biz_id_checkin)]
    print(rest_review.shape[0])
    word_dict = get_features()
    rest_word_freq_df = pd.DataFrame(columns = word_dict)
    for rest_id in biz_id_checkin:
        rest_txt_df = rest_review.loc[rest_review['business_id'] == rest_id]
        df = get_word_freq_per_rest(rest_txt_df,rest_id,word_dict)
        rest_word_freq_df = pd.concat([rest_word_freq_df,df])
    
    print(rest_word_freq_df)
    rest_word_freq_df.to_csv(RESTAURANT_DATA+'sub_rest_word_freq.csv')

def get_word_freq_per_rest(rest_txt_df,rest_id,word_dict):
    processed_txt = rest_txt_df['text'].to_list()

    word_freq_df = pd.DataFrame(0, index=np.arange(0, 1),columns = word_dict)
    for review_txt in processed_txt:
        for w in review_txt:
            if w in word_dict: 
                word_freq_df.iloc[0][w] = word_freq_df.iloc[0][w] + 1
    word_freq_df.rename({0:rest_id},inplace = True,axis='index')
    print(word_freq_df)
    return word_freq_df

def add_price_att():
    rest_word_feq_df = pd.read_csv(RESTAURANT_DATA+'sub_rest_word_freq.csv',index_col = 0)
    restaurant_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv')
    
    rest_word_feq_df = rest_word_feq_df.apply(pd.to_numeric, errors='coerce')
    # idx =set(rest_word_feq_df.index.values.tolist())
    # biz_id = set(restaurant_df['business_id'].tolist())
    # intersect_id = list(idx.intersection(biz_id))
    
    # rest_id = rest_tf_idf.index.values.tolist()
    

    # restaurant_df = restaurant_df.loc[restaurant_df['business_id'].isin(rest_id)]
    restaurant_df.set_index('business_id',inplace = True)
    rest_root_cat = restaurant_df[['category']]
    root_categories = rest_root_cat['category'].tolist()
    root_categories = list(set(root_categories))
    rest_tf_idf = pd.DataFrame()
    for root_cat in root_categories:
        biz_ls = rest_root_cat.index[rest_root_cat['category'] == root_cat].tolist()
        df = get_tf_idf(rest_word_feq_df.loc[biz_ls])
        rest_tf_idf = rest_tf_idf.append(df)

    
    rest_price = restaurant_df['attributes.RestaurantsPriceRange2']
    rest_bag_of_word_price = pd.concat([rest_tf_idf,rest_price], axis = 1)
    rest_bag_of_word_price.rename(columns = {'attributes.RestaurantsPriceRange2':'RestaurantsPriceRange2'}, inplace = True) 
    # fill na price range with mean range
    rest_bag_of_word_price['RestaurantsPriceRange2'].replace('None', np.nan, inplace=True)
    rest_bag_of_word_price['RestaurantsPriceRange2'] = rest_bag_of_word_price['RestaurantsPriceRange2'].astype(float)
    rest_bag_of_word_price['RestaurantsPriceRange2'].fillna(round(rest_bag_of_word_price['RestaurantsPriceRange2'].mean()),inplace = True)
    # convert to int
    rest_bag_of_word_price['RestaurantsPriceRange2'] = rest_bag_of_word_price['RestaurantsPriceRange2'].astype(float).astype(int)
    print(rest_bag_of_word_price)
    rest_bag_of_word_price.to_csv(RESTAURANT_DATA + 'sub_rest_bag_of_words.csv')
    rest_tf_idf.to_csv(RESTAURANT_DATA + 'sub_rest_tf_idf.csv')
    return rest_bag_of_word_price

def get_tf_idf(rest_word_feq_df):
    # tf
    # tf = rest_word_feq_df.apply(lambda x: get_term_freq(x))
    rest_word_feq_df = rest_word_feq_df.fillna(0)
    tf = rest_word_feq_df.div(rest_word_feq_df.sum(axis = 1),axis = 0)
    # tf = tf.div(tf.sum(axis = 1),axis = 0)
    # tf = tf.round(4)
    tf.dropna(inplace = True)
    tf[tf.columns] = np.log(tf[tf.columns]+1)
    # idf   
    rest_word_feq_df = rest_word_feq_df.loc[tf.index]
    idf = rest_word_feq_df
    idf = np.log((rest_word_feq_df.shape[0])/(rest_word_feq_df.astype(bool).sum(axis=0)+1))
    # idf[idf.columns] = (rest_word_feq_df.shape[0])/ (idf[idf.columns] +1)
    # idf[idf.columns] = np.log(idf[idf.columns])
    # print(idf)
    rest_tf_idf = pd.DataFrame(tf.values * idf.values, columns = rest_word_feq_df.columns, index = rest_word_feq_df.index)
    # print(rest_tf_idf[rest_tf_idf.isna().any(axis=1)])
    # print(rest_tf_idf)
    
    return rest_tf_idf

def k_means_sub_cat(rest_bag_of_word_price, num_clusters_ls):
    restaurant_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv')
    restaurant_df.set_index('business_id',inplace = True)
    restaurant_df.rename(columns = {'category':'root_category'}, inplace = True)
    rest_root_cat = restaurant_df[['root_category']]
    
    root_categories = rest_root_cat['root_category'].tolist()
    root_categories = list(set(root_categories))

    rest_bag_of_word_price = pd.concat( [rest_bag_of_word_price,rest_root_cat] , axis = 1)
    sub_rest_cat_df = pd.DataFrame()
    for root_cat in root_categories:
        num_clusters = num_clusters_ls[root_cat]
        root_rest_bag_of_word_price =  rest_bag_of_word_price[rest_bag_of_word_price['root_category'] == root_cat]
        # test_k_means_rest(root_rest_bag_of_word_price, root_cat)
        # test_spherical_kmeans(root_rest_bag_of_word_price,root_cat)
        # df = k_means_rest(root_rest_bag_of_word_price, root_cat,num_clusters)
        df = spherical_k_means_rest(root_rest_bag_of_word_price, root_cat,num_clusters)
        sub_rest_cat_df = sub_rest_cat_df.append(df)

    sub_rest_cat_df.to_csv(RESULT + 'sub/restaurant_sub_cat_sk.csv')

def test_spherical_kmeans(root_rest_bag_of_word_price, root_cat):
    print(root_cat)
    # initialize 
    pca_rest = pca(root_rest_bag_of_word_price)
    # pca_rest = StandardScaler().fit_transform(root_rest_bag_of_word_price)
    num_samples = root_rest_bag_of_word_price.shape[0]
    
    if num_samples <= 2:
        return 
    elif num_samples -1 <90:
        y_kmeans = np.zeros((num_samples, num_samples -1))
        max_cluster =int ((num_samples -1) /3) 
    else: 
        y_kmeans = np.zeros((num_samples, 29))
        max_cluster = 30
    sse_res_ls = []
    sil_score_ls = []
    distortions = []
    dbIndex=[]

    for k in range(2,max_cluster +1 ,1):
        spherical_kmeans_cluster = SphericalKMeans(n_clusters=k,random_state=42)
        spherical_kmeans_cluster.fit(pca_rest)
        # kmeans_cluster = KMedoids(n_clusters=k,metric = "cosine",random_state=42).fit(pca_rest)
        step = int((k-2)/1)
        y_kmeans[:,step] = spherical_kmeans_cluster.predict(pca_rest)
        centroids = spherical_kmeans_cluster.cluster_centers_
        sse_res_ls.append(compute_sse(pca_rest,y_kmeans[:,step],centroids, k))
        sil_score_ls.append(get_sil_score(pca_rest,y_kmeans[:,step]))
        distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
        dbIndex.append(davies_bouldin_score(pca_rest,y_kmeans[:,step]))
        # distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
        # print(distortions)
    # print('distortions: ')    
    # print(distortions)
    # print('dbIndex: ') 
    # print(dbIndex)
    plot_dbIndex(dbIndex, root_cat, max_cluster)
    plot_elbow(distortions, root_cat, max_cluster)
    plot_sse(sse_res_ls, root_cat, max_cluster)
    plot_sil_score(sil_score_ls, root_cat, max_cluster)
def test_k_means_rest(root_rest_bag_of_word_price, root_cat):
    print(root_cat)
    # initialize 
    pca_rest = pca(root_rest_bag_of_word_price)
    # pca_rest = StandardScaler().fit_transform(root_rest_bag_of_word_price)
    num_samples = root_rest_bag_of_word_price.shape[0]
    
    if num_samples <= 2:
        return 
    elif num_samples -1 <90:
        y_kmeans = np.zeros((num_samples, num_samples -1))
        max_cluster =int ((num_samples -1) /3) 
    else: 
        y_kmeans = np.zeros((num_samples, 29))
        max_cluster = 30
    sse_res_ls = []
    sil_score_ls = []
    distortions = []
    dbIndex=[]

    for k in range(2,max_cluster +1 ,1):
        kmeans_cluster = KMeans(n_clusters=k,random_state=42).fit(pca_rest)
        # kmeans_cluster = KMedoids(n_clusters=k,metric = "cosine",random_state=42).fit(pca_rest)
        step = int((k-2)/1)
        y_kmeans[:,step] = kmeans_cluster.predict(pca_rest)
        centroids = kmeans_cluster.cluster_centers_
        sse_res_ls.append(compute_sse(pca_rest,y_kmeans[:,step],centroids, k))
        sil_score_ls.append(get_sil_score(pca_rest,y_kmeans[:,step]))
        distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
        dbIndex.append(davies_bouldin_score(pca_rest,y_kmeans[:,step]))
        # distortions.append(sum(np.min(cdist(pca_rest, centroids, 'cosine'), axis=1)) / pca_rest.shape[0])
        # print(distortions)
    # print('distortions: ')    
    # print(distortions)
    # print('dbIndex: ') 
    # print(dbIndex)
    plot_dbIndex(dbIndex, root_cat, max_cluster)
    plot_elbow(distortions, root_cat, max_cluster)
    plot_sse(sse_res_ls, root_cat, max_cluster)
    plot_sil_score(sil_score_ls, root_cat, max_cluster)

def spherical_k_means_rest(root_rest_bag_of_word_price, root_cat, num_clussters):
    print(root_cat)
    # initialize 
    
    
    category = [0 for i in range(root_rest_bag_of_word_price.shape[0])]
    if num_clussters > 1: 
        pca_rest = pca(root_rest_bag_of_word_price)
        kmeans_cluster = SphericalKMeans(n_clusters=num_clussters,random_state=42).fit(pca_rest)
        category = kmeans_cluster.predict(pca_rest)
        category = category.tolist()
        sil_score = silhouette_score(pca_rest, category, metric='cosine')
        print(sil_score)
    sub_category = [str(root_cat) + "_" + str(sub) for sub in category] 
    rest_cat = pd.DataFrame(sub_category, columns=['sub_category'],index = root_rest_bag_of_word_price.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv', index_col=0)
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    
    # print(result)
    # result.to_csv(RESULT+'restaurant_cat.csv')
    return result

def k_means_rest(root_rest_bag_of_word_price, root_cat, num_clussters):
    print(root_cat)
    # initialize 
    
    
    category = [0 for i in range(root_rest_bag_of_word_price.shape[0])]
    if num_clussters > 1: 
        pca_rest = pca(root_rest_bag_of_word_price)
        kmeans_cluster = KMeans(n_clusters=num_clussters,random_state=42).fit(pca_rest)
        category = kmeans_cluster.predict(pca_rest)
        category = category.tolist()
        sil_score = silhouette_score(pca_rest, category, metric='euclidean')
        print(sil_score)
    sub_category = [str(root_cat) + "_" + str(sub) for sub in category] 
    rest_cat = pd.DataFrame(sub_category, columns=['sub_category'],index = root_rest_bag_of_word_price.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv', index_col=0)
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    
    # print(result)
    # result.to_csv(RESULT+'restaurant_cat.csv')
    return result

def pca(data):
    norm_data = StandardScaler().fit_transform(data)

    #Plotting the Cumulative Summation of the Explained Variance
    # plot_pca(norm_data)
    pca = PCA(0.85)
    pca_data = pca.fit_transform(norm_data)
    print(pca.n_components_)
    return pca_data
def plot_pca(norm_data):
    pca = PCA().fit(norm_data)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Yelp Dataset PCA Explained Variance')
    plt.show()

def compute_sse(test, y, centroids,K):
    distance = np.zeros(test.shape[0])
    for k in range(K):
        distance[y == k] = np.linalg.norm(test[y == k] - centroids[k], axis=1)  
        sse = np.sum(np.square(distance))
    return sse

def get_sil_score(data, predicted):
    
    score = silhouette_score(data, predicted, metric='cosine')
    return score

def plot_sil_score(sil_score_ls, root_cat, max_cluster):
    plt.figure()
    filename="../figure/sub/spherical_kmeans_silhouette_score" + str(root_cat)
    x = list(range(2,max_cluster+1,1))
    plt.plot(x, sil_score_ls, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel('K')
    plt.ylabel('Silhouette')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)

def plot_sse(sse_res, root_cat, max_cluster):
    plt.figure()
    filename="../figure/sub/spherical_k_means_sse_" + str(root_cat)
    x = list(range(2,max_cluster+1,1))
    plt.plot(x, sse_res,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig(filename)

def plot_elbow(distortions, root_cat, max_cluster):
    plt.figure()
    filename="../figure/sub/spherical_kmeans_elbow_" + str(root_cat)
    x = list(range(2,max_cluster+1,1))
    plt.plot(x, distortions, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel('K')
    plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)

def plot_dbIndex(dbIndex, root_cat, max_cluster):
    plt.figure()
    filename = "../figure/sub/spherical_kmeans_dbIndex_" + str(root_cat)
    x = list(range(2,max_cluster +1,1))
    plt.plot(x, dbIndex,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel("K")
    plt.ylabel("DBIndex")
    plt.savefig(filename)

def most_freq_word_cat():
    rest_cat = pd.read_csv(RESULT+"sub/restaurant_sub_cat_sk.csv")
    rest_cat = rest_cat[['business_id','sub_category']]
    rest_word_feq_df = pd.read_csv(RESTAURANT_DATA+'sub_rest_word_freq.csv',index_col = 0)
    rest_word_feq_df.reset_index(inplace= True)
    rest_word_feq_df.rename(columns = {'index':'business_id'},inplace = True)
    df = pd.merge(rest_cat, rest_word_feq_df, on='business_id')
    distribution = df.groupby('sub_category').size().to_frame()
    distribution.to_csv(RESULT+'distribution_sub_cat_sk.csv')
    features = list(rest_word_feq_df.columns)
    features.remove('business_id')
    word_cat = df.groupby(by = ['sub_category']).sum()
  
    word_cat =word_cat.transpose()
    most_freq_cat = pd.DataFrame(columns= ['most frequent words'],index = word_cat.columns)
    for col in word_cat.columns:
        most_freq = word_cat[col].argsort()
        freq_word_df = word_cat[col][most_freq].tail(25)
        fw_ls = freq_word_df.index.tolist()
        fw_ls.reverse()
        fw_str = ", ".join([word for word in fw_ls])
        most_freq_cat.loc[col] = fw_str
    print(most_freq_cat)
    most_freq_cat.to_csv(RESULT+'most_freq_sub_cat_sk.csv')
# get_features()
# get_restaurant_reviews()
rest_bag_of_word_price = add_price_att()
# num_clusters_ls = [5, 7, 5, 3, 2, 9, 6, 6, 6, 1, 15, 2, 4, 14, 6]
# num_clusters_ls = [5, 8, 6, 3, 2, 27, 3, 5, 3, 1, 6, 2, 4, 10, 6]
# num_clusters_ls = [2, 8, 1, 3, 1, 24, 3, 5, 3, 1, 6, 1, 1, 3, 6]
# num_clusters_ls = [2, 8, 1, 2, 1, 10, 3, 8, 2, 1, 5, 2, 1, 3, 8]
num_clusters_ls = [3, 9, 3, 3, 1, 6, 4, 5, 3, 1, 10, 2, 3, 6, 4]
k_means_sub_cat(rest_bag_of_word_price, num_clusters_ls)
most_freq_word_cat()



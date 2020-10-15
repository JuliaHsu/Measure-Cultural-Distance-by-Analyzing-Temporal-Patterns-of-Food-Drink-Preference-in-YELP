import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

DATA = '../yelp_dataset/'
RAW_DATA = '../yelp_dataset/raw/'
PROCESSED_DATA = '../yelp_dataset/processed_data/'
rest_checkin_csv = PROCESSED_DATA + 'restaurant_checkin.csv'
rest_review_csv =  PROCESSED_DATA + 'processed_review.csv'
RESULT = '../result/'
def get_word_dict():
    word_dict = []
    with open(DATA + 'features.txt','r') as f:
        for line in f:
            # remove the line reak
            current_word = line[:-1]
            # add feature to the list
            word_dict.append(current_word)
    # print(word_dict)
    return word_dict

def get_restaurant_reviews():
    rest_checkin = pd.read_csv(PROCESSED_DATA + 'restaurant_checkin.csv',dtype = object,index_col = 0)
    # convert processed text to list
    rest_review = pd.read_csv(PROCESSED_DATA + 'processed_review.csv', index_col = 0, converters={'processed_text': lambda n: n[1:-1].replace("'","").split(', ')}) 
    print(rest_review)
    # get restaurants that have check-in data avaliable
    biz_id_checkin = rest_checkin['business_id'].to_list()
    rest_review = rest_review.loc[rest_review['business_id'].isin(biz_id_checkin)]
    
    word_dict = get_word_dict()
    rest_word_freq_df = pd.DataFrame(columns = word_dict)
    for rest_id in biz_id_checkin:
        rest_txt_df = rest_review.loc[rest_review['business_id'] == rest_id]
        df = get_word_freq_per_rest(rest_txt_df,rest_id,word_dict)
        rest_word_freq_df = pd.concat([rest_word_freq_df,df])
    
    print(rest_word_freq_df)
    rest_word_freq_df.to_csv(PROCESSED_DATA+'rest_word_freq.csv')

def get_word_freq_per_rest(rest_txt_df,rest_id,word_dict):
    processed_txt = rest_txt_df['processed_text'].to_list()

    word_freq_df = pd.DataFrame(0, index=np.arange(0, 1),columns = word_dict)
    for review_txt in processed_txt:
        for w in review_txt:
            if w in word_dict: 
                word_freq_df.iloc[0][w] = word_freq_df.iloc[0][w] + 1
    word_freq_df.rename({0:rest_id},inplace = True,axis='index')
    print(word_freq_df)
    return word_freq_df

def add_price_att():
    rest_word_feq_df = pd.read_csv(PROCESSED_DATA+'rest_word_freq.csv',index_col = 0)
    rest_word_feq_df = rest_word_feq_df.apply(pd.to_numeric, errors='coerce')
    rest_tf_idf = get_tf_idf(rest_word_feq_df)
    rest_id = rest_tf_idf.index.values.tolist()

    restaurant_df = pd.read_csv(PROCESSED_DATA+'restaurant.csv')
    restaurant_df = restaurant_df.loc[restaurant_df['business_id'].isin(rest_id)]
    restaurant_df.set_index('business_id',inplace = True)
    rest_price = restaurant_df['attributes.RestaurantsPriceRange2']
    rest_bag_of_word_price = pd.concat([rest_tf_idf,rest_price], axis = 1)
    rest_bag_of_word_price.rename(columns = {'attributes.RestaurantsPriceRange2':'price'}, inplace = True)

    rest_bag_of_word_price['price'].replace('None', np.nan, inplace=True)
    rest_bag_of_word_price['price'] = rest_bag_of_word_price['price'].astype(float)
    rest_bag_of_word_price['price'].fillna(round(rest_bag_of_word_price['price'].mean()),inplace = True)
    # convert to int
    rest_bag_of_word_price['price'] = rest_bag_of_word_price['price'].astype(float).astype(int)
    # print(rest_bag_of_word_price)
    return rest_bag_of_word_price

def get_tf_idf(rest_word_feq_df):
    # tf
    # tf = rest_word_feq_df.apply(lambda x: get_term_freq(x))
    
    tf = rest_word_feq_df.div(rest_word_feq_df.sum(axis = 0),axis = 1)
    # tf = tf.div(tf.sum(axis = 1),axis = 0)
    tf = tf.round(4)
    tf = tf.fillna(0)
    tf[tf.columns] = np.log(tf[tf.columns]+1)
    
    # idf   
    idf = rest_word_feq_df
    idf[idf.columns] = (rest_word_feq_df.shape[0])/ (idf[idf.columns] +1)
    
    idf[idf.columns] = np.log(idf[idf.columns])
    # print(idf)
    rest_tf_idf = pd.DataFrame(tf.values * idf.values, columns = rest_word_feq_df.columns, index = rest_word_feq_df.index)
    # print(rest_tf_idf[rest_tf_idf.isna().any(axis=1)])
    rest_tf_idf.to_csv(PROCESSED_DATA + 'rest_tf_idf.csv')
    return rest_tf_idf

def k_means_rest(rest_bag_of_word_price):
    # initialize 
    y_kmeans = np.zeros((rest_bag_of_word_price.shape[0],10))
    sse_res_ls = []
    sil_score_ls = []
    distortions = []
    pca_rest = pca(rest_bag_of_word_price)
    # for k in range(12,32,2):
    #     kmeans_cluster = KMeans(n_clusters=k,random_state=42).fit(pca_rest)
    #     step = int((k-12)/2)
    #     y_kmeans[:,step] = kmeans_cluster.predict(pca_rest)
    #     centroids = kmeans_cluster.cluster_centers_
    #     sse_res_ls.append(compute_sse(pca_rest,y_kmeans[:,step],centroids, k))
    #     sil_score_ls.append(get_sil_score(pca_rest,y_kmeans[:,step]))
    #     distortions.append(sum(np.min(cdist(pca_rest, centroids, 'euclidean'), axis=1)) / pca_rest.shape[0])
    #     print(distortions)
    # print('score: ')    
    # print(sil_score_ls)

    # plot_sse(sse_res_ls)
    # plot_elbow(distortions)
    kmeans_cluster = KMeans(n_clusters=26,random_state=42).fit(pca_rest)
    category = kmeans_cluster.predict(pca_rest)
    rest_cat = pd.DataFrame(category, columns=['category'],index = rest_bag_of_word_price.index.values.tolist()) 
    rest_cat.reset_index(inplace= True)
    rest_cat.rename(columns = {'index':'business_id'},inplace = True)
    # print(rest_cat)
    restaurant_df = pd.read_csv(PROCESSED_DATA+'restaurant.csv')
    result = pd.merge(rest_cat, restaurant_df, on='business_id')
    sil_score = silhouette_score(pca_rest, category, metric='euclidean')
    print(sil_score)
    # print(result)
    result.to_csv(RESULT+'restaurant_cat.csv')

   
def EM(rest_bag_of_word_price):
    y_em = np.zeros((rest_bag_of_word_price.shape[0],11))
    
    sil_score_ls = []
    pca_rest = pca(rest_bag_of_word_price)
    for k in range(50,101,5):
        step = int((k-20)/5)
        y_em[:,step] = GaussianMixture(n_components=k,random_state=42).fit_predict(pca_rest)
        # compute silhouette coefficient via original data and predicted clusters
        sil_score_ls.append(sil_score(pca_rest,y_em[:,step]))
        print(sil_score_ls)
       
def pca(data):
    norm_data = StandardScaler().fit_transform(data)

    #Plotting the Cumulative Summation of the Explained Variance
    # plt_pca(norm_data)
    pca = PCA(0.85)
    pca_data = pca.fit_transform(norm_data)
    print(pca.n_components_)
    return pca_data

# get_restaurant_reviews()
def plt_pca(norm_data):
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
    
    score = silhouette_score(data, predicted, metric='euclidean')
    return score

def plot_sse(sse_res):
    plt.figure()
    x = list(range(12,32,2))
    plt.plot(x, sse_res,'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.savefig("k_means_sse")

def plot_elbow(distortions):
    plt.figure()
    filename="kmeans_elbow"
    x = list(range(12,32,2))
    plt.plot(x, distortions, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 2.0))
    plt.xlabel('K')
    plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)

def most_freq_word_cat():
    rest_cat = pd.read_csv(RESULT+"restaurant_cat.csv")
    rest_cat = rest_cat[['business_id','category']]
    rest_word_feq_df = pd.read_csv(PROCESSED_DATA+'rest_tf_idf.csv',index_col = 0)
    rest_word_feq_df.reset_index(inplace= True)
    rest_word_feq_df.rename(columns = {'index':'business_id'},inplace = True)
    df = pd.merge(rest_cat, rest_word_feq_df, on='business_id')
    distribution = df.groupby('category').size().to_frame()
    distribution.to_csv(RESULT+'distribution_cat.csv')
    features = list(rest_word_feq_df.columns)
    features.remove('business_id')
    word_cat = df.groupby(by = ['category']).sum()
  
    word_cat =word_cat.transpose()
    most_freq_cat = pd.DataFrame(columns= ['most frequent words'],index = word_cat.columns)
    for col in word_cat.columns:
        most_freq = word_cat[col].argsort()
        freq_word_df = word_cat[col][most_freq].tail(15)
        fw_ls = freq_word_df.index.tolist()
        fw_ls.reverse()
        fw_str = ", ".join([word for word in fw_ls])
        most_freq_cat.iloc[col] = fw_str
    print(most_freq_cat)
    most_freq_cat.to_csv(RESULT+'most_freq_word_cat.csv')
    # print(distribution)
    # ax = distribution.plot.bar(alpha=0.5)
    # plt.show()
   



# rest_bag_of_word_price = add_price_att()
# k_means_rest(rest_bag_of_word_price)
most_freq_word_cat()


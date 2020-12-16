import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import interpolate
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import glob
import math
from scipy.spatial.distance import cdist
from matplotlib import colors
import seaborn as sns

PITTSBURGH_RACE = '../census_data/Pittsburgh_race.csv'
PITTSBURGH_NEIGH_CENTROID = '../census_data/PittsburghNeighCentroid.csv'
PITTSBURGH_RACE_CLUSTER = '../result/Neighborhood_race_boundary_6.csv'
PITTSBURGH_CULT_CLUSTER = '../result/neighborhood_culture_boundary_Pittsburgh_6.csv'

REST_SUBCAT_CITY = '../result/sub/restaurant_sub_cat_sk.csv'
REST_SUBCAT_NEIGHBORHOOD = '../result/sub/neighborhood/'
JOIN_TABLE = '../yelp_dataset/join_table/'
RESULT = '../result/'
REST_SUBCAT_TEMP = RESULT+'subcat_temp/rest_subcat_temp.csv'
CITY_SUBCAT_TEMP = RESULT+'subcat_temp/city_subcat_temp.csv'
NEIGHBORHOOD_SUBCAT_TEMP = RESULT + 'subcat_temp/'

CULT_BOUND_CITY = RESULT+ 'city_culture_boundary_'
CULT_BOUND_NEIGHBORHOOD = RESULT+ 'neighborhood_culture_boundary_3.csv'

city_ls = ['Cleveland', 'Pittsburgh', 'Toronto']

def join_cat_checkin():
    rest_sub_cat = pd.read_csv(REST_SUBCAT_CITY, index_col=0)
    rest_sub_cat = rest_sub_cat[['business_id','sub_category','name','city','state']]
    rest_checkin_times = pd.read_csv(JOIN_TABLE + 'restaurant_checkin.csv',dtype = object,index_col = 0)
    rest_checkin_times = rest_checkin_times[ ['business_id','date']]
    rest_checkin_times['date'] = rest_checkin_times.apply(lambda x: x['date'].split(', '), axis = 1)
    rest_checkin_times['date'] = rest_checkin_times.apply(lambda x: sorted([datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') for date_time in x['date'] ] ), axis = 1)
    sub_cat_checkin = pd.merge(rest_sub_cat, rest_checkin_times, on = 'business_id')
    sub_cat_checkin.to_csv(RESULT+'sub_cat_checkin.csv')
    return sub_cat_checkin

def get_temp_pattern(sub_cat_checkin):
    rest_cat_df = sub_cat_checkin.drop(columns = ['date'])
    rest_cat_temp_df = pd.DataFrame()
    for row in range(sub_cat_checkin.shape[0]):
        checkin_times = sub_cat_checkin.loc[row,'date']
        sub_cat = sub_cat_checkin.loc[row,'sub_category']
        count_temp_dic = count_temp_pattern(checkin_times, sub_cat)
        temp_df = pd.DataFrame([count_temp_dic])
        df = rest_cat_df.loc[[row]].reset_index(drop=True)
        rest_temp_df = pd.concat( [ df, temp_df ], axis = 1)
        rest_cat_temp_df =  rest_cat_temp_df.append (rest_temp_df)
        print(rest_cat_temp_df)
        # rest_temp_df = rest_temp_df.append(temp_df)
    rest_cat_temp_df.reset_index(drop = True, inplace = True)
    print(rest_cat_temp_df)
    # rest_cat_temp_df = pd.concat([rest_cat_df,rest_temp_df], axis = 1,ignore_index = True)
    rest_cat_temp_df.to_csv(REST_SUBCAT_TEMP)
    temp_ls = rest_cat_temp_df.columns.tolist()
    del temp_ls[0 : 5] 
    agg_df = rest_cat_temp_df.groupby(['city'])[temp_ls].sum().reset_index()
    print(agg_df)
    print(len(temp_ls))
    agg_df.to_csv(CITY_SUBCAT_TEMP)
        # for temp_key in list(count_temp_dic.keys()):
        #     cat_checkin[temp_key] = count_temp_dic.get(temp_key)
        

def get_temp_pattern_neighborhood(sub_cat_checkin):
    # rest_cat_df = sub_cat_checkin.drop(columns = ['date'])
    neighborhodd_csv_ls = glob.glob(REST_SUBCAT_NEIGHBORHOOD+'*.csv')
    
    for neighborhood_csv in neighborhodd_csv_ls:
        
        neighborhood_df = pd.read_csv(neighborhood_csv)
        neighborhood_df.rename(columns = {'hood': 'neighborhood', 'AREA_NAME': 'neighborhood', 'SPA_NAME': 'neighborhood'}, inplace = True)
        neighborhood_df = neighborhood_df[['business_id','neighborhood']]
        
        sub_cat_checkin =pd.merge(sub_cat_checkin, neighborhood_df, on= 'business_id')
    
        rest_cat_df = sub_cat_checkin.drop(columns = ['date'])
        
        rest_cat_temp_df = pd.DataFrame()
        for row in sub_cat_checkin.index.tolist():
            checkin_times = sub_cat_checkin.loc[row,'date']
            sub_cat = sub_cat_checkin.loc[row,'sub_category']
            count_temp_dic = count_temp_pattern(checkin_times, sub_cat)
            temp_df = pd.DataFrame([count_temp_dic])
            df = rest_cat_df.loc[[row]].reset_index(drop=True)
            rest_temp_df = pd.concat( [ df, temp_df ], axis = 1)
            rest_cat_temp_df =  rest_cat_temp_df.append (rest_temp_df)
            print(rest_cat_temp_df)
            # rest_temp_df = rest_temp_df.append(temp_df)
        rest_cat_temp_df.reset_index(drop = True, inplace = True)
        print(rest_cat_temp_df)
        # rest_cat_temp_df = pd.concat([rest_cat_df,rest_temp_df], axis = 1,ignore_index = True)
        temp_ls = rest_cat_temp_df.columns.tolist()
        del temp_ls[0 : 5] 
        agg_df = rest_cat_temp_df.groupby(['neighborhood'])[temp_ls].sum().reset_index()
        print(agg_df)
        print(len(temp_ls))
        city_name = neighborhood_csv.split('/')[4]
        city_name = city_name.replace('.csv','')
        agg_df.to_csv(RESULT+ 'subcat_temp/'+city_name + '_cat_temp.csv')

def count_temp_pattern(checkin_times, sub_cat):
    count_temp = {'t0_weekday':0, 't0_weekend': 0, 't1_weekday':0, 't1_weekend':0, 't2_weekday': 0, 't2_weekend':0, 't3_weekday':0, 't3_weekend':0  }
    # print(report_times)
    t0 = datetime.datetime.strptime('23:00:00', "%H:%M:%S").time()
    t1 = datetime.datetime.strptime('06:00:00', "%H:%M:%S").time()
    t2 = datetime.datetime.strptime('11:00:00', "%H:%M:%S").time()
    t3 = datetime.datetime.strptime('18:00:00', "%H:%M:%S").time()
    for checkin in checkin_times:
        checkin_time = checkin.time()
        day_of_week = checkin.weekday()
        # Monday = 0 ~ Friday = 4
        if day_of_week <5:
            # 6:00:01 ~11:00:00
            if checkin_time > t1 and checkin_time <=t2:
                count_temp['t1_weekday'] +=1
            # 11:00:01 ~18:00:00
            elif checkin_time > t2 and checkin_time <=t3:
                count_temp['t2_weekday'] +=1
            # 18:00:01 ~ 23:00:00
            elif checkin_time > t3 and checkin_time <=t0:
                count_temp['t3_weekday'] +=1
             # 23:00:01~6:00:00 > 23:00 or <=6:00 
            elif checkin_time > t0 or checkin_time <= t1:
                count_temp['t0_weekday'] +=1
        # Saturday = 5 ~ Sunday = 6
        else:
            # 6:00:01 ~11:00:00
            if checkin_time > t1 and checkin_time <=t2:
                count_temp['t1_weekend'] +=1
            # 11:00:01 ~18:00:00
            elif checkin_time > t2 and checkin_time <=t3:
                count_temp['t2_weekend'] +=1
            # 18:00:01 ~ 23:00:00
            elif checkin_time > t3 and checkin_time <=t0:
                count_temp['t3_weekend'] +=1
            # 23:00:01~6:00:00 > 23:00 or <=6:00 
            elif checkin_time > t0 or checkin_time <= t1:
                count_temp['t0_weekend'] +=1
    count_temp_res = {sub_cat + "_" + str(key): val for key, val in count_temp.items()} 
    return count_temp_res

def visualize_cult_dist(place_level, cat_temporal_df,k):
    rest_subCat_temp_df = pd.read_csv(cat_temporal_df)
    places_ls = rest_subCat_temp_df[place_level].tolist()
    clusters_ls = rest_subCat_temp_df['cluster'].tolist()
    rest_subCat_temp_df.drop(columns = [place_level, 'cluster'], inplace = True)
    rest_subCat_temp_pca = pca(rest_subCat_temp_df,2)
    
   
    # figure 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("cultural distance")
    x_ls = np.array(rest_subCat_temp_pca[:,0])
    y_ls = np.array(rest_subCat_temp_pca[:,1])
    scatter = ax.scatter(x_ls, y_ls, marker='s', c = clusters_ls, cmap="viridis")
    # legend = plt.legend(*scatter.legend_elements(),loc="lower left", title=place_level)
    texts = []
    for place, x, y in zip(places_ls,x_ls, y_ls):
        # plt.annotate(
        # place,
        # xy=(x, y),xytext=(0,3),
        # textcoords='offset points', ha='right', va='bottom')
        texts.append(plt.text(x, y, place))
    x = np.arange(min(x_ls), max(x_ls),  0.0005)
    f = interpolate.interp1d(x_ls, y_ls)
    y = f(x)
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    fname = "../figure/cult_dist/" + place_level+ "_cult_bound_" + str(k)
    plt.savefig(fname,bbox_inches='tight')

def visualize_cult_dist_neighborhood(place_level, city, cat_temporal_df,k):
    rest_subCat_temp_df = pd.read_csv(cat_temporal_df)
    place_ls = rest_subCat_temp_df[place_level].tolist()
    cluster_ls = rest_subCat_temp_df['cluster'].tolist()
    rest_subCat_temp_df.drop(columns = ['neighborhood','neighborhood_name','city','cluster'], inplace = True)
    rest_subCat_temp_pca = pca(rest_subCat_temp_df,2)
    
   
    # figure 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("cultural distance")
    x_ls = np.array(rest_subCat_temp_pca[:,0])
    y_ls = np.array(rest_subCat_temp_pca[:,1])
    marker_ls = ['.','x', '^']
    color_ls =  ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(len(x_ls)):
        if place_ls[i].startswith('Cleveland_'):
            m = 0
        elif place_ls[i].startswith('Pittsburgh_'):
            m = 1
        else:
            m = 2
        scatter = ax.scatter(x_ls[i], y_ls[i], marker=marker_ls[m], c = color_ls[cluster_ls[i]], cmap="viridis")
        legend1 = ax.legend(*scatter.legend_elements(prop = 'colors'),loc="lower left", title="City")
        ax.add_artist(legend1)
    fname = "../figure/cult_dist/" + place_level+ "_"+ city + "_cult_bound_" + str(k)
    plt.savefig(fname,bbox_inches='tight')

def pca(data, components):
    data_std = MinMaxScaler().fit_transform(data)
    # data_norm = normalize(data)
    pca = PCA(components)
    data_pca = pca.fit_transform(data_std)
    print(pca.n_components_)
    return data_pca


def test_spectral_clustering_city(place_level,cat_temporal_df):
    
    sil_score_ls = []
    rest_subCat_temp_df = pd.read_csv(cat_temporal_df)
    y_kmeans = np.zeros((rest_subCat_temp_df.shape[0], 4))
    rest_subCat_temp_pca = rest_subCat_temp_df.select_dtypes(include=['float64','int64'])
    cosine_simi_df = get_pairwise_cosine_sim(rest_subCat_temp_pca)
    # rest_subCat_temp_df.drop(columns = [place_level], inplace = True)
    rest_subCat_temp_pca = pca(rest_subCat_temp_pca, 0.99)
    for k in range(2,6,1):
        spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
        step = int((k-2)/1)
        y_kmeans[:,step] = spectral_cluster.fit_predict(cosine_simi_df)
        sil_score = silhouette_score(rest_subCat_temp_pca, y_kmeans[:,step], metric='cosine')
        sil_score_ls.append(sil_score)
    plot_sil_score(sil_score_ls,'spectral_sil_city')



def test_spectral_clustering_neighborhood(neighborhood_df):
    y_kmeans = np.zeros((neighborhood_df.shape[0], 9))
    sil_score_ls = []
    distortions = []
    neighborhood_pca = neighborhood_df.select_dtypes(include=['float64','int64'])
    # neighborhood_pca = neighborhood_df.drop(columns = ['neighborhood','neighborhood_name','city'])
    neighborhood_pca = pca(neighborhood_pca, 0.99)
    cosine_simi_df = get_pairwise_cosine_sim(neighborhood_pca)
    for k in range(2, 11, 1):
        spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
        step = int((k-2)/1)
        y_kmeans[:,step] = spectral_cluster.fit_predict(cosine_simi_df)
        # centroids = spectral_cluster.cluster_centers_
        sil_score = silhouette_score(cosine_simi_df, y_kmeans[:,step], metric='cosine')
        sil_score_ls.append(sil_score)
        # distortions.append(sum(np.min(cdist(cosine_simi_df, centroids, 'cosine'), axis=1)) / cosine_simi_df.shape[0])
    plot_sil_score(sil_score_ls, 'spectral_sil_neighborhood')
    # plot_elbow(distortions, 'spectral_neighborhood')

def spectral_clustering_city(rest_subCat_temp_df, k):
    places_ls = rest_subCat_temp_df['city'].tolist()
    rest_subCat_temp_pca = rest_subCat_temp_df.select_dtypes(include=['float64','int64'])
    # rest_subCat_temp_pca = rest_subCat_temp_df.drop(columns = [place_level])
    rest_subCat_temp_pca = pca(rest_subCat_temp_pca, 0.99)
    cosine_simi_arr = get_pairwise_cosine_sim(rest_subCat_temp_pca)
    cosine_simi_df = pd.DataFrame(cosine_simi_arr, index = places_ls, columns= places_ls )
    print(cosine_simi_df)
    cosine_simi_df.to_csv(RESULT+'cult_sim/city_pairwise_cosine_sim.csv')
    plot_cosine_sim_heatmap(cosine_simi_df)

    spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
    y_clusters = spectral_cluster.fit_predict(cosine_simi_arr)
    sil_score = silhouette_score(cosine_simi_arr, y_clusters, metric='cosine')
    cult_bound = pd.DataFrame(y_clusters, columns=['cluster'],index =places_ls) 
    cult_bound.reset_index(inplace = True)
    cult_bound.rename(columns = {'index':'city'}, inplace = True)
    cult_bound_result = pd.merge(cult_bound, rest_subCat_temp_df, on ='city')
    print(sil_score)
    cult_bound_result.to_csv(RESULT+ CULT_BOUND_CITY +str(k)+'.csv')

def spectral_clustering_neighborhood(neighborhood_df, k, city):
    neighborhood_ls = neighborhood_df['neighborhood'].tolist()
    neighborhood_pca = neighborhood_df.select_dtypes(include=['float64','int64'])
    # neighborhood_pca = neighborhood_df.drop(columns = ['neighborhood','neighborhood_name','city'])
    neighborhood_pca = pca(neighborhood_pca, 0.99)
    cosine_simi_arr = get_pairwise_cosine_sim(neighborhood_pca)
    cosine_simi_df = pd.DataFrame(cosine_simi_arr, index = neighborhood_ls, columns= neighborhood_ls )
    print(cosine_simi_df)
    plot_cosine_sim_heatmap(cosine_simi_df)
    spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
    y_clusters = spectral_cluster.fit_predict(cosine_simi_arr)
    sil_score = silhouette_score(cosine_simi_arr, y_clusters, metric='cosine')
    cult_bound = pd.DataFrame(y_clusters, columns=['cluster'],index =neighborhood_ls) 
    cult_bound.reset_index(inplace = True)
    cult_bound.rename(columns = {'index':'neighborhood'}, inplace = True)
    cult_bound_result = pd.merge(cult_bound, neighborhood_df, on ='neighborhood')
    print(sil_score)
    cult_bound_result.to_csv(RESULT+ 'neighborhood_culture_boundary_'+city+"_"+str(k)+'.csv')

def get_pairwise_cosine_sim(df):
    return cosine_similarity(df)




def test_spectral_pit_race(place_level,pitt_race_df):

    y_kmeans = np.zeros((pitt_race_df.shape[0], 9))
    sil_score_ls = []
    pitt_race_pca = pitt_race_df.drop(columns = [place_level])
    pitt_race_pca = pca(pitt_race_pca, 0.99)
    cosine_simi_df = get_pairwise_cosine_sim(pitt_race_pca)
    for k in range(2, 11, 1):
        spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
        step = int((k-2)/1)
        y_kmeans[:,step] = spectral_cluster.fit_predict(cosine_simi_df)
        sil_score = silhouette_score(cosine_simi_df, y_kmeans[:,step], metric='cosine')
        sil_score_ls.append(sil_score)
    plot_sil_score(sil_score_ls, 'spectral_sil_pitt_race')


def spectral_pit_race(place_level,pitt_race_df, k):
    print(pitt_race_df)
    places_ls = pitt_race_df[place_level].tolist()
    pitt_race_pca = pitt_race_df.drop(columns = [place_level])
    pitt_race_pca = pca(pitt_race_pca, 0.99)
    cosine_simi_df = get_pairwise_cosine_sim(pitt_race_pca)
    spectral_cluster = SpectralClustering(n_clusters= k, random_state = 42)
    y_clusters = spectral_cluster.fit_predict(cosine_simi_df)
    sil_score = silhouette_score(cosine_simi_df, y_clusters, metric='cosine')
    race_bound = pd.DataFrame(y_clusters, columns=['cluster'],index =places_ls) 
    race_bound.reset_index(inplace = True)
    race_bound.rename(columns = {'index':place_level}, inplace = True)
    race_bound_result = pd.merge(race_bound, pitt_race_df, on = place_level)
    print(sil_score)
    print(race_bound_result)

    race_bound_result.to_csv(RESULT+ 'neighborhood_race_boundary_sc_'+str(k)+'.csv')

def plot_sil_score(sil_score_ls, fname):
    plt.figure()
    filename="../figure/cult_dist/" + fname
    x = list(range(2,len(sil_score_ls)+2,1))
    plt.plot(x, sil_score_ls, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel('K')
    plt.ylabel('Silhouette')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)

def plot_elbow(distortions, fname):
    plt.figure()
    filename="../figure/cult_dist/" + fname
    x = list(range(2,len(distortions)+2,1))
    plt.plot(x, distortions, 'bx-')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.xlabel('K')
    plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    plt.savefig(filename)
def plot_cosine_sim_heatmap(cosine_sim_df):
    fig, ax = plt.subplots(figsize=(10,10)) 
    sns.heatmap(cosine_sim_df, vmin=-1, vmax=1, ax=ax, cmap="YlGnBu")
    plt.show()
def get_pittsburgh_race():
    pitt_race_df = pd.read_csv(PITTSBURGH_RACE)
    filter_col = ['Neighborhood'] + [col for col in pitt_race_df if col.startswith('Estimate')]
    pitt_race_df = pitt_race_df[filter_col]
    pitt_race_df.rename(columns = {'Neighborhood': 'neighborhood_name'}, inplace = True)
    return pitt_race_df



def get_cult_bound_city_level():
    # get_temp_pattern(sub_cat_checkin)
    rest_subCat_temp_df = pd.read_csv(CITY_SUBCAT_TEMP, index_col=0)
    # test_spectral_clustering_city('city', CITY_SUBCAT_TEMP)
    spectral_clustering_city(rest_subCat_temp_df, 4)
    visualize_cult_dist('city',CULT_BOUND_CITY+ str(4) + '.csv',4)

def get_cult_bound_neighborhood_across_cities(city_ls):
    neighborhood_df = pd.DataFrame()
    for city in city_ls:
        df = pd.read_csv(NEIGHBORHOOD_SUBCAT_TEMP + city + '_subcat_temp.csv', index_col = 0)
        df.reset_index(inplace=True)
        df['city'] = city
        df['neighborhood_id'] = df['city'] + '_' + df['index'].apply(str)
        df.drop(columns = ['index'],inplace = True)
        df.rename(columns = {'neighborhood_id': 'neighborhood','neighborhood': 'neighborhood_name'}, inplace = True)
        neighborhood_df = neighborhood_df.append(df)
    neighborhood_df.reset_index(drop=True, inplace = True)
    neighborhood_df.fillna(0, inplace=True)
    neighborhood_ls = neighborhood_df['neighborhood'].tolist()
    neighborhood_pca = neighborhood_df.select_dtypes(include=['float64','int64'])
    neighborhood_pca = pca(neighborhood_pca, 0.99)
    cosine_simi_arr = get_pairwise_cosine_sim(neighborhood_pca)
    cosine_simi_df = pd.DataFrame(cosine_simi_arr, index = neighborhood_ls, columns= neighborhood_ls )
    print(cosine_simi_df)
    cosine_simi_df.to_csv(RESULT+ 'cult_sim/neighborhood_cities_cosine_sim.csv')
    plot_cosine_sim_heatmap(cosine_simi_df)
    test_spectral_clustering_neighborhood(neighborhood_df)
    # number of clusters = 3
    spectral_clustering_neighborhood(neighborhood_df, 3, "across_cities")
    visualize_cult_dist_neighborhood('neighborhood','across_cities',CULT_BOUND_NEIGHBORHOOD,3)

def get_cult_bound_neighborhood_within_city(city):
    neighborhood_df = pd.read_csv(NEIGHBORHOOD_SUBCAT_TEMP + city + '_subcat_temp.csv', index_col = 0)
    neighborhood_df.reset_index(inplace=True)
    neighborhood_df['city'] = city
    neighborhood_df['neighborhood_id'] = neighborhood_df['city'] + '_' + neighborhood_df['index'].apply(str)
    neighborhood_df.drop(columns = ['index'],inplace = True)
    neighborhood_df.rename(columns = {'neighborhood_id': 'neighborhood','neighborhood': 'neighborhood_name'}, inplace = True)
    neighborhood_df.reset_index(drop=True, inplace = True)
    neighborhood_df.fillna(0, inplace=True)
    neighborhood_ls = neighborhood_df['neighborhood'].tolist()
    neighborhood_pca = neighborhood_df.select_dtypes(include=['float64','int64'])
    neighborhood_pca = pca(neighborhood_pca, 0.99)
    cosine_simi_arr = get_pairwise_cosine_sim(neighborhood_pca)
    cosine_simi_df = pd.DataFrame(cosine_simi_arr, index = neighborhood_ls, columns= neighborhood_ls )
    print(cosine_simi_df)
    cosine_simi_df.to_csv(RESULT+ 'cult_sim/neighborhood_'+ city+ '_cosine_sim.csv')
    plot_cosine_sim_heatmap(cosine_simi_df)
    test_spectral_clustering_neighborhood(neighborhood_df)
    # number of clusters = 7 for pittsburgh
    spectral_clustering_neighborhood(neighborhood_df, 7, city)
    # visualize_cult_dist_neighborhood('neighborhood', city, CULT_BOUND_NEIGHBORHOOD,7)
def get_pitts_race_bound():
    pitt_race_df = get_pittsburgh_race()
    test_spectral_pit_race('neighborhood_name',pitt_race_df)
    # number of clusters = 7
    spectral_pit_race('neighborhood_name',pitt_race_df,7)


# sub_cat_checkin =join_cat_checkin() # comment out to save running time
# get_temp_pattern(sub_cat_checkin) # comment out to save running time

'''
city-level
'''
get_cult_bound_city_level()
'''
neighborhood-level
'''
get_cult_bound_neighborhood_across_cities(city_ls)
get_cult_bound_neighborhood_within_city("Pittsburgh")

get_pitts_race_bound()







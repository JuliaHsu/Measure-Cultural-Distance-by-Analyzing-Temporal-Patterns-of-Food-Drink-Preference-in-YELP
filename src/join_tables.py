import pandas as pd
import numpy as np

RAW_DATA = '../yelp_dataset/raw/'
PROCESSED_DATA = '../yelp_dataset/processed_data/'

business_csv = RAW_DATA + 'yelp_academic_dataset_business.csv'
review_csv =  RAW_DATA + 'yelp_academic_dataset_review.csv'
checkin_csv =  RAW_DATA + 'yelp_academic_dataset_checkin.csv'
cols = ['business_id','name','categories','attributes.RestaurantsPriceRange2','attributes.Ambience','review_count','address','postal_code','city','state','longitude','latitude']
cities_ls = ['pittsburgh','toronto','las vegas','phoenix','charlotte','cleveland']
food_drink_ls = ['food','restaurant']
food_drink_pattern = '|'.join(food_drink_ls)

# shopping_ls = ['shopping','department stores']
# shopping_pattern = '|'.join(shopping_ls)
def get_restaurant_business():
    business_df = pd.read_csv(business_csv,dtype=object)
    business_df = business_df.replace(np.nan, '', regex=True)
    business_df= business_df[cols]
    business_df[['name','categories','city']] = business_df[['name','categories','city']].apply(lambda x: x.astype(str).str.lower())
    
    restaurant_df = business_df[business_df['categories'].str.contains(food_drink_pattern)]
    #  remove shopping, grocery stores
    restaurant_df = restaurant_df[~restaurant_df['categories'].str.contains('shopping')]
    
    restaurant_df = restaurant_df.loc[restaurant_df['city'].isin(cities_ls)]
    # print(rest_review)
    restaurant_df.to_csv(PROCESSED_DATA +'restaurant.csv',index = False)
    # print(restaurant_df)
    print(restaurant_df.groupby(['city']).size())
    return restaurant_df

def get_restaurant_review():
    review_df = pd.read_csv(review_csv,dtype = object)
    review_df = review_df.replace(np.nan, '', regex=True)
    restaurant_df = pd.read_csv(PROCESSED_DATA +'restaurant.csv')
    rest_review = pd.merge(restaurant_df, review_df, on='business_id')
    # print(rest_review)
    rest_review.to_csv(PROCESSED_DATA + 'restaurant_review.csv')

def get_restaurant_checkin():
    checkin_df = pd.read_csv(checkin_csv,dtype = object)
    restaurant_df = pd.read_csv(PROCESSED_DATA +'restaurant.csv')
    rest_checkin = pd.merge(restaurant_df, checkin_df, on='business_id')
    rest_checkin.to_csv(PROCESSED_DATA + 'restaurant_checkin.csv')
    # print(rest_checkin)
    print(rest_checkin.groupby(['city']).size())

get_restaurant_business()
get_restaurant_review()
get_restaurant_checkin()
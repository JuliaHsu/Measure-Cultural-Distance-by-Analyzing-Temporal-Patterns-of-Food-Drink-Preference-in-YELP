### MEASURE CULTURAL SIMILARITY BY ANALYZING TEMPORAL PATTERNS OF FOOD AND DRINK PREFERENCE IN YELP
<p>
We analyze food and drink cultures in 6 different cities from Northern America andidentify cultural boundaries across population at different scales based on the temporalpatterns of food and drink preferences. Each region in the map would subtly convey to us thecultural preferences which has resulted in this classification.
</p>

#### Dataset and immediate data

* [Yelp academic dataset](https://www.yelp.com/dataset) 
*  Converted csv files: ``./yelp_dataset/raw/``
* Join tables: ``./yelp_dataset/join_table/``
    * ```restaurant_checkin.csv```
    * ``` restaurant_review.csv```
* Restaurant data: ``./yelp_dataset/restaurant_data/``

    * ```restaurant.csv```: businesses that are tagged with restaurant and food (shopping and grocery store are removed)
    * ```processed_review.csv```: processed reviews (tokenize, lemmatize...)

        * text: processed reviews 
        * processed_text: business tags + processed reviews
    * ```rest_biz_tags.csv``` : business_id + business tags list
    * ```sub_rest_word_freq.csv```: immediate result of bag of word models of restaurant (frequency of term)
    * ```sub_rest_tf_idf.csv```: weighted temrs (tags and reviews) using tf-idf

* Features: ```./yelp_dataset/features/```
    * ```featured_biz_tags.txt```: filtered business tags after removing some business outliers (spa, nail spa...)
    * ```business_tags.csv```: original list of business tags (extracted from businesses that are tagged with food, restaurant)
    * ```1000_freq_word_review.csv```: top 1000 frequent words appear in reviews and business tags (uesd for selecting relevant features)
    * ```features_review.txt```: selected features related to food, drink, ambience and price from reviews
    * ```word_freq_review.csv```: immediate result of relevant word frequency of each restaurant from its reviews


#### Follow below instructions to reproduce the project

1. Convert json to csv and join tables
    * [json_to_csv.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/json_to_csv.py)
        * convert business, reviews, and chekin json files to csv files
        * filter out businesses that are not restaurant
    * [join_tables.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/join_tables.py)
        * join restaurant and checkin tables
        * join restaurant and review tables

2. Detect business outliers and categorize restaurants based on business tags
    * [rest_root_cat.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/rest_root_cat.py)
    ```
    # functions for detecting business outliers, where K = 32
    unique_tags, rest_tags = get_rest_tags()
    rest_tag_df = get_rest_tag_vec(unique_tags, rest_tags)
    k_means_rest_root(rest_tag_df, NUM_CLUSTERS = 32)
    ```

    ```
    # functions for categorizing restaurants based on business tags, where K = 15
    removed_categories_kmean = [20, 27, 28, 30, 16, 26, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 21, 22, 24, 31]
    filtered_rest = remove_business(removed_categories_kmean)
    filtered_tags = get_filtered_tags()
    filtered_rest_vec = get_rest_tag_vec(filtered_tags, filtered_rest)
    k_means_rest_root(filtered_rest_vec, NUM_CLUSTERS = 15)
    ```
3. Relevant features extraction
    * [feature_extraction.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/feature_extraction.py)
    ```
    python3 feature_extraction.py
    ```



4. Further categorize restaurants into subcategories base on selected features in ```features_review.txt```
    * [rest_sub_clustering.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/rest_sub_clustering.py)

    #### Determine number of clusters K
    Use elbow method to determine number of clusters

    ```
    python3 rest_sub_clustering.py
    ```
5. Measure Clutural similarties and boundaries at different scales with pairwise cosine similarties
    * [get_cult_bound.py](https://github.com/JuliaHsu/Measure-Cultural-Distance-by-Analyzing-Temporal-Patterns-of-Food-Drink-Preference-in-YELP/blob/master/src/get_cult_bound.py)
     ```
    python3 get_cult_bound.py
    ```
<!-- 
## Deployment

Add additional notes about how to deploy this on a live system -->

<!-- ## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).  -->

## Authors

* **Julia Hsu &  Aiswarya Kannan** - *Initial work* - 

<!-- ## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc -->


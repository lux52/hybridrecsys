# Importing Libraries and cookbooks
import recsys
import prepocess

ratings = pd.read_csv('./ml-latest-small/ratings.csv')
ratings.head()

# Importing movie data and having a look at first five columns
movies = pd.read_csv('./ml-latest-small/movies.csv')
movies.head()

# Creating interaction matrix using rating data
interactions = create_interaction_matrix(df = ratings,
                                         user_col = 'userId',
                                         item_col = 'movieId',
                                         rating_col = 'rating')
interactions.head()

# Create User Dict
user_dict = create_user_dict(interactions=interactions)
# Create Item dict
movies_dict = create_item_dict(df = movies,
                               id_col = 'movieId',
                               name_col = 'title')

mf_model = runMF(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)

## Calling 10 movie recommendation for user id 11
rec_list = sample_recommendation_user(model = mf_model, 
                                      interactions = interactions, 
                                      user_id = 11, 
                                      user_dict = user_dict,
                                      item_dict = movies_dict, 
                                      threshold = 4,
                                      nrec_items = 10,
                                      show = True)

## Calling 15 user recommendation for item id 1
sample_recommendation_item(model = mf_model,
                           interactions = interactions,
                           item_id = 1,
                           user_dict = user_dict,
                           item_dict = movies_dict,
                           number_of_user = 15)

## Calling 10 recommended items for item id 
rec_list = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                    item_id = 5378,
                                    item_dict = movies_dict,
                                    n_items = 10)

print(rec_list)

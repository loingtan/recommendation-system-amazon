import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
class CollaborativeFilteringRecommender:
    def __init__(self, df):
        self.df = df

        self.user_item_matrix = self.df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        self.user_similarity_matrix = self._get_user_similarity_matrix().dot(
            self._get_user_demographic_matrix(weightage_factor=0.6))

    def _get_user_similarity_matrix(self):
        user_similarity_matrix = pd.DataFrame(cosine_similarity(self.user_item_matrix),
                                              index=self.user_item_matrix.index,
                                              columns=self.user_item_matrix.index)
        return user_similarity_matrix

    def _get_user_demographic_matrix(self, weightage_factor):
        age_encoder = LabelEncoder()

        user_profiles = self.df[['user_id', 'age', 'gender', 'location']].drop_duplicates()
        user_profiles = user_profiles.set_index('user_id')
        user_profiles['age_encoded'] = age_encoder.fit_transform(user_profiles['age'])

        onehot_cols = ['gender', 'location']
        user_profiles_encoded = pd.get_dummies(user_profiles, columns=onehot_cols)
        user_profiles_encoded = user_profiles_encoded.drop('age', axis=1)

        user_demographic_matrix = pd.DataFrame(cosine_similarity(user_profiles_encoded),
                                               index=user_profiles_encoded.index,
                                               columns=user_profiles_encoded.index)
        return user_demographic_matrix * weightage_factor

    def get_recommendations(self, user_id, top_n=10):

        user_similarity = self.user_similarity_matrix[user_id]
        similar_user_indices = user_similarity.argsort()[::-1][1:top_n + 1]
        recommended_products = set()
        for index in similar_user_indices:
            similar_user_id = self.df.iloc[index]['user_id']
            products_interacted = self.df[self.df['user_id'] == similar_user_id]['item_id'].tolist()
            recommended_products.update(products_interacted)
        target_user_products = self.df[self.df['user_id'] == user_id]['item_id']
        recommended_products = recommended_products - set(target_user_products)
        top_n_items = self.df[self.df['item_id'].isin(recommended_products)]['item_id'].head(top_n)
        return top_n_items
class ContentBasedRecommender:
    def __init__(self, df):
        self.df = df
        self.category_similar_matrix = self._get_similarity_matrix(self.df['sub_cat'])
        self.brand_similar_matrix = self._get_similarity_matrix(self.df['brand'])

    def _get_similarity_matrix(self, attribute_data):
        # Create a DataFrame with item_id and attribute_data
        df_attribute = pd.DataFrame({'item_id': self.df['item_id'], 'attribute_data': attribute_data})

        # Drop duplicate entries based on item_id, keeping only the first occurrence
        df_attribute_unique = df_attribute.drop_duplicates(subset='item_id', keep='first')

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        doc_term = tfidf_vectorizer.fit_transform(df_attribute_unique['attribute_data'])
        dt_matrix = pd.DataFrame(doc_term.toarray().round(3), index=[i for i in df_attribute_unique['item_id']],
                                 columns=tfidf_vectorizer.get_feature_names_out())
        cos_similar_matrix = pd.DataFrame(cosine_similarity(dt_matrix.values), columns=df_attribute_unique['item_id'],
                                          index=df_attribute_unique['item_id'])
        return cos_similar_matrix

    def get_recommendations(self, user_id, top_n=5):
        # Create an empty list to store recommendations
        top_n_recommendations = []

        # Get the user's item interactions from the dataset
        user_items = self.df[self.df['user_id'] == user_id]['item_id']

        # Combine category and brand similarity (e.g., by taking the average)
        combined_similar_matrix = (self.category_similar_matrix + self.brand_similar_matrix) / 2

        # Iterate through the user's interactions
        for item_id in user_items:
            # Get similar items based on the combined similarity matrix for the current item
            similar_items = combined_similar_matrix.loc[item_id]
            similar_items = similar_items.sort_values(ascending=False)

            # Exclude items that the user has already interacted with
            similar_items = similar_items[~similar_items.index.isin(user_items)]

            # Get top-N recommended items for the user from the current item's similarity
            top_n_items = similar_items.head(top_n).index.tolist()
            top_n_recommendations.extend(top_n_items)

        # Return the list of top-N recommendations for the user (excluding their own interactions)
        return list(set(top_n_recommendations) - set(user_items))

class PopularityBasedRecommender:
    def __init__(self, df):
        self.df = df
        self.user_item_matrix = self._get_user_item_interaction_matrix()

    def _get_user_item_interaction_matrix(self):
        # Create user-item interaction matrix
        user_item_matrix = self.df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
        return user_item_matrix

    def get_trending_items(self, period=15, top_n=5):
        # Filter the dataset to get interactions in the last "period" days
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        recent_interactions = self.df[self.df['timestamp'] >= self.df['timestamp'].max() - pd.Timedelta(days=period)]

        # Calculate popularity based on recent interactions
        item_popularity = recent_interactions['item_id'].value_counts()

        # Get the titles, interaction count, and average rating of the trending items
        trending_items = item_popularity.head(top_n).index.tolist()
        trending_titles = self.df[self.df['item_id'].isin(trending_items)]['title'].unique()
        trending_interaction_count = item_popularity.head(top_n).values
        trending_avg_rating = recent_interactions.groupby('item_id')['rating'].mean().loc[trending_items].values

        # Return the top N trending items along with their titles, interaction count, and average rating
        return pd.DataFrame({'item_id': trending_items, 'title': trending_titles,
                             'interaction_count': trending_interaction_count, 'avg_rating': trending_avg_rating})

    def get_most_popular_items(self, top_n=5):
        # Calculate popularity based on overall interactions
        item_popularity = self.df['item_id'].value_counts()

        # Get the interaction count and average rating of the most popular items
        most_popular_items = item_popularity.head(top_n).index.tolist()
        get_title_by_item_id = lambda item_id: self.df[self.df['item_id'] == item_id]['title'].values[0]
        most_popular_titles = list(map(get_title_by_item_id, most_popular_items))
        most_popular_interaction_count = item_popularity.head(top_n).values
        most_popular_avg_rating = self.df[self.df['item_id'].isin(most_popular_items)].groupby('item_id')[
            'rating'].mean().values

        # Return the top N most popular items along with their interaction count and average rating
        return pd.DataFrame({'item_id': most_popular_items, 'title': most_popular_titles,
                             'interaction_count': most_popular_interaction_count,
                             'avg_rating': most_popular_avg_rating})

    def get_top_rated_items(self, min_interactions=10, top_n=5):
        # Calculate average ratings for items with minimum interactions
        item_ratings = self.df.groupby('item_id')['rating'].agg(['count', 'mean'])
        item_ratings = item_ratings[item_ratings['count'] >= min_interactions]

        # Get the interaction count and average rating of the top rated items
        top_rated_items = item_ratings.nlargest(top_n, 'mean').index.tolist()
        top_rated_titles = self.df[self.df['item_id'].isin(top_rated_items)]['title'].unique()
        top_rated_interaction_count = item_ratings.nlargest(top_n, 'mean')['count'].values
        top_rated_avg_rating = item_ratings.nlargest(top_n, 'mean')['mean'].values

        # Return the top N highest rated items along with their interaction count and average rating
        return pd.DataFrame({'item_id': top_rated_items, 'title': top_rated_titles,
                             'interaction_count': top_rated_interaction_count, 'avg_rating': top_rated_avg_rating})

    def get_bestsellers_in_demographic(self, age, gender, location, top_n=5):
        # Filter the dataset to get interactions in the user's demographic
        demographic_interactions = self.df[(self.df['age'] == age) &
                                           (self.df['gender'] == gender) &
                                           (self.df['location'] == location)]

        # Calculate popularity based on interactions in the user's demographic
        item_popularity = demographic_interactions['item_id'].value_counts()

        # Get the interaction count and average rating of the bestsellers in the demographic
        bestsellers_items = item_popularity.head(top_n).index.tolist()
        bestsellers_titles = self.df[self.df['item_id'].isin(bestsellers_items)]['title'].unique()
        bestsellers_interaction_count = item_popularity.head(top_n).values
        bestsellers_avg_rating = demographic_interactions.groupby('item_id')['rating'].mean().loc[
            bestsellers_items].values

        # Return the top N bestsellers in the demographic along with their interaction count and average rating
        return pd.DataFrame({'item_id': bestsellers_items, 'title': bestsellers_titles,
                             'interaction_count': bestsellers_interaction_count, 'avg_rating': bestsellers_avg_rating})

    def get_popular_in_location(self, location, top_n=5):
        # Filter the dataset to get interactions in the user's location
        location_interactions = self.df[self.df['location'] == location]

        # Calculate popularity based on interactions in the user's location
        item_popularity = location_interactions['item_id'].value_counts()

        # Get the interaction count and average rating of the popular items in the location
        popular_items = item_popularity.head(top_n).index.tolist()
        popular_titles = self.df[self.df['item_id'].isin(popular_items)]['title'].unique()
        popular_interaction_count = item_popularity.head(top_n).values
        popular_avg_rating = location_interactions.groupby('item_id')['rating'].mean().loc[popular_items].values

        # Return the top N popular items in the location along with their interaction count and average rating
        return pd.DataFrame({'item_id': popular_items, 'title': popular_titles,
                             'interaction_count': popular_interaction_count, 'avg_rating': popular_avg_rating})

class HybridRecommender:
    def __init__(self, df, content_based_weight=0.5, collaborative_filtering_weight=0.5):
        self.content_based_recommender = ContentBasedRecommender(df)
        self.collaborative_filtering_recommender = CollaborativeFilteringRecommender(df)
        self.content_based_weight = content_based_weight
        self.collaborative_filtering_weight = collaborative_filtering_weight

    def get_recommendations(self, user_id, top_n=10):
        # Get recommendations from both content-based and collaborative filtering recommenders
        content_based_recommendations = self.content_based_recommender.get_recommendations(user_id, top_n)
        collaborative_filtering_recommendations = self.collaborative_filtering_recommender.get_recommendations(user_id,
                                                                                                               top_n)

        # Combine the recommendations using weighted average
        hybrid_recommendations = {}
        for item in content_based_recommendations:
            hybrid_recommendations[item] = hybrid_recommendations.get(item, 0) + self.content_based_weight
        for item in collaborative_filtering_recommendations:
            hybrid_recommendations[item] = hybrid_recommendations.get(item, 0) + self.collaborative_filtering_weight

        # Sort the hybrid recommendations based on the weighted score and select the top N items
        hybrid_recommendations = sorted(hybrid_recommendations.items(), key=lambda x: x[1], reverse=True)
        hybrid_recommendations = [item for item, score in hybrid_recommendations[:top_n]]

        return hybrid_recommendations
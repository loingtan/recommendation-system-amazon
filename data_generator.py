import pandas as pd
import os
import json
import gzip


class DataGenerator:
    def __init__(self, meta_files_dir, review_files_dir):
        self.meta_files_dir = meta_files_dir
        self.review_files_dir = review_files_dir

    def read_metadata(self, file):
        ### load the meta data
        data = []
        with gzip.open(filename=file) as f:
            for l in f:
                data.append(json.loads(l.strip()))

        # convert list into pandas dataframe
        df = pd.DataFrame.from_dict(data)
        df = df[['asin', 'title', 'brand']]
        df = df.rename(columns={'asin': 'item_id'})
        return df

    def read_reviewsdata(self, file):
        # Load the reviews data
        data = []
        with gzip.open(filename=file) as f:
            for l in f:
                data.append(json.loads(l.strip()))

        # Convert list into pandas dataframe
        df = pd.DataFrame.from_dict(data)
        df = df[['asin', 'reviewerID', 'overall', 'reviewTime']]
        df = df.rename(columns={'asin': 'item_id', 'reviewerID': 'user_id',
                                'overall': 'rating', 'reviewTime': 'timestamp'})

        # Add sub_cat column with file name without extension
        file_name = os.path.basename(file)
        sub_cat = file_name.split('.')[0]
        df['sub_cat'] = sub_cat

        # Add main_cat column based on sub_cat
        df['main_cat'] = ''
        df.loc[df['sub_cat'].isin(['All_Beauty', 'AMAZON_FASHION',
                                   'Luxury_Beauty']), 'main_cat'] = 'Beauty and Fashion'
        df.loc[df['sub_cat'].isin(['Software', 'Video_Games', 'Electronics', 'Kindle_Store',
                                   'Cell_Phones_and_Accessories']), 'main_cat'] = 'Electronics and Technology'
        df.loc[df['sub_cat'].isin(['Digital_Music', 'Magazine_Subscriptions',
                                   'Movies_and_TV']), 'main_cat'] = 'Media and Entertainment'

        return df

    def merge_data(self, metadata, reviewsdata):
        df = pd.merge(metadata, reviewsdata, on='item_id', how='right')
        return df

    def generate_data(self, meta_files, review_files):
        merged_dfs = []

        # Process each meta file and review file
        for meta_file, review_file in zip(meta_files, review_files):
            # Read metadata
            df1 = self.read_metadata(os.path.join(self.meta_files_dir, meta_file))
            # Read reviews data
            df2 = self.read_reviewsdata(os.path.join(self.review_files_dir, review_file))
            # Merge DataFrames
            merged_df = self.merge_data(df1, df2)
            # Append merged DataFrame to the list
            merged_dfs.append(merged_df)

        # Concatenate all merged DataFrames into a single DataFrame and sample it
        df = pd.concat(merged_dfs, ignore_index=True)
        df = df.sample(n=50000, random_state=42)
        return df

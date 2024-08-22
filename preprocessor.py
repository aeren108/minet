import pandas as pd
import numpy as np
import gzip
import ast
from pandarallel import pandarallel

ratings_books = "raw-data/ratings_Books.csv"
ratings_movies = "raw-data/ratings_Movies_and_TV.csv"
meta_books = "raw-data/meta_Books.csv"
meta_movies = "raw-data/meta_Movies.csv"

user_df = None

#book and movie ratings dataframes
rbdf = None
rmdf = None

# book and movie metadata dataframes
mmdf = None
mbdf = None

# grouped dataframes (by user_id)
gbdf = None
gmdf = None

# final book and movie dataframes 
mdf = None
bdf = None

blacklist = ["the", "a", "of", "in", "on", "at", "to", "into", "an", "dvd", "[vhs]", "&amp;", "-"]
index_map = {}
last_index = 0

pandarallel.initialize()

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def df_fromjson(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def init_users():
    global user_df
    user_df = pd.read_csv(f"raw-data/users.csv")

def filter_users_with_enough_data(save=False):
    global rbdf, rmdf

    rbdf = pd.read_csv(ratings_books)
    rmdf = pd.read_csv(ratings_movies)
    mbdf = pd.read_csv(meta_books)
    mmdf = pd.read_csv(meta_movies)

    # remove rows if item does not have a metadata
    rbdf = rbdf[rbdf["item_id"].isin(mbdf["asin"])]
    rmdf = rmdf[rmdf["item_id"].isin(mmdf["asin"])]
    
    rbdfp = rbdf[rbdf["rating"] > 3]
    rmdfp = rmdf[rmdf["rating"] > 3]

    # remove rows if user does not have at least 5 ratings in both domains
    user_book_count = rbdfp.groupby('user_id').size().reset_index(name='rating_count')
    user_movie_count = rmdfp.groupby('user_id').size().reset_index(name='rating_count')
    
    print(user_book_count)
    print(rbdf[rbdf["user_id"] == user_book_count.iat[0, 0]])

    users_enough_book = set(user_book_count[user_book_count['rating_count'] >= 5]['user_id'])
    users_enough_movie = set(user_movie_count[user_movie_count['rating_count'] >= 5]['user_id'])

    users_enough_both = users_enough_book.intersection(users_enough_movie)

    rbdf = rbdf[rbdf["user_id"].isin(users_enough_both)]
    rmdf = rmdf[rmdf["user_id"].isin(users_enough_both)]
    
    print("---- COMMON USERS----")
    print(len(users_enough_both))
    print("---- BDF ----")
    print(rbdf.columns)
    print(rbdf.shape)
    print(rbdf.nunique())
    print("\n---- MDF ----")
    print(rmdf.columns)
    print(rmdf.shape)

    if save:
        rbdf.to_csv("raw-data/ratings_Books_enough_data.csv")
        rmdf.to_csv("raw-data/ratings_Movies_enough_data.csv")

def find_shared_users(file1, file2, output_file):
    users1 = set(pd.read_csv(file1, header=None)[0])
    users2 = set(pd.read_csv(file2, header=None)[0])
    
    shared_users = users1.intersection(users2)
    
    with open(output_file, 'w') as f:
        for user in shared_users:
            f.write(str(user) + '\n')

def filter_shared_users_books(ratings_books, shared_users, output_file):
    shared_users = set(pd.read_csv(shared_users, header=None)[0])
    ratings_books = pd.read_csv(ratings_books)
    
    ratings_shared = ratings_books[ratings_books['user_id'].isin(shared_users)]
    
    ratings_shared.to_csv(output_file, index=False)

def flatten_list(arr):
    return [item for sublist in arr for item in sublist]

def process_title(title: str):
    words = title.split(" ")
    words = [word.lower() for word in words if word.lower() not in blacklist]
    
    if len(words) < 10:
        words.extend([0] * (10 - len(words)))
    
    return words[:10]

def process_categories(categories):
    cats = flatten_list(ast.literal_eval(categories))
    
    if len(cats) < 10:
        cats.extend([0] * (10 - len(cats)))

    return cats[:10]

def get_target_item(user_id, timestamp, i):
    global mmdf
    gdf = gmdf.get_group(user_id)
    gdf = gdf[gdf["timestamp"] < timestamp]
    
    item = {"item_id": 0, "title": [0]*10, "categories": [0]*10}

    if i < len(gdf):
        item_id =  gdf.iat[i, gdf.columns.get_loc("item_id")]
        row = mmdf[mmdf["asin"] == item_id].iloc[0]

        if type(row["title"]) == str:
            title = process_title(row["title"])
            item["title"] = title
            
        if type(row["categories"]) == str:
            categories = process_categories(row["categories"])
            item["categories"] = categories
        
        item["item_id"] = item_id

    columns = [item["item_id"]] + item["title"] + item["categories"]
    return pd.Series(columns)

def get_source_item(user_id, timestamp, i):
    global mbdf
    gdf = gbdf.get_group(user_id)
    gdf = gdf[gdf["timestamp"] < timestamp]

    item = {"item_id": 0, "title": [0]*10, "categories": [0]*10}

    if i < len(gdf):
        item_id = gdf.iat[i, gdf.columns.get_loc("item_id")]
        row = mbdf[mbdf["asin"] == item_id].iloc[0]
        
        if type(row["title"]) == str:
            title = process_title(row["title"])
            item["title"] = title
        if type(row["categories"]) == str:
            categories = process_categories(row["categories"])
            item["categories"] = categories

        item["item_id"] = item_id
    
    columns = [item["item_id"]] + item["title"] + item["categories"]
    return pd.Series(columns)

def gen_new_column_names(index, domain):
    names = [f"{domain}_id_{index}"]
    for i in range(10):
        names.append(f"{domain}_title_{index}_{i}")
    for i in range(10):
        names.append(f"{domain}_cat_{index}_{i}")
    
    return names

def meta_column_names():
    names = ["item_id"]
    for i in range(10):
        names.append(f"title_{i}")
    for i in range(10):
        names.append(f"cat_{i}")
    return names

def get_book_meta(item_id):
    item = {"item_id": 0, "title": [0]*10, "categories": [0]*10}
    row = mbdf[mbdf["asin"] == item_id].iloc[0]

    if type(row["title"]) == str:
        title = process_title(row["title"])
        item["title"] = title
    if type(row["categories"]) == str:
        categories = process_categories(row["categories"])
        item["categories"] = categories

    item["item_id"] = item_id
    columns = [item["item_id"]] + item["title"] + item["categories"]
    return pd.Series(columns)

def get_movie_meta(item_id):
    item = {"item_id": 0, "title": [0]*10, "categories": [0]*10}
    row = mmdf[mmdf["asin"] == item_id].iloc[0]

    if type(row["title"]) == str:
        title = process_title(row["title"])
        item["title"] = title
    if type(row["categories"]) == str:
        categories = process_categories(row["categories"])
        item["categories"] = categories

    item["item_id"] = item_id
    columns = [item["item_id"]] + item["title"] + item["categories"]
    return pd.Series(columns)

def add_book_meta(save=True):
    global bdf
    bdf = rbdf.copy()
    bdf[meta_column_names()] = rbdf.parallel_apply(lambda row: get_book_meta(row["item_id"]), axis=1)
    
    if save:
        bdf.to_csv("ratings_Books_wmeta.csv")

def add_movie_meta(save=True):
    global mdf
    mdf[meta_column_names()] = rmdf.parallel_apply(lambda row: get_movie_meta(row["item_id"]), axis=1)
    
    if save:
        mdf.to_csv("ratings_Movie_wmeta.csv")

def add_history(save=True):
    global gbdf, gmdf, rbdf, rmdf, mdf

    gbdf = rbdf.sort_values(["user_id", "timestamp"], ascending=False)[rbdf["rating"] > 3].groupby("user_id")
    gmdf = rmdf.sort_values(["user_id", "timestamp"], ascending=False)[rmdf["rating"] > 3].groupby("user_id")

    uid = rmdf.iat[0, rmdf.columns.get_loc("user_id")]
    ts = gmdf.get_group(uid).iat[0, rmdf.columns.get_loc("timestamp")]
    
    print(gmdf.get_group(uid))
    print(get_target_item(uid, ts, 0))
    print(gen_new_column_names(0, "movie"))
    print(ts)

    mdf = rmdf.copy()

    for i in range(10):
       mdf[gen_new_column_names(i, "movie")] = rmdf.parallel_apply(lambda row: get_target_item(row["user_id"], row["timestamp"], i), axis=1)
       print("target - ", i, " is done.")

    for i in range(20):
        mdf[gen_new_column_names(i, "book")] = rmdf.parallel_apply(lambda row: get_source_item(row["user_id"], row["timestamp"], i), axis=1)
        print("source - ", i, " is done.")

    mdf.sort_values(["user_id", "timestamp"])
    
    if save:
        mdf.to_csv("ratings_Movies_history_metadata.csv", index=False)
        print("Target and source history are added.")

def init_data(phase=0):
    global rmdf, rbdf, mmdf, mbdf, mdf, bdf

    if phase == 1:
        mdf = pd.read_csv("ratings_Movie_wmeta_orderfixed.csv")
        bdf = pd.read_csv("ratings_Books_wmeta.csv")
        
        fix_column_order()
        
        return

    elif phase == 2:
        mdf = pd.read_csv("ratings_Movie_asindex_ctr.csv")
        bdf = pd.read_csv("ratings_Books_asindex_ctr.csv")

        fix_column_order()

        return

    rbdf = pd.read_csv("raw-data/ratings_Books_enough_data.csv")
    rmdf = pd.read_csv("raw-data/ratings_Movies_enough_data.csv")

    mbdf = pd.read_csv(meta_books)
    mmdf = pd.read_csv(meta_movies)
    
    print("Dataset is read.")

    rbdf = rbdf.drop(columns=rbdf.columns.difference(["user_id", "item_id", "rating", "timestamp"]))
    rmdf = rmdf.drop(columns=rmdf.columns.difference(["user_id", "item_id", "rating", "timestamp"]))
    
    mmdf = mmdf.drop(columns=mmdf.columns.difference(["asin", "categories", "title", "brand"]))
    mbdf = mbdf.drop(columns=mbdf.columns.difference(["asin", "categories", "title", "brand"]))

def json2csv():
    mbdf = df_fromjson("raw-data/meta_Books.json.gz")
    mbdf = mbdf.drop(columns=["salesRank", "price", "imUrl", "description", "related"])
    mbdf.to_csv("raw-data/meta_Books.csv")

def test_shared():
    rbdf = pd.read_csv("raw-data/ratings_Books_enough_data.csv")
    rmdf = pd.read_csv("raw-data/ratings_Movies_enough_data.csv")

    print(rbdf[rbdf["user_id"] == rbdf["user_id"][2]])

def fix_column_order():
    global mdf, bdf
    
    columns = ["timestamp", "rating", "user_id", "item_id"]
    columns += meta_column_names()

    for i in range(10):
        columns += gen_new_column_names(i, "movie")

    for i in range(20):
        columns += gen_new_column_names(i, "book")
    
    mdf = mdf.drop(columns=mdf.columns.difference(columns))
    mdf = mdf[columns]

    columns = ["timestamp", "rating", "user_id", "item_id"]
    columns += meta_column_names()

    bdf = bdf.drop(columns=bdf.columns.difference(columns))
    bdf = bdf[columns]

def get_index(value):
    global index_map, last_index

    value = str(value)
    if value == '0':
        return 0
    
    if value in index_map:
        return index_map[value]
    
    last_index = last_index + 1
    index_map[value] = last_index
    
    return last_index

def encode_as_index(save=False):
    global mdf, bdf
    exclude_cols = ["timestamp", "rating"]
    mdf.loc[:, ~mdf.columns.isin(exclude_cols)] = mdf.loc[:, ~mdf.columns.isin(exclude_cols)].applymap(get_index)
    print("Movie dataset is encoded as index.")
    bdf.loc[:, ~bdf.columns.isin(exclude_cols)] = bdf.loc[:, ~bdf.columns.isin(exclude_cols)].applymap(get_index)
    print("Book dataset is encoded as index.")

    if save:
        mdf.to_csv("ratings_Movie_asindex.csv")
        bdf.to_csv("ratings_Book_asindex.csv")

def convert_rating_to_clicks(save=False):
    mdf["rating"] = mdf["rating"].apply(lambda x: 1 if x > 3 else 0)
    bdf["rating"] = bdf["rating"].apply(lambda x: 1 if x > 3 else 0)

    if save:
        mdf.to_csv("ratings_Movie_asindex_ctr.csv")
        bdf.to_csv("ratings_Books_asindex_ctr.csv")

def split_dataset(save=True):
    global mdf, bdf

    mdf.sort_values(by="timestamp")
    bdf.sort_values(by="timestamp")

    mdf = mdf.drop(columns=["timestamp"])
    bdf = bdf.drop(columns=["timestamp"])

    movie_train_size = int(0.8 * len(mdf))
    movie_val_size = int(0.1 * len(mdf))
    movie_test_size = len(mdf) - movie_train_size - movie_val_size

    book_train_size = int(0.8 * len(bdf))
    book_val_size = int(0.1 * len(bdf))
    book_test_size = len(bdf) - book_train_size - book_val_size

    train_mdf = mdf.iloc[:movie_train_size]
    valid_mdf = mdf.iloc[movie_train_size:movie_train_size + movie_val_size]
    test_mdf = mdf.iloc[movie_train_size + movie_val_size:]

    train_bdf = bdf.iloc[:book_train_size]
    valid_bdf = bdf.iloc[book_train_size : book_train_size + book_val_size]
    test_bdf = bdf.iloc[book_train_size + book_val_size :]

    if save:
        train_mdf.to_csv("data-amazon/Movies_train.csv", index=False, header=False)
        valid_mdf.to_csv("data-amazon/Movies_val.csv", index=False, header=False)
        test_mdf.to_csv("data-amazon/Movies_test.csv", index=False, header=False)

        train_bdf.to_csv("data-amazon/Books_train.csv", index=False, header=False)
        valid_bdf.to_csv("data-amazon/Books_val.csv", index=False, header=False)
        test_bdf.to_csv("data-amazon/Books_test.csv", index=False, header=False)

if __name__ == "__main__":
    init_data(phase=0)
    print(rmdf.nunique())
    print(rbdf.nunique())


   
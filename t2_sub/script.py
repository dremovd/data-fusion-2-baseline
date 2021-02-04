import pandas as pd
import pickle
from joblib import Parallel, delayed
N_JOBS=8


def find_all_brands(names_series, brands_re):
    return names_series.str.findall(brands_re)


def find_first_brand(names_series, brands_re):
    brands = find_all_brands(names_series, brands_re)
    first_brand = brands.map(
        lambda b: b[0] if len(b) > 0 else 'отсутствие бренда'
    )
    return first_brand


def find_parallel(names_series, brands_re, n_jobs=N_JOBS):
    n_batches = n_jobs
    batches = [names_series.iloc[i::n_batches] for i in range(n_batches)]
    brand = Parallel(n_jobs=n_jobs)(
        delayed(find_first_brand)(batch, brands_re)
        for batch in batches
    )
    brand = pd.concat(brand)
    item_brand_mapping = pd.concat([names_series, brand], axis=1, ignore_index=False)
    item_brand_mapping.columns = ['item_name', 'brands']
    return item_brand_mapping


test = pd.read_parquet('data/task2_test_for_user.parquet')
assert 'id' in test.columns

test_unique = test['item_name'].drop_duplicates()
test_unique = test_unique.str.lower()
test_unique = test_unique.drop_duplicates()

brands_re = pickle.load(open('brands_re', 'rb'))

item2brand = find_parallel(test_unique, brands_re)
item2brand = dict(zip(item2brand['item_name'].values, item2brand['brands'].values))

test['pred'] = test['item_name'].str.lower().apply(lambda x: item2brand.get(x, 'отсутствие бренда'))
test[['id', 'pred']].to_csv('answers.csv', index=None)
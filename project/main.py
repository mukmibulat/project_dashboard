import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from segment_tree import SegmentTreeIndex
from hash_map import build_hash_index, build_product_hash_index
from pagerank import get_top_products
from binary_heap import get_top_selling_products
from bootstrap_analysis import analyze_customer_spending

app = FastAPI()

df_hash = pd.read_csv('data/amazon.csv')
keys = df_hash['OrderID'].astype(str)
hash_index = build_hash_index(keys)

product_keys = df_hash['ProductID'].astype(str)
product_hash_index = build_product_hash_index(product_keys)

df_segment = pd.read_csv('data/amazon.csv')
df_segment = df_segment.sort_values('OrderDate').reset_index(drop=True)

segment_index = SegmentTreeIndex(df_segment)

app.mount(
    '/static',
    StaticFiles(directory='static'),
    name='static'
)


@app.get('/')
def home():
    return FileResponse('static/index.html')

@app.get('/api/hash-map/{order_id}')
def get_order_by_id(order_id: str):
    row_id = hash_index.get(order_id)

    if row_id is None:
        raise HTTPException(
            status_code=404,
            detail='order не найден'
        )

    return df_hash.iloc[row_id].to_dict()


@app.get('/api/segment-tree/sum')
def get_quantity_sum(left: int, right: int):
    if left < 0 or right >= len(segment_index.values) or left > right:
        raise HTTPException(
            status_code=400,
            detail='Invalid range'
        )

    total = segment_index.range_query(left, right)

    return {
        'left': left,
        'right': right,
        'quantity_sum': total
    }


@app.put('/api/segment-tree/update')
def update_quantity(pos: int, value: int):
    if pos < 0 or pos >= len(segment_index.values):
        raise HTTPException(
            status_code=400,
            detail='Invalid position'
        )

    segment_index.update(pos, value)

    return {
        'position': pos,
        'new_value': value,
        'message': 'quantity обновлено'
    }

@app.get('/api/pagerank')
def get_pagerank():
    return get_top_products(df_hash)

@app.get('/api/hash-map/product/{product_id}')
def get_product_by_id(product_id: str):
    row_ids = product_hash_index.get(product_id)

    if row_ids is None:
        raise HTTPException(
            status_code=404,
            detail='Товар не найден'
        )

    product_rows = df_hash.iloc[row_ids]

    return {
        'ProductID': product_id,
        'ProductName': product_rows.iloc[0]['ProductName'],
        'total_quantity': int(product_rows['Quantity'].sum()),
        'orders_count': int(product_rows['OrderID'].nunique())
    }

@app.get('/api/heap/top-products')
def get_heap_top_products():
    return get_top_selling_products(df_hash, 10)

@app.get('/api/bootstrap')
def get_bootstrap_analysis(repeats: int = 1000):
    if repeats < 100 or repeats > 10000:
        raise HTTPException(
            status_code=400,
            detail='Количество повторений должно быть от 100 до 10000'
        )

    try:
        return analyze_customer_spending(df_hash, repeats)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
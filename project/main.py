import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from segment_tree import SegmentTreeIndex
from hash_map import build_hash_index, build_product_hash_index
from pagerank import get_top_products
from binary_heap import get_top_selling_products
from bootstrap_analysis import analyze_customer_spending

app = FastAPI()

df = None
hash_index = None
product_hash_index = None
segment_index = None

app.mount(
    '/static',
    StaticFiles(directory='static'),
    name='static'
)


def check_dataset():
    if df is None:
        raise HTTPException(
            status_code=400,
            detail='Сначала загрузите CSV'
        )


@app.get('/')
def home():
    return FileResponse('static/index.html')


@app.post('/api/dataset')
def upload_dataset(file: UploadFile = File(...)):
    global df, hash_index, product_hash_index, segment_index

    try:
        new_df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail='Не удалось прочитать CSV'
        )

    df = new_df
    hash_index = build_hash_index(df['OrderID'].astype(str))
    product_hash_index = build_product_hash_index(df['ProductID'].astype(str))

    df_sorted = df.sort_values('OrderDate').reset_index(drop=True)
    segment_index = SegmentTreeIndex(df_sorted)

    return {
        'rows': len(df),
        'message': 'Датасет загружен'
    }


@app.delete('/api/dataset')
def reset_dataset():
    global df, hash_index, product_hash_index, segment_index

    df = None
    hash_index = None
    product_hash_index = None
    segment_index = None

    return {'message': 'Датасет сброшен'}


@app.get('/api/hash-map/{order_id}')
def get_order_by_id(order_id: str):
    check_dataset()

    row_id = hash_index.get(order_id)

    if row_id is None:
        raise HTTPException(
            status_code=404,
            detail='order не найден'
        )

    return df.iloc[row_id].to_dict()


@app.get('/api/segment-tree/sum')
def get_quantity_sum(left: int, right: int):
    check_dataset()

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
    check_dataset()

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
    check_dataset()
    return get_top_products(df)

@app.get('/api/hash-map/product/{product_id}')
def get_product_by_id(product_id: str):
    check_dataset()

    row_ids = product_hash_index.get(product_id)

    if row_ids is None:
        raise HTTPException(
            status_code=404,
            detail='Товар не найден'
        )

    product_rows = df.iloc[row_ids]

    return {
        'ProductID': product_id,
        'ProductName': product_rows.iloc[0]['ProductName'],
        'total_quantity': int(product_rows['Quantity'].sum()),
        'orders_count': int(product_rows['OrderID'].nunique())
    }

@app.get('/api/heap/top-products')
def get_heap_top_products():
    check_dataset()
    return get_top_selling_products(df, 10)

@app.get('/api/bootstrap')
def get_bootstrap_analysis(repeats: int = 1000):
    check_dataset()

    if repeats < 100 or repeats > 10000:
        raise HTTPException(
            status_code=400,
            detail='Количество повторений должно быть от 100 до 10000'
        )

    try:
        return analyze_customer_spending(df, repeats)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
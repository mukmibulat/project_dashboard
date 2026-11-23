import numpy as np


def bootstrap(values, repeats=1000, statistic=np.mean, seed=42):
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(repeats):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        results.append(statistic(sample))

    return np.array(results)


def analyze_customer_spending(df, repeats=1000):
    delivered = df[df['OrderStatus'] == 'Delivered']

    customer_spending = (
        delivered.groupby('CustomerID')['TotalAmount'].sum()
    )

    all_customers = df['CustomerID'].drop_duplicates()

    customer_spending = customer_spending.reindex(
        all_customers,
        fill_value=0
    )

    values = customer_spending.to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        raise ValueError('Недостаточно данных для Bootstrap')

    boot = bootstrap(
        values,
        repeats=repeats,
        statistic=np.mean,
        seed=42
    )

    observed = np.mean(values)
    low, high = np.percentile(boot, [2.5, 97.5])

    return {
        'customers_count': len(values),
        'repeats': repeats,
        'mean_spending': round(float(observed), 2),
        'confidence_interval': {
            'lower': round(float(low), 2),
            'upper': round(float(high), 2)
        }
    }

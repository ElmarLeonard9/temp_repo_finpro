import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Seller reputation lookup (train-time build + refreshable serve-time load)
# ---------------------------------------------------------------------------

def build_seller_lookup(items_df, k_smooth=10):
    """Build a seller -> smoothed late-rate lookup table from item-level rows.

    Uses additive (Laplace-style) smoothing toward the global late rate so
    sellers with very few shipments don't get an extreme 0%/100% rate.
    """
    items_df = items_df.sort_values(['seller_id', 'order_purchase_timestamp']).copy()
    items_df['seller_late_item'] = (
        items_df['shipping_limit_date'] < items_df['order_delivered_carrier_date']
    ).astype(int)

    global_late_rate = items_df['seller_late_item'].mean()

    seller_lookup = (
        items_df.groupby('seller_id')['seller_late_item']
        .agg(total_shipments='count', late_shipments='sum')
    )
    seller_lookup['seller_late_rate'] = (
        (seller_lookup['late_shipments'] + k_smooth * global_late_rate)
        / (seller_lookup['total_shipments'] + k_smooth)
    )
    return seller_lookup, global_late_rate


def save_seller_lookup(seller_lookup, global_late_rate, out_dir):
    """Save a dated snapshot plus a `_latest` pointer for the seller lookup table."""
    os.makedirs(out_dir, exist_ok=True)
    tag = date.today().isoformat()
    dated_path = f'{out_dir}/seller_lookup_{tag}.joblib'
    latest_path = f'{out_dir}/seller_lookup_latest.joblib'

    bundle = {'seller_lookup': seller_lookup, 'global_late_rate': global_late_rate}
    joblib.dump(bundle, dated_path)
    joblib.dump(bundle, latest_path)
    return dated_path


def load_seller_lookup(path):
    bundle = joblib.load(path)
    return bundle['seller_lookup'], bundle['global_late_rate']


def refresh_seller_lookup(latest_items_df, out_dir, k_smooth=10):
    """Rebuild the lookup table from fresh data and save it. Run on a schedule."""
    lookup, global_rate = build_seller_lookup(latest_items_df, k_smooth)
    path = save_seller_lookup(lookup, global_rate, out_dir)
    print(f'Refreshed seller lookup: {lookup.shape[0]} sellers, saved to {path}')
    return lookup, global_rate


def attach_seller_late_rate_frozen(items_df, lookup, global_rate):
    """Attach each row's seller_late_rate from a frozen (already-built) lookup table.
    Unseen sellers (not present in `lookup`) fall back to `global_rate`.
    """
    merged = items_df.merge(
        lookup['seller_late_rate'], on='seller_id', how='left'
    )
    merged['seller_late_rate'] = merged['seller_late_rate'].fillna(global_rate)
    return merged


# ---------------------------------------------------------------------------
# Standalone feature helpers
# ---------------------------------------------------------------------------

def convert_bool_to_int(df):
    """Used inside preprocessor's boolean_pipeline (a FunctionTransformer)."""
    return df.astype(int)

def calculate_haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two lat/lon points."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    r_earth = 6371
    return r_earth * c


def get_black_friday_date(year):
    november = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30', freq='D')
    fourth_thursday = november[november.weekday == 3][3]
    return (fourth_thursday + pd.Timedelta(days=1)).normalize()


def is_black_friday_or_holiday(ts, black_friday_window_days=2):
    black_friday = get_black_friday_date(ts.year)
    is_black_friday_period = abs((ts.normalize() - black_friday).days) <= black_friday_window_days
    is_december_holiday = ts.month == 12
    return bool(is_black_friday_period or is_december_holiday)


# ---------------------------------------------------------------------------
# Item-level -> order-level feature aggregation
# ---------------------------------------------------------------------------

def aggregate_order_features(items_df, seller_lookup=None, global_late_rate=None):
    """Aggregate raw, item-level order rows into one row per order.
    """
    feat = items_df.copy()
    feat = attach_seller_late_rate_frozen(feat, seller_lookup, global_late_rate)
    feat['item_volume_cm3'] = (
        feat['product_length_cm'] * feat['product_height_cm'] * feat['product_width_cm']
    )
    feat['seller_customer_distance_km'] = calculate_haversine(
        feat['seller_lat'], feat['seller_lng'], feat['customer_lat'], feat['customer_lng']
    )

    feat['purchase_dow'] = feat['order_purchase_timestamp'].dt.day_name()
    feat['purchase_month'] = feat['order_purchase_timestamp'].dt.month_name()
    feat['purchase_hour'] = feat['order_purchase_timestamp'].dt.hour
    feat['promised_days'] = (feat['order_estimated_delivery_date'] - feat['order_purchase_timestamp']).dt.days
    feat['is_black_friday_or_holiday'] = feat['order_purchase_timestamp'].apply(is_black_friday_or_holiday)

    dominant = feat.loc[feat.groupby('order_id')['price'].idxmax()].set_index('order_id')

    out = pd.DataFrame({
        'total_price': feat.groupby('order_id')['price'].sum(),
        'total_freight_value': feat.groupby('order_id')['freight_value'].sum(),
        'total_weight_g': feat.groupby('order_id')['product_weight_g'].sum(),
        'total_volume_cm3': feat.groupby('order_id')['item_volume_cm3'].sum(),

        'n_items': feat.groupby('order_id').size(),

        'n_distinct_products': feat.groupby('order_id')['product_id'].nunique(),
        'n_distinct_categories': feat.groupby('order_id')['product_category_name_english'].nunique(),
        'n_distinct_sellers': feat.groupby('order_id')['seller_id'].nunique(),

        'seller_customer_distance_km': feat.groupby('order_id')['seller_customer_distance_km'].max(),

        'order_purchase_timestamp': feat.groupby('order_id')['order_purchase_timestamp'].first(),
        'order_estimated_delivery_date': feat.groupby('order_id')['order_estimated_delivery_date'].first(),
        'purchase_dow': feat.groupby('order_id')['purchase_dow'].first(),
        'purchase_month': feat.groupby('order_id')['purchase_month'].first(),
        'purchase_hour': feat.groupby('order_id')['purchase_hour'].first(),
        'promised_days': feat.groupby('order_id')['promised_days'].first(),
        'is_black_friday_or_holiday': feat.groupby('order_id')['is_black_friday_or_holiday'].first(),
        'customer_geo_state': feat.groupby('order_id')['customer_geo_state'].first(),
        'payment_type': feat.groupby('order_id')['payment_type'].first(),
        'payment_installments': feat.groupby('order_id')['payment_installments'].first(),
        'total_payment_value': feat.groupby('order_id')['total_payment_value'].first(),
        'n_vouchers': feat.groupby('order_id')['n_vouchers'].first(),
        'voucher_value': feat.groupby('order_id')['voucher_value'].first(),
    })

    if 'is_late' in feat.columns:
        out['is_late'] = feat.groupby('order_id')['is_late'].first().astype(int)

    out['product_category'] = dominant['product_category_name_english'].reindex(out.index)
    out['seller_geo_state'] = dominant['seller_geo_state'].reindex(out.index)
    out['seller_late_rate'] = dominant['seller_late_rate'].reindex(out.index)

    bins = [-1, 50, 300, 1000, 7000]
    labels = ['Short-Range (<50km)', 'Regional (50-300km)', 'Interregional (300-1000km)', 'Long-Haul (>1000km)']

    out['distance_group'] = pd.cut(out['seller_customer_distance_km'], bins=bins, labels=labels)

    out['is_interstate'] = out['seller_geo_state'] != out['customer_geo_state']
    out['route'] = out['seller_geo_state'] + '_to_' + out['customer_geo_state']

    return out


class OrderFeatureAggregator(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around aggregate_order_features.

    fit() is a no-op by design: seller_lookup and global_late_rate are
    injected at construction time (frozen at training time, or freshly
    loaded at serving time), never refit on whatever data passes through.
    """

    def __init__(self, seller_lookup=None, global_late_rate=None):
        self.seller_lookup = seller_lookup
        self.global_late_rate = global_late_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return aggregate_order_features(X, self.seller_lookup, self.global_late_rate)


# ---------------------------------------------------------------------------
# Serving wrapper
# ---------------------------------------------------------------------------

class ServingModel:
    """Wraps the calibrated model + seller lookup so callers can call
    predict_proba directly on raw, item-level rows -- the aggregation
    (which changes row count: items -> orders) happens as an explicit
    step before scoring, rather than living inside the object that gets
    calibrated. CalibratedClassifierCV pre-allocates its output array
    from the *input* row count, so a row-count-changing step inside it
    breaks -- this wrapper is what keeps the one-call predict_proba
    interface working despite that constraint.
    """

    def __init__(self, calibrated_pipeline, seller_lookup, global_late_rate):
        self.calibrated_pipeline = calibrated_pipeline
        self.aggregator = OrderFeatureAggregator(
            seller_lookup=seller_lookup,
            global_late_rate=global_late_rate,
        )

    def predict_proba(self, raw_items_df):
        order_level_df = self.aggregator.transform(raw_items_df)
        return self.calibrated_pipeline.predict_proba(order_level_df)

    def predict(self, raw_items_df, threshold=0.5):
        return (self.predict_proba(raw_items_df)[:, 1] >= threshold).astype(int)


def load_serving_model(model_path, lookup_path):
    """Load the pickled calibrated pipeline + the latest seller lookup snapshot,
    and combine them into a ServingModel ready for predict_proba on raw rows."""
    import pickle
    calibrated_pipeline = pickle.load(open(model_path, 'rb'))
    lookup, global_rate = load_seller_lookup(lookup_path)
    return ServingModel(calibrated_pipeline, lookup, global_rate)

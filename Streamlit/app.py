"""
Late Delivery Predictor - Streamlit app
=========================================
Two prediction modes:
  1. Batch prediction: 
     upload a dataset of raw, item-level order rows and
     get a late/on-time prediction for every order in it.
  2. Single order:  
     fill in a form for one order (one or more items),
     get an instant prediction.

Run with:
    streamlit run app.py
"""

import io
import pickle
from datetime import date, datetime, time as dtime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

sys.path.append(os.path.abspath("..")) 

from Utils.serving_utils import ServingModel

APP_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Late Delivery Predictor", page_icon="📦", layout="wide")

RAW_COLUMNS = [
    "order_id",
    "seller_id",
    "product_id",
    "product_category_name_english",
    "price",
    "freight_value",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "seller_lat",
    "seller_lng",
    "seller_geo_state",
    "customer_lat",
    "customer_lng",
    "customer_geo_state",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "payment_type",
    "payment_installments",
    "total_payment_value",
    "n_vouchers",
    "voucher_value",
]

BR_STATES = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]

# Batch template shows both zip-code and lat/lng/geo_state columns: fill in
# the zip code and coordinates+state are looked up automatically, or fill in
# lat/lng/geo_state directly to skip the lookup (or as a per-row fallback for
# a zip that isn't recognized).
BATCH_TEMPLATE_COLUMNS = [
    "order_id", "seller_id", "product_id", "product_category_name_english",
    "price", "freight_value", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
    "seller_zip_code", "seller_lat", "seller_lng", "seller_geo_state",
    "customer_zip_code", "customer_lat", "customer_lng", "customer_geo_state",
    "order_purchase_timestamp", "order_estimated_delivery_date",
    "payment_type", "payment_installments", "total_payment_value",
    "n_vouchers", "voucher_value",
]

# lat/lng and geo_state are only strictly required if no zip code column is
# supplied instead (the zip code can derive both).
CORE_REQUIRED_COLUMNS = [
    c for c in RAW_COLUMNS
    if c not in (
        "seller_lat", "seller_lng", "seller_geo_state",
        "customer_lat", "customer_lng", "customer_geo_state",
    )
]

ITEM_COLUMN_DEFAULTS = {
    "seller_id": "",
    "product_id": "",
    "product_category_name_english": "",
    "price": 0.0,
    "freight_value": 0.0,
    "product_weight_g": 0.0,
    "product_length_cm": 0.0,
    "product_height_cm": 0.0,
    "product_width_cm": 0.0,
    "seller_zip_code": "",
    "seller_geo_state": "",
}


# ---------------------------------------------------------------------------
# Zip code -> coordinates lookup
# ---------------------------------------------------------------------------

def _find_col(columns, keywords):
    for c in columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            return c
    return None


def normalize_zip(zip_code):
    """Extract digits and take the first 5 as the CEP prefix, zero-padded."""
    if zip_code is None:
        return None
    digits = "".join(ch for ch in str(zip_code) if ch.isdigit())
    if not digits:
        return None
    return digits[:5].zfill(5)


@st.cache_resource(show_spinner="Loading zip code lookup...")
def load_zip_lookup(csv_bytes: bytes):
    """Parse a zip->coordinates CSV into (lookup dict, state-average fallback dict).
    Flexibly matches column names containing zip/cep, lat, lng/lon, state/uf."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    zip_col = _find_col(df.columns, ["zip", "cep"])
    lat_col = _find_col(df.columns, ["lat"])
    lng_col = _find_col(df.columns, ["lng", "lon"])
    state_col = _find_col(df.columns, ["state", "uf"])
    if not all([zip_col, lat_col, lng_col, state_col]):
        raise ValueError(
            "Couldn't find zip/latitude/longitude/state columns in that file."
        )
    df = df[[zip_col, lat_col, lng_col, state_col]].copy()
    df.columns = ["zip", "lat", "lng", "state"]
    df["zip"] = df["zip"].apply(normalize_zip)
    df = df.dropna(subset=["zip"])
    lookup = df.drop_duplicates("zip").set_index("zip")[["lat", "lng", "state"]].to_dict("index")
    state_fallback = df.groupby("state")[["lat", "lng"]].mean().to_dict("index")
    return lookup, state_fallback


def resolve_location(zip_code, manual_state, zip_lookup, state_fallback):
    """Resolve (lat, lng, state, status) for a zip code.

    The zip code is the primary source for BOTH coordinates and state.
    `manual_state` is only used as a fallback when the zip isn't recognized.

    status is one of:
      'zip_exact'  - exact zip match found (most accurate)
      'state_avg'  - zip not found, used manual_state's average coordinates
      'unresolved' - zip not found AND no usable manual_state given
    """
    z = normalize_zip(zip_code)
    if zip_lookup and z and z in zip_lookup:
        entry = zip_lookup[z]
        return entry["lat"], entry["lng"], entry["state"], "zip_exact"

    state = (str(manual_state).strip().upper()) if manual_state and str(manual_state).strip() else None
    if state and state_fallback and state in state_fallback:
        entry = state_fallback[state]
        return entry["lat"], entry["lng"], state, "state_avg"
    if state:
        # a state was given but the lookup file has no rows for it -> can't average
        return 0.0, 0.0, state, "state_avg"
    return None, None, None, "unresolved"


def enrich_location_batch(df, prefix, zip_lookup, state_fallback):
    """For a batch dataframe, fill f'{prefix}_lat'/f'{prefix}_lng'/f'{prefix}_geo_state'
    from f'{prefix}_zip_code'. A zip-resolved state always overwrites whatever is
    in the geo_state column (the zip is authoritative); geo_state is only used
    as this row's fallback when its zip isn't found. Explicit lat/lng that's
    already filled in is left untouched. Returns (df, n_filled, n_approx, unresolved_idx)."""
    zip_col = f"{prefix}_zip_code"
    lat_col = f"{prefix}_lat"
    lng_col = f"{prefix}_lng"
    state_col = f"{prefix}_geo_state"

    if lat_col not in df.columns:
        df[lat_col] = np.nan
    if lng_col not in df.columns:
        df[lng_col] = np.nan
    if state_col not in df.columns:
        df[state_col] = np.nan

    n_filled = 0
    n_approx = 0
    unresolved_idx = []

    for idx, row in df.iterrows():
        has_latlng = pd.notna(row[lat_col]) and pd.notna(row[lng_col])
        manual_state = row[state_col] if pd.notna(row[state_col]) else None
        zip_code = row.get(zip_col) if zip_col in df.columns else None

        if has_latlng and manual_state:
            continue  # both already explicit, nothing to resolve

        lat, lng, state, status = resolve_location(zip_code, manual_state, zip_lookup, state_fallback)

        if status == "unresolved":
            unresolved_idx.append(idx)
            continue

        if not has_latlng:
            df.at[idx, lat_col] = lat
            df.at[idx, lng_col] = lng
            n_filled += 1
        df.at[idx, state_col] = state  # zip-resolved state is authoritative
        if status == "state_avg":
            n_approx += 1

    return df, n_filled, n_approx, unresolved_idx


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(model_bytes: bytes):
    return pickle.loads(model_bytes)


@st.cache_resource(show_spinner="Loading seller lookup...")
def load_lookup(lookup_bytes: bytes):
    bundle = joblib.load(io.BytesIO(lookup_bytes))
    return bundle["seller_lookup"], bundle["global_late_rate"]


def get_serving_model():
    model = st.session_state.get("model")
    lookup = st.session_state.get("lookup")
    global_rate = st.session_state.get("global_rate")
    if model is None or lookup is None:
        return None
    return ServingModel(model, lookup, global_rate)


# ---------------------------------------------------------------------------
# Sidebar: model + lookup + zip lookup + threshold
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Setup")

st.sidebar.subheader("1. Model artifacts")
model_file = st.sidebar.file_uploader("deploy_pipeline.pkl", type=["pkl"])
lookup_file = st.sidebar.file_uploader("seller_lookup_latest.joblib", type=["joblib"])

if model_file is not None:
    st.session_state["model"] = load_model(model_file.getvalue())
if lookup_file is not None:
    lookup, global_rate = load_lookup(lookup_file.getvalue())
    st.session_state["lookup"] = lookup
    st.session_state["global_rate"] = global_rate

model_ready = st.session_state.get("model") is not None
lookup_ready = st.session_state.get("lookup") is not None

col_a, col_b = st.sidebar.columns(2)
col_a.markdown(f"Model: {'✅' if model_ready else '⬜'}")
col_b.markdown(f"Lookup: {'✅' if lookup_ready else '⬜'}")

st.sidebar.subheader("2. Zip code → coordinates (optional)")
zip_upload = st.sidebar.file_uploader(
    "Zip code coordinates CSV", type=["csv"],
    help="Columns for zip code, city, state, latitude, longitude. "
         "If left empty, the app auto-loads brazil_zip_codes_coordinates.csv "
         "bundled next to app.py, if present.",
)

zip_lookup = zip_state_fallback = None
zip_source = None
zip_error = None
try:
    if zip_upload is not None:
        zip_lookup, zip_state_fallback = load_zip_lookup(zip_upload.getvalue())
        zip_source = zip_upload.name
    else:
        default_zip_path = APP_DIR / "Dataset" / "Supporting Data" / "brazil_zip_codes_coordinates_full.csv"
        if default_zip_path.exists():
            zip_lookup, zip_state_fallback = load_zip_lookup(default_zip_path.read_bytes())
            zip_source = "bundled brazil_zip_codes_coordinates_full.csv"
except Exception as e:
    zip_error = str(e)

if zip_error:
    st.sidebar.error(f"Couldn't load zip lookup: {zip_error}")
elif zip_lookup:
    st.sidebar.caption(
        f"📍 {len(zip_lookup)} zip codes loaded from *{zip_source}*. "
        "Zips not in the table fall back to that state's average coordinates."
    )
else:
    st.sidebar.caption("📍 No zip lookup loaded — enter latitude/longitude manually.")

st.sidebar.subheader("3. Prediction threshold")
late_threshold = st.sidebar.slider(
    "Probability threshold for 'LATE'", 0.0, 1.0, 0.076, 0.001,
    help="An order is flagged LATE if predicted probability >= this value. "
         "Default 0.076 matches the operating threshold chosen for this model.",
)

serving_model = get_serving_model()

st.title("📦 Late Delivery Predictor")

if not (model_ready and lookup_ready):
    st.info("👈 Upload **deploy_pipeline.pkl** and **seller_lookup_latest.joblib** in the sidebar to get started.")

# ---------------------------------------------------------------------------
# About the model (shown above both tabs)
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.markdown("#### 🧠 About this model")
    st.markdown(
        "This is a **binary classifier** trained on the Olist Brazilian e-commerce "
        "dataset. For each order it estimates the probability that the order is "
        "**delivered later than its estimated delivery date**. It is a *pre-shipment* "
        "model: it only uses information available at purchase time (items, price, "
        "freight, product dimensions, payment, seller/customer location, the promised "
        "delivery date), so it can be run the moment an order comes in (before any "
        "carrier tracking event exists)."
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            "**Input granularity**  \n"
            "Raw rows are *item-level* (one row per order item). The app aggregates "
            "them into one row per `order_id` before scoring (sums for price, "
            "freight, weight and volume, counts for items/products/categories/sellers, "
            "dominant category and primary seller taken from the highest-priced item, "
            "and the maximum seller→customer distance)."
        )
    with a2:
        st.markdown(
            "**Engineered features**  \n"
            "Geolocation is turned into seller→customer distance from zip-code "
            "coordinates. Timestamps become calendar features and the promised "
            "delivery window. Each seller carries a *historical late-to-carrier rate* "
            "with shrinkage toward the global rate, so sellers with few past orders "
            "aren't over-trusted; unseen sellers fall back to the global rate."
        )
    with a3:
        st.markdown(
            "**Leakage controls**  \n"
            "Train/test split is **time-based** (a purchase-date cutoff, not random), "
            "so the model is always evaluated on orders that happened *after* the ones "
            "it learned from. The seller history feature is computed with an "
            "expanding window on training rows and a frozen lookup table at serving "
            "time which is the same lookup this app loads from the sidebar."
        )

    with st.expander("Why the default threshold is 0.076, not 0.5"):
        st.markdown(
            "Late deliveries are the **minority class**, and the two mistakes cost "
            "very different amounts. A false negative (a late order you never flagged) "
            "costs a refund, a support ticket and a churned customer; a false positive "
            "costs one unnecessary follow-up. So the operating point isn't chosen to "
            "maximise accuracy (it's chosen by **expected value**), scoring the "
            "confusion matrix against the cost/benefit of each cell "
            "(Provost & Fawcett's expected-profit framework) and comparing the model "
            "against the two trivial baselines: *do nothing* and *chase every order*.\n\n"
            "A low threshold means the model is deliberately tuned for **recall**. Where "
            "it would rather over-flag than miss a late order. Raise the slider in the "
            "sidebar to flag fewer orders with more confidence in each; lower it to "
            "catch more late orders at the cost of more false alarms."
        )

    with st.expander("What the model does *not* do"):
        st.markdown(
            "- It doesn't predict *how* late an order will be, only late vs on time.\n"
            "- It has no visibility into anything after purchase (strikes, weather, "
            "carrier backlogs, address failures).\n"
            "- It is trained on Brazilian orders from the Olist marketplace; the "
            "learned distance and seller effects won't transfer directly to another "
            "market or logistics network.\n"
            "- A probability is not a guarantee. Treat the output as a triage signal "
            "for which orders deserve a human look first."
        )

tab_batch, tab_single = st.tabs(["📊 Batch prediction (dataset)", "🧾 Single order prediction"])

# ---------------------------------------------------------------------------
# TAB 1: Batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown(
        "Upload a **raw, item-level** dataset (one row per order item, "
        "several rows can share the same `order_id`). Every order in the "
        "file will get one late-delivery prediction. For location, just fill "
        "in `seller_zip_code`/`customer_zip_code` — coordinates *and* state are "
        "looked up automatically. `seller_lat`/`seller_lng`/`seller_geo_state` "
        "(and the customer equivalents) only need filling in directly if you'd "
        "rather skip the zip lookup, or as a fallback for a zip the lookup "
        "doesn't recognize."
    )

    template_df = pd.DataFrame(columns=BATCH_TEMPLATE_COLUMNS)
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="order_items_template.csv",
        mime="text/csv",
    )

    batch_file = st.file_uploader("Upload dataset (CSV)", type=["csv"], key="batch_csv")

    if batch_file is not None:
        try:
            raw_df = pd.read_csv(batch_file)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            raw_df = None

        if raw_df is not None:
            missing = [c for c in CORE_REQUIRED_COLUMNS if c not in raw_df.columns]
            for prefix in ("seller", "customer"):
                has_latlng = f"{prefix}_lat" in raw_df.columns and f"{prefix}_lng" in raw_df.columns
                has_zip = f"{prefix}_zip_code" in raw_df.columns
                if not has_latlng and not has_zip:
                    missing.append(f"{prefix}_lat/{prefix}_lng (or {prefix}_zip_code)")

            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
                raw_df = None
            else:
                for col in ["order_purchase_timestamp", "order_estimated_delivery_date"]:
                    raw_df[col] = pd.to_datetime(raw_df[col])

                total_approx = 0
                all_unresolved_idx = []
                for prefix in ("seller", "customer"):
                    raw_df, n_filled, n_approx, unresolved_idx = enrich_location_batch(
                        raw_df, prefix, zip_lookup, zip_state_fallback
                    )
                    total_approx += n_approx
                    all_unresolved_idx += [(prefix, i) for i in unresolved_idx]

                if all_unresolved_idx:
                    bad_rows = ", ".join(
                        f"row {i + 2} ({prefix})" for prefix, i in all_unresolved_idx[:20]
                    )
                    more = f" and {len(all_unresolved_idx) - 20} more" if len(all_unresolved_idx) > 20 else ""
                    st.error(
                        f"❌ {len(all_unresolved_idx)} row(s) have a zip code that isn't in the "
                        f"lookup table AND no fallback state filled in — can't determine their "
                        f"state, which the model needs. Fix these rows (fill in the zip code "
                        f"correctly, or fill `_geo_state` as a fallback) and re-upload: {bad_rows}{more}"
                    )
                    raw_df = None
                elif total_approx > 0:
                    st.warning(
                        f"⚠️ {total_approx} row(s) had a zip code that wasn't found in the "
                        "lookup table — their coordinates were approximated using that row's "
                        "fallback state's average. This only affects the seller-customer "
                        "distance feature; the state itself came from your fallback column."
                    )

        if raw_df is not None:
            raw_df = raw_df[RAW_COLUMNS]

            st.write(f"Loaded **{len(raw_df)}** item rows covering "
                     f"**{raw_df['order_id'].nunique()}** orders.")
            st.dataframe(raw_df.head(20), use_container_width=True)

            if st.button("🚀 Run batch prediction", type="primary", disabled=serving_model is None):
                if serving_model is None:
                    st.warning("Upload the model + seller lookup in the sidebar first.")
                else:
                    with st.spinner("Aggregating items into orders and scoring..."):
                        try:
                            probs = serving_model.predict_proba(raw_df)[:, 1]
                            order_ids = serving_model.aggregator.transform(raw_df).index
                            results = pd.DataFrame({
                                "order_id": order_ids,
                                "late_probability": probs,
                            })
                            results["prediction"] = np.where(
                                results["late_probability"] >= late_threshold, "LATE", "ON TIME"
                            )
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            results = None

                    if results is not None:
                        n_late = (results["prediction"] == "LATE").sum()
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Orders scored", len(results))
                        c2.metric("Predicted LATE", int(n_late))
                        c3.metric("Predicted ON TIME", int(len(results) - n_late))

                        st.dataframe(
                            results.sort_values("late_probability", ascending=False),
                            use_container_width=True,
                        )
                        st.download_button(
                            "⬇️ Download predictions CSV",
                            data=results.to_csv(index=False).encode("utf-8"),
                            file_name="late_delivery_predictions.csv",
                            mime="text/csv",
                        )

# ---------------------------------------------------------------------------
# TAB 2: Single order prediction
# ---------------------------------------------------------------------------
with tab_single:
    st.markdown("Fill in the order details below. Add a row per item if the order has more than one.")

    with st.form("single_order_form"):
        st.subheader("Order-level details")
        c1, c2, c3 = st.columns(3)
        with c1:
            order_id = st.text_input("Order ID", value=f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}")
            purchase_date = st.date_input("Purchase date", value=date.today())
            purchase_time = st.time_input("Purchase time", value=dtime(12, 0))
        with c2:
            est_delivery_date = st.date_input("Estimated delivery date")
            payment_type = st.selectbox("Payment type", PAYMENT_TYPES)
            customer_state_fallback = st.selectbox(
                "Customer state (fallback)", [""] + BR_STATES,
                help="Only used if the customer zip code below isn't found in the lookup.",
            )
        with c3:
            customer_zip_code = st.text_input(
                "Customer zip code", placeholder="e.g. 01310-100",
                help="Primary source for both coordinates AND state. If not found "
                     "in the lookup, the fallback state to the left is used instead.",
            )
            payment_installments = st.number_input("Payment installments", min_value=1, value=1, step=1)

        c4, c5, c6 = st.columns(3)
        with c4:
            total_payment_value = st.number_input("Total payment value", min_value=0.0, value=0.0, format="%.2f")
        with c5:
            n_vouchers = st.number_input("Number of vouchers used", min_value=0, value=0, step=1)
        with c6:
            voucher_value = st.number_input("Total voucher value", min_value=0.0, value=0.0, format="%.2f")

        st.subheader("Item(s) in this order")
        st.caption(
            "`seller_zip_code` is the primary source for both the seller's coordinates "
            "and state. `seller_geo_state` only needs filling in as a fallback, for a "
            "zip that isn't found in the lookup."
        )
        items_df_edit = st.data_editor(
            pd.DataFrame([ITEM_COLUMN_DEFAULTS]),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "seller_geo_state": st.column_config.SelectboxColumn(
                    "seller_geo_state (fallback)", options=[""] + BR_STATES,
                ),
                "seller_zip_code": st.column_config.TextColumn(),
            },
            key="items_editor",
        )

        submitted = st.form_submit_button("🔮 Predict", type="primary")

    if submitted:
        if serving_model is None:
            st.warning("Upload the model + seller lookup in the sidebar first.")
        elif items_df_edit.empty:
            st.warning("Add at least one item.")
        else:
            purchase_ts = datetime.combine(purchase_date, purchase_time)
            est_delivery_ts = datetime.combine(est_delivery_date, dtime(0, 0))

            items = items_df_edit.copy()
            items["order_id"] = order_id
            items["order_purchase_timestamp"] = purchase_ts
            items["order_estimated_delivery_date"] = est_delivery_ts
            items["payment_type"] = payment_type
            items["payment_installments"] = payment_installments
            items["total_payment_value"] = total_payment_value
            items["n_vouchers"] = n_vouchers
            items["voucher_value"] = voucher_value

            # Resolve seller location per item from zip code (state included)
            approx_notes = []
            hard_errors = []
            seller_lats, seller_lngs, seller_states = [], [], []
            for i, r in items.iterrows():
                lat, lng, state, status = resolve_location(
                    r.get("seller_zip_code"), r.get("seller_geo_state"),
                    zip_lookup, zip_state_fallback,
                )
                if status == "unresolved":
                    hard_errors.append(
                        f"item {i + 1}: zip '{r.get('seller_zip_code') or '(blank)'}' not found "
                        f"and no fallback state given"
                    )
                    lat, lng, state = 0.0, 0.0, ""
                elif status == "state_avg":
                    approx_notes.append(f"item {i + 1} seller (zip '{r.get('seller_zip_code') or '(blank)'}')")
                seller_lats.append(lat)
                seller_lngs.append(lng)
                seller_states.append(state)
            items["seller_lat"] = seller_lats
            items["seller_lng"] = seller_lngs
            items["seller_geo_state"] = seller_states

            # Resolve customer location from zip code (single value for the order)
            cust_lat, cust_lng, cust_state, cust_status = resolve_location(
                customer_zip_code, customer_state_fallback, zip_lookup, zip_state_fallback
            )
            if cust_status == "unresolved":
                hard_errors.append(
                    f"customer: zip '{customer_zip_code or '(blank)'}' not found and no "
                    f"fallback state given"
                )
                cust_lat, cust_lng, cust_state = 0.0, 0.0, ""
            elif cust_status == "state_avg":
                approx_notes.append(f"customer (zip '{customer_zip_code or '(blank)'}')")
            items["customer_lat"] = cust_lat
            items["customer_lng"] = cust_lng
            items["customer_geo_state"] = cust_state

            if hard_errors:
                st.error(
                    "Can't determine state for: " + "; ".join(hard_errors) +
                    ". Fix the zip code, or fill in a fallback state, and predict again."
                )
                proba = None
            else:
                items = items[RAW_COLUMNS]
                try:
                    proba = serving_model.predict_proba(items)[0, 1]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    proba = None

            if proba is not None:
                is_late = proba >= late_threshold
                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Late probability", f"{proba:.1%}")
                m2.metric("Prediction", "🔴 LATE" if is_late else "🟢 ON TIME")

                if approx_notes:
                    st.caption(
                        "📍 Coordinates approximated (zip not found, used fallback state's average) for: "
                        + ", ".join(approx_notes)
                    )

                if is_late:
                    st.warning(
                        f"⚠️ Order **{order_id}** is flagged LATE "
                        f"({proba:.0%} predicted probability, threshold {late_threshold:.1%}). "
                        "Recommended action: intervene early — follow up with the seller/carrier "
                        "or notify the customer proactively."
                    )
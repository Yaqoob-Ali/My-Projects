# package_recommender_model.py

# Database credentials
username = 'root'
password = 'Aliyaqoob%40123'  # '@' is URL‐encoded as '%40'
host = '127.0.0.1'
port = 3306
database = 'aiml'

from sqlalchemy import create_engine, text
import pandas as pd
import json
from prophet import Prophet
from pandas.tseries.offsets import MonthEnd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Create the database connection
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
try:
    conn_test = engine.connect()
    conn_test.close()
    print("Database connection successful!")
except Exception as e:
    print(f"Database connection error: {e}")

# --- Data Validation & Preprocessing ---

def validate_dataframe(df, required_columns):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Empty dataframe received")
    return df

def map_and_preprocess_data(df_user):
    required = [
        'msisdn', 'creation_date', 'expire_date', 'validity',
        'Consumed_Data', 'Consumed_SMS',
        'Consumed_On-net Mins', 'Consumed_Off-net Mins',
        'Consumed_Social Data'
    ]
    validate_dataframe(df_user, required)

    mapping = {
        'msisdn': 'msisdn',
        'Consumed_Data': 'data_usage',
        'Consumed_SMS': 'sms_usage',
        'Consumed_On-net Mins': 'onnet_usage',
        'Consumed_Off-net Mins': 'offnet_usage',
        'Consumed_Social Data': 'social_usage',
        'creation_date': 'created_date',
        'expire_date': 'expiry_date',
        'validity': 'validity'
    }
    df = df_user.rename(columns=mapping)[list(mapping.values())]
    df['validity'] = pd.to_numeric(df['validity'], errors='coerce').fillna(0).astype(int)
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce', dayfirst=True)
    df['expiry_date']  = pd.to_datetime(df['expiry_date'],  errors='coerce', dayfirst=True)
    return df.loc[df['created_date'].notna() & df['expiry_date'].notna()].copy()

def process_consumption_data(df_user):
    """
    - Slice each package’s usage into calendar-month chunks proportionally.
    - Return a DataFrame with one row per (msisdn, month_range) and summed metrics.
    """
    df = map_and_preprocess_data(df_user)
    df['days_active'] = (df['expiry_date'] - df['created_date']).dt.days + 1
    df['days_active'] = df['days_active'].clip(lower=1)

    metrics = ['data_usage','sms_usage','onnet_usage','offnet_usage','social_usage']
    for m in metrics:
        df[f'daily_{m}'] = df[m] / df['days_active']

    records = []
    for _, row in df.iterrows():
        start, end = row['created_date'], row['expiry_date']
        total_days = (end - start).days + 1
        if total_days <= 0:
            continue

        if row['validity'] == 1:
            # All usage belongs to a single month
            records.append({
                'msisdn': row['msisdn'],
                'month_range': start.to_period('M'),
                **{m: row[m] for m in metrics},
                'package_count': 1
            })
        else:
            # Split across calendar-month chunks
            cur = start
            while cur <= end:
                me = (cur + MonthEnd(0)).date()
                if me > end.date():
                    me = end.date()
                days_chunk = (me - cur.date()).days + 1
                prop = days_chunk / total_days

                records.append({
                    'msisdn': row['msisdn'],
                    'month_range': cur.to_period('M'),
                    **{m: row[m] * prop for m in metrics},
                    'package_count': prop
                })
                cur = pd.Timestamp(me) + pd.Timedelta(days=1)

    df_exp = pd.DataFrame(records)
    validate_dataframe(
        df_exp,
        ['msisdn','month_range'] + metrics + ['package_count']
    )
    agg = {m: 'sum' for m in metrics}
    agg['package_count'] = 'sum'
    return df_exp.groupby(['msisdn','month_range']).agg(agg).reset_index()

# --- Forecasting ---

def forecast_metric(user_monthly_data, metric, forecast_date):
    """Forecast a single usage metric with Prophet using only the last 6 months of data."""
    # build (ds, y)
    dfp = user_monthly_data[['month_range', metric]].copy()
    dfp.columns = ['ds','y']
    dfp['ds'] = pd.to_datetime(dfp['ds'].dt.to_timestamp())

    # convert forecast_date to timestamp
    target = pd.Timestamp(forecast_date)

    # ⬅️ filter to only the last 6 months prior to the target
    # six_months_ago = target - pd.DateOffset(months=6)
    twelve_months_ago = target - pd.DateOffset(months=12)
    # dfp = dfp[dfp['ds'] >= six_months_ago]
    dfp = dfp[dfp['ds'] >= twelve_months_ago]

    # if after filtering there’s no or only one point, fall back
    if dfp['y'].dropna().shape[0] < 2:
        # return last known usage or zero
        if dfp['y'].dropna().empty:
            return 0.0
        return round(float(dfp['y'].dropna().iloc[-1]), 2)

    # fit Prophet on that 6-month window
    model = Prophet(
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=6,
        growth='linear'
    )
    model.fit(dfp)


    last   = dfp['ds'].max()
    periods = max(1, ((target - last).days // 30) + 1)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    month_start = target.replace(day=1)
    if month_start not in forecast['ds'].values:
        month_start = forecast['ds'].iloc[
            (forecast['ds'] - month_start).abs().argmin()
        ]
    yhat = float(forecast.loc[forecast['ds']==month_start, 'yhat'].iloc[0])
    return round(max(0, yhat), 2)

# --- Package Catalog Retrieval ---

def get_available_packages(validity=None):
    """
    Load package catalog. If 'validity' is provided, filter to that period.
    """
    sql = """
      SELECT
        offerid    AS package_id,
        validity,
        `On-net Mins`  AS onnet,
        `Off-net Mins` AS offnet,
        sms            AS sms,
        data           AS data,
        `Social Data`  AS social,
        price
      FROM packages
    """
    if validity:
        sql += f" WHERE validity = {validity}"
    return pd.read_sql(sql, engine)

# --- Recommendation Logic ---

def recommend_package(forecasted, trends):
    """
    1. Compute per-day needs from forecast totals & validity.
    2. Load only packages of that validity (fallback to all if empty).
    3. Compute each package’s per-day allowance.
    4. Filter to packages covering all per-day needs.
    5. Among those, choose the one minimizing max over-provision ratio.
    6. Otherwise, fallback to smallest sum of absolute diffs.
    Returns a dict with core fields + 'match_type'.
    """
    def pick_core(row, match_type):
        return {
            'package_id': row['package_id'],
            'validity':   row['validity'],
            'onnet':      row['onnet'],
            'offnet':     row['offnet'],
            'sms':        row['sms'],
            'data':       row['data'],
            'social':     row['social'],
            'price':      row['price'],
            'match_type': match_type
        }

    metrics = ('onnet','offnet','sms','data','social')

    # 1) per-day needs
    days = int(forecasted.get('validity', trends['preferred_validity']))
    per_day_need = {m: (forecasted[f'{m}_usage'] / days if days>0 else 0) for m in metrics}
    desired_validity = days

    # 2) load matching-length packages (fallback to all)
    pkgs = get_available_packages(desired_validity)
    if pkgs.empty:
        pkgs = get_available_packages()
    if pkgs.empty:
        return {'error': 'No packages available'}

    # 3) per-day allowance for each package
    for m in metrics:
        pkgs[f'per_day_{m}'] = pkgs[m] / pkgs['validity']

    # 4) determine which metrics are actually offered
    supported = [m for m in metrics if pkgs[m].max() > 0]

    # 5) filter to packages covering per-day needs
    mask = pd.Series(True, index=pkgs.index)
    for m in supported:
        need = per_day_need[m]
        col  = f'per_day_{m}'
        if need > 0:
            mask &= pkgs[col] >= need
        else:
            mask &= pkgs[col] == 0
    cand = pkgs[mask].copy()

    # 6) if any fully cover, choose by min(max_ratio)
    if not cand.empty:
        for m in supported:
            need = per_day_need[m]
            cand[f'{m}_ratio'] = (cand[f'per_day_{m}'] / need) if need>0 else 0
        cand['max_ratio'] = cand[[f'{m}_ratio' for m in supported]].max(axis=1)

        # Uncomment to cap over-provision tolerance:
        # cand = cand[cand['max_ratio'] <= 2]

        if not cand.empty:
            best = cand.sort_values(['max_ratio','price']).iloc[0]
            return pick_core(best, 'min_max_ratio')

    # 7) fallback: pick by sum of absolute per-day diffs
    diffs = pkgs.copy()
    for m in supported:
        diffs[f'{m}_diff'] = (diffs[f'per_day_{m}'] - per_day_need[m]).abs()
    diffs['score'] = diffs[[f'{m}_diff' for m in supported]].sum(axis=1)
    best = diffs.nsmallest(1, 'score').iloc[0]
    return pick_core(best, 'closest')

# --- Main Orchestrator ---

def get_package_recommendation(msisdn, forecast_date):
    """
    - Normalize MSISDN.
    - Load raw rows from 'data' table.
    - Determine preferred validity (mode of past purchases).
    - Build monthly-aggregated usage, forecast each metric, and inject validity.
    - Call 'recommend_package' to get final recommendation.
    Returns a dict with 'forecast', 'recommendation', and 'generated_at'.
    """
    clean_num = "".join(filter(str.isdigit, str(msisdn))).lstrip('0')
    if not clean_num:
        raise ValueError("Invalid MSISDN format")

    with engine.connect() as conn:
        df_raw = pd.read_sql(
            text("""
                SELECT
                  msisdn,
                  creation_date,
                  expire_date,
                  validity,
                  Consumed_Data,
                  Consumed_SMS,
                  `Consumed_On-net Mins`,
                  `Consumed_Off-net Mins`,
                  `Consumed_Social Data`
                FROM data
                WHERE msisdn = :msisdn
            """),
            conn,
            params={'msisdn': clean_num}
        )

    if df_raw.empty:
        raise ValueError("No data for this MSISDN")

    counts = df_raw['validity'].value_counts()
    trends = {'preferred_validity': int(counts.idxmax()) if not counts.empty else 30}

    monthly = process_consumption_data(df_raw)
    metrics = ['data_usage','sms_usage','onnet_usage','offnet_usage','social_usage']
    forecasted = {}

    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(forecast_metric, monthly, m, forecast_date): m for m in metrics}
        for fut in futures:
            forecasted[futures[fut]] = fut.result()

    forecasted['validity'] = trends['preferred_validity']
    recommendation = recommend_package(forecasted, trends)

    ordered_forecast = {
        'validity':      forecasted['validity'],
        'onnet_usage':   forecasted['onnet_usage'],
        'offnet_usage':  forecasted['offnet_usage'],
        'sms_usage':     forecasted['sms_usage'],
        'data_usage':    forecasted['data_usage'],
        'social_usage':  forecasted['social_usage'],
    }

    return {
        'forecast':       ordered_forecast,
        'recommendation': recommendation,
        'generated_at':   datetime.now().isoformat()
    }
# if __name__ == "__main__":
#     result = get_package_recommendation('03000000667', '2025-06-06')
#     print(json.dumps(result, indent=2))
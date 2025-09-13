"""
Data collection module for the DEX valuation study.
Fetches data from CoinGecko and DefiLlama and assembles a daily panel.

All raw JSON responses are saved to `data/raw/{source}_{protocol}_{YYYY-MM-DD}.json`.
API keys are read from environment variables (loaded via .env by the runner).
"""
import os
import json
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import yaml


RAW_DIR = os.path.join('data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

load_dotenv(override=False)


def _today_str() -> str:
    return dt.datetime.utcnow().strftime('%Y-%m-%d')


def _save_raw_json(source: str, protocol: str, payload: dict) -> None:
    fname = f"{source}_{protocol}_{_today_str()}.json"
    fpath = os.path.join(RAW_DIR, fname)
    try:
        with open(fpath, 'w') as f:
            json.dump(payload, f)
    except Exception:
        # Do not fail the pipeline on raw save issues
        pass


# ---- Protocol config helpers ----

DEFAULT_PROTOCOLS_CG = {
    # Protocol name (as in protocols.yml) -> CoinGecko coin id
    'Uniswap': 'uniswap',
    'SushiSwap': 'sushi',
    'Curve': 'curve-dao-token',
    'Balancer': 'balancer',
    'PancakeSwap': 'pancakeswap-token',
}

DEFAULT_PROTOCOLS_LLAMA = {
    # Protocol name -> DefiLlama slug
    'Uniswap': 'uniswap',
    'SushiSwap': 'sushiswap',
    'Curve': 'curve-dex',
    'Balancer': 'balancer',
    'PancakeSwap': 'pancakeswap',
}


def load_protocols(config_path: str = 'protocols.yml') -> List[Dict]:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('protocols', [])


# ---- CoinGecko ----

def fetch_coingecko_market_data(coin_id: str, start_ts: int, end_ts: int) -> Dict:
    """Fetch market caps and total volumes (USD) from CoinGecko market_chart/range.

    Returns the raw JSON with keys: market_caps, total_volumes, prices.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': start_ts,
        'to': end_ts,
    }
    headers = {}
    api_key = os.getenv('COINGECKO_API_KEY')
    if api_key:
        headers['x-cg-pro-api-key'] = api_key
    r = requests.get(url, params=params, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()
    # Fallback to market_chart with days interval
    days = max(1, int((end_ts - start_ts) / 86400))
    url2 = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params2 = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    r2 = requests.get(url2, params=params2, headers=headers, timeout=60)
    if r2.status_code == 200:
        return r2.json()
    # Try pro-api domain if available
    if api_key:
        url3 = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r3 = requests.get(url3, params=params2, headers=headers, timeout=60)
        r3.raise_for_status()
        return r3.json()
    r2.raise_for_status()


def coingecko_timeseries_to_df(raw: Dict, protocol: str) -> pd.DataFrame:
    """Convert CoinGecko market_chart payload to daily DataFrame."""
    mcs = raw.get('market_caps', [])
    vols = raw.get('total_volumes', [])
    # Convert to dict by date
    def to_map(arr):
        d = {}
        for ts_ms, val in arr:
            date = dt.datetime.utcfromtimestamp(ts_ms / 1000).date()
            d[date] = float(val) if val is not None else None
        return d
    mc_map = to_map(mcs)
    vol_map = to_map(vols)
    dates = sorted(set(mc_map.keys()) | set(vol_map.keys()))
    rows = []
    for dtt in dates:
        rows.append({
            'protocol': protocol,
            'date': pd.Timestamp(dtt),
            'market_cap_circulating': mc_map.get(dtt),
            'volume_24h': vol_map.get(dtt),
        })
    return pd.DataFrame(rows)


# ---- DefiLlama ----

def fetch_llama_protocol(slug: str) -> Dict:
    url = f"https://api.llama.fi/protocol/{slug}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def llama_tvl_to_df(raw: Dict, protocol: str) -> pd.DataFrame:
    tvl_series = raw.get('tvl', []) or raw.get('chainTvls', {}).get('tvl', [])
    rows = []
    for item in tvl_series:
        # Some series use 'date' in seconds since epoch
        ts = item.get('date') or item.get('t')
        if ts is None:
            continue
        date = pd.Timestamp(dt.datetime.utcfromtimestamp(int(ts)).date())
        tvl_val = item.get('totalLiquidityUSD') or item.get('totalLiquidityUsd') or item.get('totalLiquidity') or item.get('tvl') or item.get('value')
        if tvl_val is None:
            # Some entries might use 'u' or other; ignore if absent
            continue
        rows.append({'protocol': protocol, 'date': date, 'tvl': float(tvl_val)})
    return pd.DataFrame(rows)


def fetch_llama_fees_daily(slug: str) -> Optional[Dict]:
    """Attempt to fetch daily fees/revenue from DefiLlama fees API.
    Endpoint formats vary; this function tries a couple of known paths.
    Returns raw JSON or None if unavailable.
    """
    urls = [
        f"https://api.llama.fi/summary/fees/{slug}?dataType=daily",
        f"https://api.llama.fi/fees/{slug}",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


def llama_fees_to_df(raw: Dict, protocol: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=['protocol', 'date', 'fees_24h', 'revenue_24h'])
    rows = []
    # Compatible with summary format: {'totalDataChart': [[ts, val], ...], 'protocols': {..}} or similar
    series = raw.get('totalDataChart') or raw.get('data')
    if isinstance(series, list):
        for item in series:
            if isinstance(item, dict):
                ts = item.get('t') or item.get('date')
                fees = item.get('fees') or item.get('value') or item.get('feeRevenue')
                rev = item.get('revenue') or item.get('protocolRevenue')
                if ts is None:
                    continue
                date = pd.Timestamp(dt.datetime.utcfromtimestamp(int(ts)//1000 if int(ts) > 10**12 else int(ts)).date())
                rows.append({'protocol': protocol, 'date': date, 'fees_24h': fees, 'revenue_24h': rev})
            else:
                # Assume [ts, val]
                ts, val = item[0], item[1]
                date = pd.Timestamp(dt.datetime.utcfromtimestamp(int(ts)//1000 if int(ts) > 10**12 else int(ts)).date())
                rows.append({'protocol': protocol, 'date': date, 'fees_24h': float(val), 'revenue_24h': None})
    return pd.DataFrame(rows)


# ---- Panel assembly ----

def build_panel(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Build panel across protocols between dates (inclusive)."""
    protocols_cfg = load_protocols()
    cg_map = DEFAULT_PROTOCOLS_CG
    llama_map = DEFAULT_PROTOCOLS_LLAMA
    start_ts = int(time.mktime(dt.datetime(start_date.year, start_date.month, start_date.day).timetuple()))
    end_ts = int(time.mktime(dt.datetime(end_date.year, end_date.month, end_date.day, 23, 59).timetuple()))

    frames = []
    for p in protocols_cfg:
        name = p['name']
        cg_id = cg_map.get(name)
        llama_slug = llama_map.get(name)
        # --- CoinGecko ---
        try:
            raw_cg = fetch_coingecko_market_data(cg_id, start_ts, end_ts)
            _save_raw_json('coingecko', name, raw_cg)
            df_cg = coingecko_timeseries_to_df(raw_cg, name)
        except Exception:
            df_cg = pd.DataFrame(columns=['protocol', 'date', 'market_cap_circulating', 'volume_24h'])
        # --- DefiLlama TVL ---
        try:
            raw_llama = fetch_llama_protocol(llama_slug)
            _save_raw_json('defillama_protocol', name, raw_llama)
            df_tvl = llama_tvl_to_df(raw_llama, name)
        except Exception:
            df_tvl = pd.DataFrame(columns=['protocol', 'date', 'tvl'])
        # --- DefiLlama fees/revenue ---
        try:
            raw_fees = fetch_llama_fees_daily(llama_slug)
            if raw_fees is not None:
                _save_raw_json('defillama_fees', name, raw_fees)
            df_fees = llama_fees_to_df(raw_fees, name)
        except Exception:
            df_fees = pd.DataFrame(columns=['protocol', 'date', 'fees_24h', 'revenue_24h'])

        # Merge
        df = df_cg.merge(df_tvl, on=['protocol', 'date'], how='left')
        if not df_fees.empty:
            df = df.merge(df_fees, on=['protocol', 'date'], how='left')

        # Add placeholders for missing columns required by schema
        df['active_users_24h'] = df.get('active_users_24h', pd.Series([None]*len(df)))
        df['transactions_24h'] = df.get('transactions_24h', pd.Series([None]*len(df)))
        # Static
        chains = p.get('chains') or []
        df['token_holders'] = None
        df['governance_proposals_30d'] = None
        df['token_distribution'] = None
        df['chain_deployment'] = len(chains)
        df['token_age_days'] = None

        # Restrict to [start_date, end_date]
        df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    # Ensure columns order per template
    template_cols = [
        'protocol','date','market_cap_circulating','volume_24h','fees_24h','revenue_24h','tvl',
        'active_users_24h','transactions_24h','token_holders','governance_proposals_30d','token_distribution',
        'chain_deployment','token_age_days'
    ]
    for col in template_cols:
        if col not in panel.columns:
            panel[col] = None
    panel = panel[template_cols].sort_values(['protocol','date'])
    return panel

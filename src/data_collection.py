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
    'Maverick Protocol': 'maverick-protocol',
    'Native': 'native-utility-token',
    'Tokenlon': 'tokenlon-network-token',
    'EulerSwap': 'euler',
    '0x': '0x',
    'Ondo Finance': 'ondo-finance',
    'Hashflow': 'hashflow',
    'DODO': 'dodo',
    'TanX': None,  # No token on CoinGecko
    'Integral': 'integral',
    'ShibaSwap': 'shiba-inu',
    'Angstrom': None,  # No token yet
    'AirSwap': 'airswap',
}

DEFAULT_PROTOCOLS_LLAMA = {
    # Protocol name -> DefiLlama slug
    'Uniswap': 'uniswap',
    'SushiSwap': 'sushiswap',
    'Curve': 'curve-dex',
    'Balancer': 'balancer',
    'PancakeSwap': 'pancakeswap',
    'Maverick Protocol': 'maverick-protocol',
    'Native': 'native',
    'Tokenlon': 'tokenlon',
    'EulerSwap': 'euler-swap',
    '0x': '0x',
    'Ondo Finance': 'ondo-finance',
    'Hashflow': 'hashflow',
    'DODO': 'dodo',
    'TanX': 'tanx',
    'Integral': 'integral',
    'ShibaSwap': 'shibaswap',
    'Angstrom': None,  # Not on DefiLlama yet
    'AirSwap': 'airswap',
}

# Chain:address mapping for token price lookups via coins.llama.fi
TOKEN_ADDRESSES = {
    'Uniswap': ('ethereum', '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984'),
    'SushiSwap': ('ethereum', '0x6b3595068778dd592e39a122f4f5a5cf09c90fe2'),
    'Curve': ('ethereum', '0xD533a949740bb3306d119CC777fa900bA034cd52'),
    'Balancer': ('ethereum', '0xba100000625a3754423978a60c9317c58a424e3D'),
    'PancakeSwap': ('bsc', '0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82'),
    'Maverick Protocol': ('ethereum', '0x7448c7456a97769F6cD04F1E83A4a23cCdC46aBD'),
    'Native': ('ethereum', '0x39AA39c021dfbaE8faC545936693aC917d5E7563'),
    'Tokenlon': ('ethereum', '0x0000000000095413afc295d19edeb1ad7b71c952'),
    'EulerSwap': ('ethereum', '0xd9Fcd98c322942075A5C3860693e9f4f03AAE07b'),
    '0x': ('ethereum', '0xE41d2489571d322189246DaFA5ebDe1F4699F498'),
    'Ondo Finance': ('ethereum', '0xfAbA6f8e4a5E8Ab82F62fe7C39859FA577269BE3'),
    'Hashflow': ('ethereum', '0xb3999F658C0391d94A37f7FF328F3feC942BcADC'),
    'DODO': ('ethereum', '0x43Dfc4159D86F3A37A5A4B3D4580b888ad7d4DDd'),
    'TanX': (None, None),  # No token
    'Integral': ('ethereum', '0xD502F487e1841Fdc805130e13eae80c61186Bc98'),
    'ShibaSwap': ('ethereum', '0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE'),
    'Angstrom': (None, None),  # No token yet
    'AirSwap': ('ethereum', '0x27054b13b1B798B345b591a4d22e6562d47eA75a'),
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
    headers = {}
    api_key = os.getenv('COINGECKO_API_KEY')

    # Always use standard API domain (works with Demo and Pro keys)
    if api_key:
        headers['x-cg-demo-api-key'] = api_key  # Use demo key header

    # Use standard API endpoint
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': start_ts,
        'to': end_ts,
    }
    r = requests.get(url, params=params, headers=headers, timeout=60)
    if r.status_code == 200:
        return r.json()

    # Fallback to market_chart with days interval
    days = max(1, int((end_ts - start_ts) / 86400))
    url2 = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params2 = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
    r2 = requests.get(url2, params=params2, headers=headers, timeout=60)
    r2.raise_for_status()
    return r2.json()


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
    # Updated to use correct endpoints that work
    urls = [
        f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees",
        f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyRevenue",
        f"https://api.llama.fi/summary/fees/{slug}",
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
        # --- CoinGecko (market cap + volume) OR fallback via llama price x CG circulating supply ---
        df_cg = pd.DataFrame(columns=['protocol', 'date', 'market_cap_circulating', 'volume_24h'])
        if cg_id:  # Only try if we have a CoinGecko ID
            try:
                raw_cg = fetch_coingecko_market_data(cg_id, start_ts, end_ts)
                _save_raw_json('coingecko', name, raw_cg)
                df_cg = coingecko_timeseries_to_df(raw_cg, name)
            except Exception:
                # Fallback: compute daily market cap as price * circulating_supply (current)
                try:
                    # Get current circulating supply
                    api_key = os.getenv('COINGECKO_API_KEY')
                    headers = {}
                    if api_key:
                        headers['x-cg-demo-api-key'] = api_key
                    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
                    r = requests.get(url, headers=headers, timeout=60)
                    r.raise_for_status()
                    info = r.json()
                    _save_raw_json('coingecko_meta', name, info)
                    circ = float(info.get('market_data', {}).get('circulating_supply') or 0.0)
                    # Gather daily prices from coins.llama.fi
                    chain, addr = TOKEN_ADDRESSES.get(name, (None, None))
                    if chain and addr and circ > 0:
                        days = (end_date - start_date).days + 1
                        rows = []
                        for i in range(days):
                            d = start_date + dt.timedelta(days=i)
                            ts = int(time.mktime(dt.datetime(d.year, d.month, d.day).timetuple()))
                            urlp = f"https://coins.llama.fi/prices/historical/{ts}/{chain}:{addr}"
                            rp = requests.get(urlp, timeout=60)
                            if rp.status_code != 200:
                                continue
                            payload = rp.json()
                            price = payload.get('coins', {}).get(f"{chain}:{addr}", {}).get('price')
                            if price is None:
                                continue
                            rows.append({
                                'protocol': name,
                                'date': pd.Timestamp(d),
                                'market_cap_circulating': price * circ,
                                'volume_24h': None,
                            })
                        if rows:
                            df_cg = pd.DataFrame(rows)
                except Exception:
                    pass
        # --- DefiLlama TVL ---
        df_tvl = pd.DataFrame(columns=['protocol', 'date', 'tvl'])
        if llama_slug:  # Only try if we have a DefiLlama slug
            try:
                raw_llama = fetch_llama_protocol(llama_slug)
                _save_raw_json('defillama_protocol', name, raw_llama)
                df_tvl = llama_tvl_to_df(raw_llama, name)
            except Exception:
                pass
        # --- DefiLlama fees/revenue ---
        df_fees = pd.DataFrame(columns=['protocol', 'date', 'fees_24h', 'revenue_24h'])
        if llama_slug:  # Only try if we have a DefiLlama slug
            try:
                raw_fees = fetch_llama_fees_daily(llama_slug)
                if raw_fees is not None:
                    _save_raw_json('defillama_fees', name, raw_fees)
                    df_fees = llama_fees_to_df(raw_fees, name)
            except Exception:
                pass

        # Merge
        if not df_cg.empty and not df_tvl.empty:
            df = df_cg.merge(df_tvl, on=['protocol', 'date'], how='outer')
        elif not df_cg.empty:
            df = df_cg
        elif not df_tvl.empty:
            df = df_tvl
        else:
            df = pd.DataFrame(columns=['protocol', 'date', 'market_cap_circulating', 'volume_24h', 'tvl'])

        if not df_fees.empty and not df.empty:
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

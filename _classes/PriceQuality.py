"""
PriceQuality: checks and repairs a daily price history before it goes into PricesDaily.

    res = clean_price_history(df, ticker="XYZ.DL", start_override=None)
    res.rejected   -> True if the whole series is junk (don't load it)
    res.df         -> cleaned series (bad rows removed, bad bars repaired)
    res.summary()  -> one-line description of what was found

df: DatetimeIndex, columns Open, High, Low, Close, Volume (adjusted, as in PricesDaily),
optional CloseRaw, VolumeRaw. Only pandas and numpy are used, so both eodhd_to_pta.py and
DataIO.py can import it.

Steps, in order:
  1. start override  - drop rows before a manual start date (dbo.PriceStartOverride), for
                       ticker reuse the heuristics miss
     manual splits   - splits no vendor records (dbo.PriceManualSplits), applied only if the
                       jump is still in the data
     leading rows    - first rows 3x+ away from the level that follows are dropped
  2. invalid rows    - non-positive or missing prices
  3. shell years     - leading calendar years dominated by flat or zero-volume bars
                       (shell company / OTC history before the real listing)
  4. no-trade bars   - flat bars (O=H=L=C) with zero volume: no trade happened that day
  5. series breaks   - a gap of 45+ days where the price level changed 1.5x+, or where the
                       part before it is under a year long: history before it is dropped
                       (ticker reuse, e.g. an old company's data under a current symbol)
  6. bar repairs     - an open/high/low far outside the day's open-close body is clipped
                       (flagged Repaired = 1)
  7. bad ticks       - a temporary price move that reverses (see SPIKE_TIERS): the close is
                       interpolated (log-linear) between the good closes on either side, the bar
                       is set to O=H=L=C, CloseRaw is scaled by the same correction, and the row
                       is flagged Repaired = 1. Permanent moves are never smoothed.
  7b. scale blocks   - a 4x+ jump later undone to within 15% (a stretch on the wrong price
                       scale, e.g. a year at 10x): the block is rescaled to match both sides
  8. missed splits   - a permanent jump that exactly matches a split ratio at both the open
                       and the close, with volume changing inversely: earlier rows rescaled
  9. reject checks   - too many bad ticks, two price levels alternating, too many 3x jumps,
                       too many flat bars, or too little data left
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ---------------- thresholds ----------------
# A bad tick is a move of at least `move` that comes back to within `band` of the pre-spike close
# within `days` trading days. Real moves of 30%+ almost never fully reverse that fast.
SPIKE_TIERS = (               # (move, band, days, needs ordinary volume)
    (1.30, 1.10, 3, True),    # 30%+ move, back within 10% inside 3 days, on ordinary volume
    (2.00, 1.35, 5, False),   # 2x+ move, back within 35% inside 5 days (any volume)
)
QUIET_VOLUME_MULT = 2.0       # "ordinary volume": the spike days' volume stays within 2x the median
                              # of the prior 20 days. Real 30% reversals (2008-09 financials,
                              # biotech news) come with heavy volume; bad prints usually don't.
MAX_SPIKE_ROWS = 10           # reject if repaired bad-tick rows exceed this, or exceed
MAX_SPIKE_PER_10Y = 10        # this many per 10 years of history (2520 rows), whichever is larger
BIG_JUMP_RATIO = 3.0          # close-to-close change of 3x up or 2/3 down
MAX_BIG_JUMPS = 2             # more surviving big jumps than this -> reject
LEVEL_RETURN_BAND = 1.5       # big jump later undone by an opposite one to within 1.5x -> reject
FLAT_MAX_SHARE = 0.20         # reject if more than 20% of remaining bars are stale (flat AND
                              # unchanged from the prior close); flat bars whose price moves are
                              # close-only data, not junk
SHELL_YEAR_SHARE = 0.30       # leading years with > 30% stale or zero-volume bars are trimmed
GAP_DAYS = 45                 # calendar days with no data that count as a series break
GAP_JUMP_RATIO = 1.5          # cut before a break if the price level changed this much...
GAP_MIN_SEGMENT = 250         # ...or if the part before it has fewer rows than this
BAR_SPIKE_RATIO = 3.0         # open/high/low this far outside the open-close body is a bad tick
SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100)
SPLIT_TOLERANCE = 0.03        # jump must match a split ratio within 3%, at open and close
MIN_ROWS = 60                 # fewer rows left than this -> reject
LEAD_OUTLIER_RATIO = 3.0      # a first row this far from the median of the next 10 rows is dropped
LEAD_OUTLIER_MAX = 20         # ...up to this many leading rows
BLOCK_JUMP_RATIO = 4.0        # a jump of 4x+ later undone by an opposite jump (product within
BLOCK_RETURN_BAND = 1.15      # 15% of 1) marks a block on the wrong price scale: it is rescaled
BLOCK_MAX_ROWS = 600          # ...if the block is at most this long (about 2.5 years)
MANUAL_SPLIT_TOL = 1.15       # a manual split is applied only if the series still jumps by its ratio

PRICE_COLS = ["Open", "High", "Low", "Close"]
REPAIRED = "Repaired"         # 1 = prices on this row were rebuilt; avoid trading on them

# Tables used by the loaders and the cleanup script (created if missing)
QUALITY_DDL = [
    """IF OBJECT_ID('dbo.JunkTickers') IS NULL
    CREATE TABLE dbo.JunkTickers(
        Ticker varchar(20) NOT NULL CONSTRAINT PK_JunkTickers PRIMARY KEY,
        Reason varchar(400) NULL, RowsBefore int NULL, [TimeStamp] smalldatetime NOT NULL)""",
    """IF OBJECT_ID('dbo.PriceQualityLog') IS NULL
    CREATE TABLE dbo.PriceQualityLog(
        Id int IDENTITY(1,1) NOT NULL CONSTRAINT PK_PriceQualityLog PRIMARY KEY,
        Ticker varchar(20) NOT NULL, [Action] varchar(20) NOT NULL, Detail varchar(400) NULL,
        RowsBefore int NULL, RowsAfter int NULL, [TimeStamp] smalldatetime NOT NULL)""",
    """IF COL_LENGTH('dbo.PricesDaily', 'CloseRaw') IS NULL
        ALTER TABLE dbo.PricesDaily ADD CloseRaw numeric(18,6) NULL, VolumeRaw bigint NULL""",
    """IF COL_LENGTH('dbo.PricesDaily', 'Repaired') IS NULL
        ALTER TABLE dbo.PricesDaily ADD Repaired bit NULL""",
    """IF OBJECT_ID('dbo.PriceManualSplits') IS NULL
    CREATE TABLE dbo.PriceManualSplits(
        Ticker varchar(20) NOT NULL, SplitDate date NOT NULL, Ratio float NOT NULL,
        Note varchar(200) NULL,
        CONSTRAINT PK_PriceManualSplits PRIMARY KEY (Ticker, SplitDate))""",
    """IF OBJECT_ID('dbo.PriceStartOverride') IS NULL
    CREATE TABLE dbo.PriceStartOverride(
        Ticker varchar(20) NOT NULL CONSTRAINT PK_PriceStartOverride PRIMARY KEY,
        StartDate date NOT NULL, Note varchar(200) NULL)""",
]


@dataclass
class QualityResult:
    df: pd.DataFrame
    rejected: bool = False
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        keys = ("override_rows", "manual_splits", "invalid_rows", "lead_rows", "shell_rows",
                "no_trade_rows", "gap_rows", "bar_fixes", "spike_rows", "block_rows", "split_fixes")
        return any(self.stats.get(k) for k in keys)

    def summary(self) -> str:
        parts = list(self.reasons)
        parts += [f"{k}={v}" for k, v in self.stats.items()
                  if v and k not in ("rows_in", "rows_out", "close_only_pct")]
        parts.append(f"rows {self.stats.get('rows_in', 0)}->{self.stats.get('rows_out', 0)}")
        return "; ".join(parts)


def is_index_ticker(ticker: str) -> bool:
    """Indexes (^SPX, .INX, $...) get only the basic checks."""
    return bool(ticker) and ticker[0] in "^.$"


def _flat(d):
    return (d["Open"] == d["High"]) & (d["High"] == d["Low"]) & (d["Low"] == d["Close"])


def _stale(d):
    """
    Flat bar that also repeats the prior close: no real trade that day. A flat bar whose price
    moves is just close-only data (common before ~2000: one price per day) and is fine.
    """
    return _flat(d) & (d["Close"] == d["Close"].shift(1))


def _no_volume(d):
    return d["Volume"].fillna(0) <= 0


def _trim_shell_years(d):
    weak = (_stale(d) | _no_volume(d)).groupby(d.index.year).mean()
    cut_year = None
    for y in sorted(weak.index):
        if weak[y] > SHELL_YEAR_SHARE:
            cut_year = y
        else:
            break
    if cut_year is None or cut_year == weak.index.max():
        return d, 0              # weak all the way through: leave it for the flat-bar reject check
    keep = d.index.year > cut_year
    return d[keep], int((~keep).sum())


def _cut_at_breaks(d):
    if len(d) < 2:
        return d, 0
    days = np.diff(d.index.values).astype("timedelta64[D]").astype(int)
    close = d["Close"].to_numpy()
    cut = None
    for k in np.where(days > GAP_DAYS)[0]:          # break between row k and row k+1
        before = np.median(close[max(0, k - 4):k + 1])
        after = np.median(close[k + 1:k + 6])
        ratio = max(after / before, before / after)
        if ratio >= GAP_JUMP_RATIO or (k + 1) < GAP_MIN_SEGMENT:
            cut = k + 1                              # keep the last qualifying break
    if cut is None:
        return d, 0
    return d.iloc[cut:], int(cut)


def _repair_bars(d):
    d = d.copy()
    o, h, l, c = (d[x].to_numpy(dtype=float).copy() for x in PRICE_COLS)
    prev_c = np.r_[np.nan, c[:-1]]
    B = BAR_SPIKE_RATIO
    with np.errstate(divide="ignore", invalid="ignore"):
        far_c = (o / c > B) | (o / c < 1 / B)
        far_p = np.where(np.isnan(prev_c), True, (o / prev_c > B) | (o / prev_c < 1 / B))
    bad_open = far_c & far_p
    o[bad_open] = c[bad_open]
    hi, lo = np.maximum(o, c), np.minimum(o, c)
    bad_high = h > hi * B
    bad_low = l < lo / B
    h[bad_high] = hi[bad_high]
    l[bad_low] = lo[bad_low]
    h2, l2 = np.maximum(h, hi), np.minimum(l, lo)   # keep bars internally consistent
    changed = bad_open | bad_high | bad_low | (h2 != h) | (l2 != l)
    d["Open"], d["High"], d["Low"] = o, h2, l2
    d.loc[changed, REPAIRED] = 1
    return d, int(changed.sum())


def _quiet_volume(v, i, j):
    """True if volume on rows i..j-1 stayed ordinary (or volume is unknown)."""
    if v is None:
        return True
    prior = v[max(0, i - 20):i]
    prior = prior[np.isfinite(prior) & (prior > 0)]
    spike = v[i:j]
    spike = spike[np.isfinite(spike)]
    if len(prior) < 5 or len(spike) == 0:
        return True
    return float(np.max(spike)) <= QUIET_VOLUME_MULT * float(np.median(prior))


def _spike_end(c, i, v=None):
    """If the move into row i is a bad tick, return the first good row after it, else None."""
    r = c[i] / c[i - 1]
    for move, band, days, needs_quiet in SPIKE_TIERS:
        if r >= move or r <= 1 / move:
            for j in range(i + 1, min(i + 1 + days, len(c))):
                back = c[j] / c[i - 1]
                if 1 / band <= back <= band:
                    if not needs_quiet or _quiet_volume(v, i, j):
                        return j
                    break                      # it reverted, but on heavy volume: a real move
    return None


def _apply_manual_splits(d, manual_splits):
    """
    Splits known only to you (dbo.PriceManualSplits). Ratio = pre-split / post-split price.
    Applied only while the series still jumps by about that ratio at the date, so applying it to
    data that is already adjusted does nothing. CloseRaw/VolumeRaw are as traded: not changed.
    """
    n = 0
    for date, ratio in manual_splits or []:
        if not ratio or ratio <= 0:
            continue
        k = int(d.index.searchsorted(pd.Timestamp(date)))
        if k < 1 or k >= len(d):
            continue
        jump = d["Close"].iat[k - 1] / d["Close"].iat[k]
        if jump > 0 and abs(np.log(jump / ratio)) <= np.log(MANUAL_SPLIT_TOL):
            pre = d.index[:k]
            d.loc[pre, PRICE_COLS] = d.loc[pre, PRICE_COLS] / ratio
            d.loc[pre, "Volume"] = (d.loc[pre, "Volume"] * ratio).round()
            n += 1
    return d, n


def _trim_leading_outliers(d):
    """Drop first rows that are far from the level that follows (bad prints before a listing)."""
    n = 0
    while n < LEAD_OUTLIER_MAX and len(d) > 11:
        first, med = d["Close"].iat[0], float(np.median(d["Close"].iloc[1:11]))
        if med > 0 and (first > med * LEAD_OUTLIER_RATIO or first < med / LEAD_OUTLIER_RATIO):
            d = d.iloc[1:]
            n += 1
        else:
            break
    return d, n


def _repair_level_blocks(d):
    """
    A stretch on the wrong price scale: a jump of BLOCK_JUMP_RATIO or more, later undone by an
    opposite jump that brings the price back to within BLOCK_RETURN_BAND of where it started.
    The prices on both sides agree, so the block is rescaled to match them (Close, OHLC and
    CloseRaw by the same factor). Real moves almost never reverse that exactly at that size.
    """
    d = d.copy()
    rows = 0
    for _ in range(100):
        c = d["Close"].to_numpy(dtype=float)
        r = c[1:] / c[:-1]
        jumps = [(int(k) + 1, float(r[k])) for k in np.where(
            (r >= BLOCK_JUMP_RATIO) | (r <= 1 / BLOCK_JUMP_RATIO))[0]]
        found = False
        for (k1, r1), (k2, r2) in zip(jumps, jumps[1:]):
            if ((r1 > 1) != (r2 > 1) and k2 - k1 <= BLOCK_MAX_ROWS
                    and 1 / BLOCK_RETURN_BAND <= r1 * r2 <= BLOCK_RETURN_BAND):
                block = d.index[k1:k2]
                d.loc[block, PRICE_COLS] = d.loc[block, PRICE_COLS] / r1
                if "CloseRaw" in d:
                    d.loc[block, "CloseRaw"] = d.loc[block, "CloseRaw"] / r1
                rows += len(block)
                found = True
                break
        if not found:
            break
    return d, rows


def _repair_spikes(d):
    """Interpolate bad ticks between the good closes on either side; flag the rows Repaired."""
    d = d.copy()
    old = d["Close"].to_numpy(dtype=float)
    c = old.copy()
    n = len(c)
    fixed = np.zeros(n, dtype=bool)
    vol = d["Volume"].to_numpy(dtype=float) if "Volume" in d else None
    i = 1
    while i < n:
        j = _spike_end(c, i, vol)
        if j is None:
            i += 1
            continue
        a, b, span = np.log(c[i - 1]), np.log(c[j]), j - (i - 1)
        for k in range(i, j):
            c[k] = np.exp(a + (b - a) * (k - (i - 1)) / span)
        fixed[i:j] = True
        i = j + 1
    if fixed.any():
        rows = d.index[fixed]
        if "CloseRaw" in d:
            d.loc[rows, "CloseRaw"] = d.loc[rows, "CloseRaw"].to_numpy(dtype=float) * (c[fixed] / old[fixed])
        for col in PRICE_COLS:              # flat bar: no fake high/low for stops or limits to hit
            d.loc[rows, col] = c[fixed]
        d.loc[rows, REPAIRED] = 1
    return d, int(fixed.sum())


def _big_jumps(d):
    c = d["Close"].to_numpy(dtype=float)
    r = c[1:] / c[:-1]
    idx = np.where((r >= BIG_JUMP_RATIO) | (r <= 1 / BIG_JUMP_RATIO))[0] + 1
    return [(int(k), float(r[k - 1])) for k in idx]


def _repair_splits(d):
    """Rescale history before a jump that looks like a split the data source never applied."""
    d = d.copy()
    fixes = 0
    for k, r in _big_jumps(d) + [(k, r) for k, r in _mid_jumps(d)]:
        k_ratio = r if r >= 1 else 1 / r
        match = next((s for s in SPLIT_RATIOS if abs(k_ratio / s - 1) <= SPLIT_TOLERANCE), None)
        if match is None or k < 20 or k > len(d) - 20:
            continue
        o, c = d["Open"].to_numpy(dtype=float), d["Close"].to_numpy(dtype=float)
        r_open = o[k] / c[k - 1]
        if abs((r_open if r >= 1 else 1 / r_open) / match - 1) > SPLIT_TOLERANCE:
            continue                                # a split moves the open by the same ratio
        v = d["Volume"].to_numpy(dtype=float)
        v_before, v_after = np.median(v[k - 20:k]), np.median(v[k:k + 20])
        if v_before <= 0 or v_after <= 0:
            continue
        q = (v_after / v_before) * r                # ~1 if volume moved inversely to price
        if not (0.4 <= q <= 2.5):
            continue
        level = np.median(c[k:k + 10]) / np.median(c[k - 10:k])   # a split's new level persists
        if abs(np.log(level / r)) > np.log(1.10):
            continue
        factor = c[k] / c[k - 1]
        pos = d.index[:k]
        d.loc[pos, PRICE_COLS] = d.loc[pos, PRICE_COLS] * factor
        d.loc[pos, "Volume"] = (d.loc[pos, "Volume"] / factor).round()
        fixes += 1
    return d, fixes


def _mid_jumps(d):
    """Jumps between 1.45x and 3x (candidates for 3:2 and 2:1 splits)."""
    c = d["Close"].to_numpy(dtype=float)
    r = c[1:] / c[:-1]
    idx = np.where(((r >= 1.45) & (r < BIG_JUMP_RATIO)) |
                   ((r <= 1 / 1.45) & (r > 1 / BIG_JUMP_RATIO)))[0] + 1
    return [(int(k), float(r[k - 1])) for k in idx]


def _level_reversal(jumps):
    """A big move later undone by an opposite big move: two price scales in one series."""
    for (k1, r1), (k2, r2) in zip(jumps, jumps[1:]):
        if (r1 > 1) != (r2 > 1) and 1 / LEVEL_RETURN_BAND <= r1 * r2 <= LEVEL_RETURN_BAND:
            return True
    return False


def clean_price_history(df: pd.DataFrame, ticker: str = "", start_override=None,
                        manual_splits=None) -> QualityResult:
    """manual_splits: list of (date, ratio) from dbo.PriceManualSplits for this ticker."""
    st = dict(rows_in=0 if df is None else len(df), override_rows=0, manual_splits=0, invalid_rows=0,
              lead_rows=0, shell_rows=0, no_trade_rows=0, gap_rows=0, bar_fixes=0, spike_rows=0,
              block_rows=0, split_fixes=0, big_jumps=0, flat_pct=0, close_only_pct=0, rows_out=0)
    res = QualityResult(df=df, stats=st)
    if df is None or df.empty:
        res.rejected, res.reasons = True, ["no data"]
        return res

    d = df.sort_index()
    d = d[~d.index.duplicated(keep="last")].copy()
    # Work in float throughout: whole-number prices (e.g. 150, 175) arrive as int64, and newer
    # pandas refuses to write a repaired value like 149.99997 into an integer column
    for col in PRICE_COLS + ["Volume", "CloseRaw", "VolumeRaw"]:
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
    d[REPAIRED] = (pd.to_numeric(d[REPAIRED], errors="coerce").fillna(0).astype(int)
                   if REPAIRED in d else 0)

    if start_override is not None and not pd.isna(start_override):
        keep = d.index >= pd.Timestamp(start_override)
        st["override_rows"] = int((~keep).sum())
        d = d[keep]

    ok = d[PRICE_COLS].notna().all(axis=1) & (d[PRICE_COLS] > 0).all(axis=1)
    st["invalid_rows"] = int((~ok).sum())
    d = d[ok]
    d, st["manual_splits"] = _apply_manual_splits(d, manual_splits)

    if not is_index_ticker(ticker):
        d, st["lead_rows"] = _trim_leading_outliers(d)
        d, st["shell_rows"] = _trim_shell_years(d)
        no_trade = _flat(d) & _no_volume(d)
        st["no_trade_rows"] = int(no_trade.sum())
        d = d[~no_trade]
        d, st["gap_rows"] = _cut_at_breaks(d)
        d, st["bar_fixes"] = _repair_bars(d)
        d, st["spike_rows"] = _repair_spikes(d)
        d, st["block_rows"] = _repair_level_blocks(d)
        d, st["split_fixes"] = _repair_splits(d)

        jumps = _big_jumps(d)
        st["big_jumps"] = len(jumps)
        flat_share = float(_stale(d).mean()) if len(d) else 0.0      # stale bars only
        st["flat_pct"] = round(100 * flat_share, 1)
        st["close_only_pct"] = round(100 * float((_flat(d) & ~_stale(d)).mean()), 1) if len(d) else 0

        if st["spike_rows"] > max(MAX_SPIKE_ROWS, MAX_SPIKE_PER_10Y * len(d) / 2520):
            res.reasons.append(f"{st['spike_rows']} bad-tick rows")
        if _level_reversal(jumps):
            res.reasons.append("two price levels in one series")
        if len(jumps) > MAX_BIG_JUMPS:
            res.reasons.append(f"{len(jumps)} jumps of 3x+")
        if flat_share > FLAT_MAX_SHARE:
            res.reasons.append(f"{st['flat_pct']}% stale bars (flat, price unchanged)")

    if len(d) < MIN_ROWS:
        res.reasons.append(f"only {len(d)} usable rows")
    res.rejected = bool(res.reasons)
    st["rows_out"] = len(d)
    res.df = d
    return res

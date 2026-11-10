```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
 REVISED MANUSCRIPT — DATA PREPARATION (v2, corrected)
=============================================================================
Fixes applied vs first run:
  1. Station-name normalization — Lushui (Liuku) now retained (14 stations).
  2. Study domain = box(90, 20, 105, 26.5); output to prep_revised_v2/.
  3. Threshold variables split by physical role:
       - Rainfall vars: wet-day filtered (P ≥ 5 mm) before ROC
       - SM           : all days
       - SWRI         : wet-day filtered
       - Rx1day/Rx5day operating thresholds: annual-scale (flood-year labels)
  4. Event trigger uses Rx5day (5-day accumulation) — not Rx1day.
  5. Daily RF / LogisticRegression: dropna before fit.
  6. Fig.11 RF target = flood-year label (10 vs 35), not event-day count.
  7. Fig.5 trend diagnostic — MD5 + spatial correlation between Rx1/Rx5.
=============================================================================
"""
import os, sys, time, json, glob, re, hashlib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from scipy.stats import theilslopes, kendalltau, ks_2samp, pearsonr, spearmanr, norm
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
import rioxarray  # noqa: F401

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
_T0 = time.time()
def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    ic = {"INFO":"ℹ️","OK":"✅","WARN":"⚠️","ERR":"❌","STEP":"🔹","SEC":"📦"}.get(level,"")
    print(f"[{ts}] {ic} {msg}", flush=True)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
ROOT      = Path(r"G:\NUIST_YTU_Water")
PREP      = ROOT / "prep_revised_v2"
RAW       = ROOT / "trend" / "Raw"
INTERM    = ROOT / "trend" / "Intermediate"
CALC      = ROOT / "trend" / "Calculated_Data"
EXTRACT   = ROOT / "trend" / "Extracted_CSV"
EVENT_DIR = ROOT / "trend" / "Event_Extraction"
THR_DIR   = ROOT / "trend" / "Flood_Threshold_Analysis"
SHP_DIR   = ROOT / "output_shapefiles"

BASIN_SHP   = SHP_DIR / "custom_basin.shp"
RIVER_SHP   = SHP_DIR / "clipped_rivers.shp"
STN_CSV     = ROOT / "data" / "rainfall" / "all_15_stations.csv"
DEM_PATH    = ROOT / "data" / "AW3D30_DEM_Shweli.tif"
ETOPO_PATH  = Path(r"G:\ETOPO1_Bed_g_gmt4.nc")

CHIRPS_DAILY = RAW / "MM_chirps_1981_2025_daily_precip.nc"
RX1DAY_GRID  = RAW / "Rx1day_MM_chirps.nc"
RX5DAY_GRID  = RAW / "Rx5day_MM_chirps.nc"
SM_MERGED    = INTERM / "Soil_Moisture_Shweli_Merged.nc"

STN_DAILY        = INTERM / "Step2_Station_Daily_BasinAvg.csv"
CHIRPS_DAILY_CSV = INTERM / "Step2_CHIRPS_Daily_BasinAvg.csv"
BUF_DAILY        = INTERM / "Step2_Shweli_20kmBuffer_Daily_Avg.csv"

NCEP_DIR = RAW / "NCEP" / "Processed"

# ---- Study domain: box(90, 20, 105, 26.5) as requested ----
STUDY_LON_MIN, STUDY_LON_MAX = 90.0, 105.0
STUDY_LAT_MIN, STUDY_LAT_MAX = 20.0, 26.5
FULL_LON_MIN,  FULL_LON_MAX  = 80.0, 110.0
FULL_LAT_MIN,  FULL_LAT_MAX  =  5.0, 30.0

START_YEAR, END_YEAR = 1981, 2025
JJAS = [6, 7, 8, 9]

# ---- 14 stations — CSV uses "Lushui (Liuku)" ----
STATIONS_14 = [
    "Bhamo","Hsipaw","Katha","Kyaukme","Lashio","Moekok","Myitkyina","Shwebo",
    "Ruili","Tengchong","Baoshan","Mangshi","Lushui (Liuku)","Lincang",
]
def _norm(s):
    return (str(s).strip()
            .replace(" (", "_").replace(")", "")
            .replace(" ", "_").replace("-", "_"))
_STATIONS_14_NORM = {_norm(s) for s in STATIONS_14}

# ---- Documented flood years / dates ----
FLOOD_YEARS = [1991,1998,2004,2010,2015,2018,2019,2020,2023,2024]
FLOOD_DATES = {
    1991:"1991-07-10", 1998:"1998-07-28", 2004:"2004-07-18", 2010:"2010-09-01",
    2015:"2015-07-16", 2018:"2018-07-09", 2019:"2019-07-10", 2020:"2020-07-02",
    2023:"2023-05-14", 2023.1:"2023-08-07", 2024:"2024-06-30", 2024.1:"2024-09-09",
}
FLOOD_DAY_PRE, FLOOD_DAY_POST = 2, 5

# ---- Event detection parameters ----
WET_MM            = 5.0      # wet-day filter for ROC / SWRI
MERGE_GAP_DAYS    = 3
TERMINATION_RAIN  = 5.0
DEFAULT_RX1_MM    = 43.4
DEFAULT_RX5_MM    = 99.5
DEFAULT_SM_M3M3   = 0.323
DEFAULT_SWRI      = 5.0

# ---- Risk mosaic ----
RISK_MIN = 55.0
RISK_MAX = 100.0
LAND_MIN_Z = 5.0

# ---- Sub-directories ----
for sub in ["fig01","fig02","fig03","fig04","fig05","fig06","fig07","fig08",
            "fig09","fig10","fig11","fig12","fig13",
            "tables","trends","thresholds","events","synoptic","climate","risk"]:
    (PREP / sub).mkdir(parents=True, exist_ok=True)

MANIFEST = []
def register(path, fig, desc):
    MANIFEST.append(dict(file=str(path), figure=fig, description=desc))


# =============================================================================
# SECTION 1 — Fig.1 metadata
# =============================================================================
def prepare_fig01_metadata():
    log("="*70, "SEC"); log("SECTION 1: Fig.1 metadata", "SEC"); log("="*70, "SEC")

    basin = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = basin.total_bounds

    stn = pd.read_csv(STN_CSV)
    stn["Station"] = stn["Station"].astype(str).str.strip()
    stn["_norm"] = stn["Station"].map(_norm)

    stn_all = stn.copy()
    stn_14  = stn[stn["_norm"].isin(_STATIONS_14_NORM)].drop(columns="_norm").copy()
    excluded = [s for s in stn_all["Station"] if _norm(s) not in _STATIONS_14_NORM]

    meta = dict(
        basin_bounds=[float(minx), float(miny), float(maxx), float(maxy)],
        basin_area_km2=None,
        n_stations_original=int(len(stn_all)),
        n_stations_retained=int(len(stn_14)),
        stations_retained=stn_14["Station"].tolist(),
        stations_excluded=excluded,
    )
    try:
        meta["basin_area_km2"] = float(basin.to_crs(epsg=32647).geometry.area.sum()/1e6)
    except Exception:
        pass

    with open(PREP/"fig01"/"fig01_metadata.json","w") as f:
        json.dump(meta, f, indent=2)
    register(PREP/"fig01"/"fig01_metadata.json","Fig.1","Basin bounds, station counts")

    stn_all.drop(columns="_norm", errors="ignore").to_csv(
        PREP/"fig01"/"fig01_stations_all15.csv", index=False)
    stn_14.to_csv(PREP/"fig01"/"fig01_stations_retained14.csv", index=False)
    register(PREP/"fig01"/"fig01_stations_retained14.csv","Fig.1","14 retained stations")

    log(f"Basin bounds: {meta['basin_bounds']}", "OK")
    log(f"Stations retained: {meta['n_stations_retained']} | excluded: {excluded}", "OK")


# =============================================================================
# SECTION 2 — Master daily table
# =============================================================================
def build_master_daily():
    log("="*70, "SEC"); log("SECTION 2: Master daily table", "SEC"); log("="*70, "SEC")

    stn = pd.read_csv(STN_DAILY, parse_dates=["date"], index_col="date").sort_index()
    ch  = pd.read_csv(CHIRPS_DAILY_CSV, parse_dates=["date"], index_col="date").sort_index()
    buf = pd.read_csv(BUF_DAILY, parse_dates=["date"], index_col="date").sort_index()

    stn = stn.rename(columns={"Stn_BasinAvg_Rainfall":"P_stn","Stn_Rx5day":"Rx5_stn"})
    ch  = ch.rename(columns={"CHIRPS_BasinAvg_Rainfall":"P_chirps","CHIRPS_Rx5day":"Rx5_chirps"})
    buf = buf.rename(columns={"Shweli_20kmBuffer_Rainfall":"P_buf","Shweli_20kmBuffer_Rx5day":"Rx5_buf"})

    log("Clipping SM to basin and computing daily mean...", "INFO")
    basin = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
    ds_sm = xr.open_dataset(SM_MERGED).rio.write_crs("EPSG:4326")
    sm_mean = (ds_sm["sm"]
               .rio.clip(basin.geometry.values, basin.crs, drop=True, all_touched=True)
               .mean(dim=["lat","lon"], skipna=True))
    sm = sm_mean.to_series()
    sm.index = pd.to_datetime(sm.index).normalize()
    sm = sm.groupby(sm.index).mean().rename("SM")
    ds_sm.close()

    idx = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="D", name="date")
    df = pd.DataFrame(index=idx)
    for s in [stn, ch, buf]:
        df = df.join(s, how="left")
    df = df.join(sm, how="left")
    df["SM"] = df["SM"].interpolate(limit=5).bfill().ffill()

    df["SM_min"] = df["SM"].min(); df["SM_max"] = df["SM"].max()
    df["SM_norm"] = (df["SM"] - df["SM_min"]) / (df["SM_max"] - df["SM_min"])
    df["SWRI"] = df["P_stn"] * df["SM_norm"]

    df["year"]  = df.index.year
    df["month"] = df.index.month
    df["doy"]   = df.index.dayofyear
    df["has_stn"]    = df["P_stn"].notna().astype(int)
    df["has_chirps"] = df["P_chirps"].notna().astype(int)
    df["has_sm"]     = df["SM"].notna().astype(int)

    df.to_csv(PREP/"fig02"/"master_daily.csv")
    register(PREP/"fig02"/"master_daily.csv","Fig.2, Fig.7, Fig.9","Daily P/SM/SWRI master table")
    log(f"Master daily rows: {len(df)} | SM non-NaN: {df['SM'].notna().sum()}", "OK")

    avail = df.groupby("year").agg(
        days=("P_stn","size"),
        stn_valid=("has_stn","sum"),
        chirps_valid=("has_chirps","sum"),
        sm_valid=("has_sm","sum"),
        rain_total_stn=("P_stn","sum"),
        rain_total_chirps=("P_chirps","sum"),
    ).reset_index()
    avail["stn_complete"] = (avail["stn_valid"] >= 360).astype(int)
    avail.to_csv(PREP/"fig02"/"annual_availability.csv", index=False)
    register(PREP/"fig02"/"annual_availability.csv","Fig.2","Annual data completeness")
    log(f"Years incomplete: {avail.loc[avail['stn_complete']==0,'year'].tolist()}", "OK")
    return df


# =============================================================================
# SECTION 3 — Fig.3 climatology
# =============================================================================
def prepare_fig03_climatology():
    log("="*70, "SEC"); log("SECTION 3: Fig.3 climatology", "SEC"); log("="*70, "SEC")

    for name, fname in [("rain","Calculated_JJAS_Total_Rainfall.nc"),
                        ("temp","Calculated_JJAS_Mean_Temp.nc"),
                        ("sm",  "Calculated_JJAS_Mean_SoilMoisture.nc")]:
        src = CALC / fname
        if src.exists():
            ds = xr.open_dataset(src)
            var = list(ds.data_vars)[0]
            ds = ds.rename({var: f"JJAS_{name}"})
            # Subset to new study domain
            for c in ["latitude","longitude"]:
                if c in ds.dims: ds = ds.rename({c: c[:3]})
            ds = ds.sortby("lat").sortby("lon").sel(
                lat=slice(STUDY_LAT_MIN, STUDY_LAT_MAX),
                lon=slice(STUDY_LON_MIN, STUDY_LON_MAX))
            out = PREP/"fig03"/f"fig03_jjas_{name}.nc"
            ds.to_netcdf(out, engine="netcdf4")
            register(out,"Fig.3",f"JJAS {name} climatology")
            log(f"Saved: {out.name} | shape {dict(ds.sizes)}", "OK")
        else:
            log(f"Missing: {src}", "WARN")

    src = CALC/"Calculated_Monthly_Climatology.csv"
    if src.exists():
        m = pd.read_csv(src)
        m.to_csv(PREP/"fig03"/"fig03_monthly_climatology.csv", index=False)
        register(PREP/"fig03"/"fig03_monthly_climatology.csv","Fig.3","Monthly climatology")
        log(f"Saved: fig03_monthly_climatology.csv | cols={list(m.columns)}", "OK")


# =============================================================================
# SECTION 4 — Fig.4 annual indices + trends
# =============================================================================
def _sens_mk(years, values):
    d = pd.DataFrame({"year":years, "v":values}).dropna()
    if len(d) < 5:
        return dict(slope=np.nan, intercept=np.nan, p_mk=np.nan, tau=np.nan)
    s, i, _, _ = theilslopes(d["v"].values, d["year"].values)
    tau, p = kendalltau(d["year"].values, d["v"].values)
    return dict(slope=s*10, intercept=i, p_mk=p, tau=tau)

def prepare_fig04_annual_and_trends(master):
    log("="*70, "SEC"); log("SECTION 4: Fig.4 annual + trends", "SEC"); log("="*70, "SEC")

    df = master.copy()
    df["Rx5_stn"]    = df["P_stn"].rolling(5, min_periods=5).sum()
    df["Rx5_chirps"] = df["P_chirps"].rolling(5, min_periods=5).sum()
    df["Rx5_buf"]    = df["P_buf"].rolling(5, min_periods=5).sum()

    wet = df["P_stn"][df["P_stn"] >= 1.0]
    p95, p99 = wet.quantile(0.95), wet.quantile(0.99)
    df["R95p_stn"] = (df["P_stn"] > p95).astype(int)
    df["R99p_stn"] = (df["P_stn"] > p99).astype(int)
    df["SDII_stn"] = df["P_stn"].where(df["P_stn"] >= 1.0)

    annual = df.groupby("year").agg(
        Rx1_stn=("P_stn","max"), Rx1_chirps=("P_chirps","max"),
        Rx5_stn=("Rx5_stn","max"), Rx5_chirps=("Rx5_chirps","max"),
        Rx5_buf=("Rx5_buf","max"),
        R95p_stn=("R95p_stn","sum"), R99p_stn=("R99p_stn","sum"),
        SDII_stn=("SDII_stn","mean"),
        SM_max=("SM","max"), SM_mean=("SM","mean"),
        SM_JJAS_mean=("SM", lambda s: s[s.index.month.isin(JJAS)].mean()),
        SWRI_max=("SWRI","max"),
        SWRI_JJAS_mean=("SWRI", lambda s: s[s.index.month.isin(JJAS)].mean()),
        rain_total=("P_stn","sum"),
        rain_JJAS=("P_stn", lambda s: s[s.index.month.isin(JJAS)].sum()),
    ).reset_index()
    annual["is_flood_year"] = annual["year"].isin(FLOOD_YEARS).astype(int)

    trend_vars = [
        ("Rx1_stn","mm/dec"),("Rx1_chirps","mm/dec"),
        ("Rx5_stn","mm/dec"),("Rx5_chirps","mm/dec"),("Rx5_buf","mm/dec"),
        ("R95p_stn","days/dec"),("R99p_stn","days/dec"),
        ("SDII_stn","mm/dec"),
        ("SM_max","m3m3/dec"),("SM_JJAS_mean","m3m3/dec"),
        ("SWRI_max","index/dec"),("SWRI_JJAS_mean","index/dec"),
        ("rain_total","mm/dec"),("rain_JJAS","mm/dec"),
    ]
    rows = []
    for v, unit in trend_vars:
        r = _sens_mk(annual["year"].values, annual[v].values)
        r.update(variable=v, unit=unit)
        rows.append(r)
    pd.DataFrame(rows).to_csv(PREP/"tables"/"table07_trends_annual.csv", index=False)
    register(PREP/"tables"/"table07_trends_annual.csv","Fig.4, Table 7","Annual trend stats")

    ks_rows = []
    for v,_ in trend_vars:
        e = annual.loc[annual["year"]<=2000, v].dropna()
        l = annual.loc[annual["year"]>=2001, v].dropna()
        if len(e)>=5 and len(l)>=5:
            stat, p = ks_2samp(e, l)
            ks_rows.append(dict(variable=v, ks_stat=stat, ks_p=p,
                                early_mean=e.mean(), late_mean=l.mean(),
                                delta_mean=l.mean()-e.mean()))
    pd.DataFrame(ks_rows).to_csv(PREP/"tables"/"table07_ks_early_late.csv", index=False)
    register(PREP/"tables"/"table07_ks_early_late.csv","Fig.4","KS early-late")

    annual.to_csv(PREP/"fig04"/"fig04_annual_indices.csv", index=False)
    register(PREP/"fig04"/"fig04_annual_indices.csv","Fig.4","Annual indices + labels")
    log(f"Annual table: {len(annual)} years, {len(annual.columns)} cols", "OK")

    df["flood_day"] = 0
    for d in FLOOD_DATES.values():
        d0 = pd.to_datetime(d) - pd.Timedelta(days=FLOOD_DAY_PRE)
        d1 = pd.to_datetime(d) + pd.Timedelta(days=FLOOD_DAY_POST)
        df.loc[(df.index>=d0)&(df.index<=d1), "flood_day"] = 1

    keep = ["year","month","P_stn","P_chirps","P_buf","SM","SM_norm","SWRI",
            "Rx5_stn","Rx5_chirps","Rx5_buf","flood_day"]
    df[keep].to_csv(PREP/"fig12"/"fig12_daily_labelled.csv")
    register(PREP/"fig12"/"fig12_daily_labelled.csv","Fig.12","Daily labelled table")
    log(f"Flood-labelled days: {int(df['flood_day'].sum())}", "OK")
    return annual, df


# =============================================================================
# SECTION 5 — Fig.5 spatial trends + diagnostic
# =============================================================================
def _theil_sen_map(da):
    t = da.time.values.astype("datetime64[Y]").astype(int) + 1970
    n = da.shape[0]
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            dt = t[j] - t[i]
            if dt <= 0: continue
            s = (da.isel(time=j).values - da.isel(time=i).values) / dt
            slopes.append(s)
    if not slopes:
        return np.full(da.shape[1:], np.nan)
    return np.nanmedian(np.stack(slopes, axis=0), axis=0) * 10

def _mk_pvalue_map(da):
    v = da.values
    n = v.shape[0]
    S = np.zeros(v.shape[1:])
    for i in range(n-1):
        S += np.sign(v[i+1:] - v[i]).sum(axis=0)
    varS = n*(n-1)*(2*n+5)/18.0
    z = np.where(S>0, (S-1)/np.sqrt(varS),
         np.where(S<0, (S+1)/np.sqrt(varS), 0.0))
    return 2*(1 - norm.cdf(np.abs(z)))

def _md5_head(path, nbytes=200_000):
    with open(path, "rb") as f:
        return hashlib.md5(f.read(nbytes)).hexdigest()[:10]

def prepare_fig05_spatial_trends():
    log("="*70, "SEC"); log("SECTION 5: Fig.5 spatial trends", "SEC"); log("="*70, "SEC")

    files = [("Rx1day", RX1DAY_GRID), ("Rx5day", RX5DAY_GRID)]
    for tag, path in files:
        if path.exists():
            log(f"  {tag} source md5-head: {_md5_head(path)}  ({path.name})", "INFO")

    computed = {}
    for tag, path in files:
        if not path.exists():
            log(f"Missing {path}", "WARN"); continue
        ds = xr.open_dataset(path)
        for c in ["latitude","longitude"]:
            if c in ds.dims: ds = ds.rename({c: c[:3]})
        var = list(ds.data_vars)[0]
        da = ds[var].sortby("lat").sortby("lon").sel(
            lat=slice(STUDY_LAT_MIN, STUDY_LAT_MAX),
            lon=slice(STUDY_LON_MIN, STUDY_LON_MAX))
        log(f"{tag}: grid {da.shape}, var='{var}'", "OK")

        slope = _theil_sen_map(da)
        pval  = _mk_pvalue_map(da)
        computed[tag] = dict(slope=slope, pval=pval,
                             lat=da.lat.values, lon=da.lon.values)

        out = xr.Dataset(
            {f"{tag}_slope_mm_per_decade": (("lat","lon"), slope.astype("float32")),
             f"{tag}_mk_pvalue":          (("lat","lon"), pval.astype("float32"))},
            coords={"lat": da.lat.values, "lon": da.lon.values})
        fp = PREP/"fig05"/f"fig05_{tag.lower()}_trend_grid.nc"
        out.to_netcdf(fp, engine="netcdf4")
        register(fp,"Fig.5",f"{tag} trend grid")
        log(f"Saved: {fp.name} | slope {np.nanmin(slope):.2f}–{np.nanmax(slope):.2f}", "OK")
        ds.close()

    if "Rx1day" in computed and "Rx5day" in computed:
        a = computed["Rx1day"]["slope"].ravel()
        b = computed["Rx5day"]["slope"].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            r = float(np.corrcoef(a[ok], b[ok])[0,1])
            log(f"  Correlation of Rx1 vs Rx5 trend fields: r = {r:.3f}", "OK")
            if r > 0.999:
                log("  ⚠️ Rx1/Rx5 trends nearly identical — check source NetCDFs!", "WARN")


# =============================================================================
# SECTION 6 — Fig.6 ML thresholds for all variables
# =============================================================================
def _roc_table(y_true, scores, name):
    fpr, tpr, thr = roc_curve(y_true, scores, drop_intermediate=False)
    a = auc(fpr, tpr)
    j = tpr - fpr
    k = int(np.argmax(j))
    return dict(
        variable=name, AUC=float(a), youden_J=float(j[k]),
        threshold_youden=float(thr[k]),
        POD_youden=float(tpr[k]), FAR_youden=float(fpr[k]),
        fpr=fpr, tpr=tpr, thr=thr, J=j)

def prepare_fig06_thresholds(annual, daily):
    log("="*70, "SEC"); log("SECTION 6: Fig.6 ML thresholds", "SEC"); log("="*70, "SEC")

    # ---------- (A) ANNUAL scale ----------
    cand_annual = {
        "Rx1day_Stn":        "Rx1_stn",
        "Rx1day_CHIRPS":     "Rx1_chirps",
        "Rx5day_Stn":        "Rx5_stn",
        "Rx5day_CHIRPS":     "Rx5_chirps",
        "Rx5day_20kmBuffer": "Rx5_buf",
        "R95p_Stn":          "R95p_stn",
        "R99p_Stn":          "R99p_stn",
        "SM_annual_max":     "SM_max",
        "SM_JJAS_mean":      "SM_JJAS_mean",
        "SWRI_annual_max":   "SWRI_max",
        "SWRI_JJAS_mean":    "SWRI_JJAS_mean",
    }
    y_annual = annual["is_flood_year"].values
    annual_rows, roc_dump = [], []
    for label, col in cand_annual.items():
        if col not in annual: continue
        x = annual[col].values
        ok = np.isfinite(x)
        if ok.sum() < 10: continue
        r = _roc_table(y_annual[ok], x[ok], label)
        annual_rows.append({k:v for k,v in r.items() if not isinstance(v, np.ndarray)})
        for f_i,t_i,thr_i in zip(r["fpr"], r["tpr"], r["thr"]):
            roc_dump.append(dict(variable=label, fpr=f_i, tpr=t_i, threshold=thr_i))

    annual_thr = pd.DataFrame(annual_rows).sort_values("AUC", ascending=False)
    annual_thr.to_csv(PREP/"tables"/"table04_thresholds_annual_scale.csv", index=False)
    pd.DataFrame(roc_dump).to_csv(PREP/"thresholds"/"fig06_roc_annual.csv", index=False)
    register(PREP/"tables"/"table04_thresholds_annual_scale.csv","Fig.6, Table 4","ML thresholds — annual")
    register(PREP/"thresholds"/"fig06_roc_annual.csv","Fig.6","ROC data — annual")
    log(f"Annual thresholds: {len(annual_thr)} variables", "OK")
    for _, r in annual_thr.iterrows():
        log(f"  {r['variable']:<20s} AUC={r['AUC']:.3f} thr={r['threshold_youden']:.2f}", "OK")

    # ---------- (B) DAILY scale — wet-day filtered rainfall, all-day SM ----------
    log(f"Daily ROC uses wet-day filter P≥{WET_MM:.0f} mm for rainfall vars", "INFO")
    wet_mask = daily["P_stn"] >= WET_MM
    daily_rows = []

    rainfall_vars = [("Rx1day_Stn","P_stn"),
                     ("Rx5day_Stn","Rx5_stn"),
                     ("Rx5day_CHIRPS","Rx5_chirps"),
                     ("SWRI","SWRI")]
    for label, col in rainfall_vars:
        sub = daily.loc[wet_mask, [col,"flood_day"]].dropna()
        if sub["flood_day"].sum() < 10:
            log(f"{label}: <10 positive wet-day samples — skipping", "WARN"); continue
        r = _roc_table(sub["flood_day"].values, sub[col].values, label)
        daily_rows.append({k:v for k,v in r.items() if not isinstance(v, np.ndarray)})

    sub = daily[["SM","flood_day"]].dropna()
    if sub["flood_day"].sum() >= 10:
        r = _roc_table(sub["flood_day"].values, sub["SM"].values, "SM")
        daily_rows.append({k:v for k,v in r.items() if not isinstance(v, np.ndarray)})

    daily_thr = pd.DataFrame(daily_rows).sort_values("AUC", ascending=False)
    daily_thr.to_csv(PREP/"tables"/"table04_thresholds_daily_scale.csv", index=False)
    register(PREP/"tables"/"table04_thresholds_daily_scale.csv","Fig.6, Table 4","ML thresholds — daily")
    for _, r in daily_thr.iterrows():
        log(f"  {r['variable']:<16s} AUC={r['AUC']:.3f} thr={r['threshold_youden']:.3f} "
            f"POD={r['POD_youden']:.2f} FAR={r['FAR_youden']:.2f}", "OK")

    # ---------- (C) RF importance on annual features ----------
    feat_cols = [c for c in ["Rx1_stn","Rx5_stn","SM_max","SWRI_max"] if c in annual]
    X = annual[feat_cols].fillna(0).values
    y = annual["is_flood_year"].values
    if len(np.unique(y)) > 1:
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(X, y)
        imp = pd.DataFrame({"feature":feat_cols,
                            "importance":rf.feature_importances_}
                           ).sort_values("importance", ascending=False)
        imp.to_csv(PREP/"thresholds"/"fig06_rf_importance_annual.csv", index=False)
        register(PREP/"thresholds"/"fig06_rf_importance_annual.csv","Fig.6","RF importance")
        log(f"RF importance: {dict(zip(imp.feature, imp.importance.round(3)))}", "OK")

    # ---------- (D) Final operating thresholds ----------
    def _a(label, fb):
        row = annual_thr[annual_thr["variable"]==label]
        return float(row["threshold_youden"].iloc[0]) if not row.empty else fb
    def _d(label, fb):
        row = daily_thr[daily_thr["variable"]==label]
        return float(row["threshold_youden"].iloc[0]) if not row.empty else fb

    final = {
        "Rx1day_Stn": _a("Rx1day_Stn", DEFAULT_RX1_MM),
        "Rx5day_Stn": _a("Rx5day_Stn", DEFAULT_RX5_MM),
        "SM":         _d("SM",         DEFAULT_SM_M3M3),
        "SWRI":       _d("SWRI",       DEFAULT_SWRI),
    }
    with open(PREP/"thresholds"/"final_operational_thresholds.json","w") as f:
        json.dump(final, f, indent=2)
    register(PREP/"thresholds"/"final_operational_thresholds.json","Fig.6, Fig.7","Final ML thresholds")
    log(f"Final operating thresholds: {final}", "OK")
    return annual_thr, daily_thr, final


# =============================================================================
# SECTION 7 — Fig.7–8 event catalog (trigger on Rx5day, not Rx1day)
# =============================================================================
def _categorize(dur):
    if dur <= 2:  return "Flash"
    if dur <= 5:  return "Short"
    return "Prolonged"

def prepare_events(daily, final_thr):
    log("="*70, "SEC"); log("SECTION 7: Fig.7–8 event catalog", "SEC"); log("="*70, "SEC")

    T_RX5  = final_thr["Rx5day_Stn"]
    T_SM   = final_thr["SM"]
    T_SWRI = final_thr["SWRI"]
    log(f"Trigger: Rx5≥{T_RX5:.1f} mm & SM≥{T_SM:.3f} & SWRI≥{T_SWRI:.3f}", "OK")

    df = daily.copy()
    trigger = ((df["Rx5_stn"] >= T_RX5) &
               (df["SM"]      >= T_SM)  &
               (df["SWRI"]    >= T_SWRI))
    log(f"Trigger days: {int(trigger.sum())}", "OK")

    windows, start = [], None
    for d in df.index:
        if start is None and trigger.loc[d]:
            start = d
        elif start is not None:
            r3 = df.loc[d-pd.Timedelta(days=2):d, "P_stn"]
            if (df.loc[d,"SWRI"] < T_SWRI) and (len(r3)==3) and (r3.max() < TERMINATION_RAIN):
                windows.append((start, d-pd.Timedelta(days=1)))
                start = None
    if start is not None:
        windows.append((start, df.index[-1]))

    windows = sorted(windows); merged = []
    for s,e in windows:
        if merged and (s-merged[-1][1]).days <= MERGE_GAP_DAYS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s,e))
    log(f"Raw windows: {len(windows)} → merged: {len(merged)}", "OK")

    rows = []
    for s,e in merged:
        w = df.loc[s:e]; dur = (e-s).days + 1
        sm_onset = df.loc[:s,"SM"].iloc[-2] if (df.index < s).any() else np.nan
        rows.append(dict(
            start_date=s, end_date=e, duration_days=dur,
            category=_categorize(dur),
            peak_SWRI=float(w["SWRI"].max()),
            rain_sum_mm=float(w["P_stn"].sum()),
            rain_max_mm=float(w["P_stn"].max()),
            Rx5_max=float(w["Rx5_stn"].max()),
            SM_onset=float(sm_onset) if pd.notna(sm_onset) else np.nan,
            SM_peak=float(w["SM"].max()),
            Rx5_hits=int((w["Rx5_stn"]>=T_RX5).sum()),
            days_trigger=int(trigger.loc[s:e].sum()),
        ))
    ev = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    ev["event_id"] = np.arange(1, len(ev)+1)
    ev["is_fast"]  = ev["category"].isin(["Flash","Short"]).astype(int)

    tol = pd.Timedelta(days=3)
    def _match(s,e):
        hits = []
        for _, d in FLOOD_DATES.items():
            dd = pd.to_datetime(d)
            if (s-tol) <= dd <= (e+tol):
                hits.append(f"{dd.date()}")
        return "; ".join(hits)
    ev["matched_news_dates"] = ev.apply(lambda r: _match(r["start_date"], r["end_date"]), axis=1)
    ev["matched"] = (ev["matched_news_dates"]!="").astype(int)

    ev.to_csv(PREP/"events"/"event_catalog_full.csv", index=False)
    register(PREP/"events"/"event_catalog_full.csv","Fig.8, Table 5","Full event catalog")
    log(f"Events: {len(ev)} | matched: {ev['matched'].sum()}", "OK")

    cat = ev.groupby("category").agg(
        n=("event_id","size"),
        mean_duration=("duration_days","mean"),
        median_duration=("duration_days","median"),
        mean_rain_sum=("rain_sum_mm","mean"),
        mean_peak_SWRI=("peak_SWRI","mean"),
        mean_SM_onset=("SM_onset","mean"),
        matched=("matched","sum"),
    ).reset_index()
    cat["fraction"] = cat["n"]/len(ev)
    cat.to_csv(PREP/"tables"/"table06_category_summary.csv", index=False)
    register(PREP/"tables"/"table06_category_summary.csv","Fig.8, Table 6","Category summary")
    log(f"Categories: {dict(zip(cat.category, cat.n))}", "OK")

    ann = ev.copy(); ann["year"] = ann["start_date"].dt.year
    yearly = ann.groupby(["year","category"]).size().unstack(fill_value=0)
    for c in ["Flash","Short","Prolonged"]:
        if c not in yearly: yearly[c] = 0
    yearly["total"] = yearly.sum(axis=1)
    yearly["fast"]  = yearly["Flash"] + yearly["Short"]
    yearly.reset_index().to_csv(PREP/"fig08"/"fig08_yearly_counts.csv", index=False)
    register(PREP/"fig08"/"fig08_yearly_counts.csv","Fig.8","Yearly event counts")

    trig_out = df[["P_stn","Rx5_stn","SM","SWRI"]].copy()
    trig_out["trigger"] = trigger.astype(int)
    trig_out["T_RX5"] = T_RX5; trig_out["T_SM"]=T_SM; trig_out["T_SWRI"]=T_SWRI
    trig_out.to_csv(PREP/"fig07"/"fig07_trigger_daily.csv")
    register(PREP/"fig07"/"fig07_trigger_daily.csv","Fig.7","Daily trigger series")
    log(f"Trigger daily saved.", "OK")
    return ev, cat, trigger


# =============================================================================
# SECTION 8 — Fig.9 category composites
# =============================================================================
def prepare_fig09_composites(daily, ev):
    log("="*70, "SEC"); log("SECTION 8: Fig.9 category composites", "SEC"); log("="*70, "SEC")
    PRE, POST = 10, 5
    records = []
    for _, r in ev.iterrows():
        s, e = r["start_date"], r["end_date"]
        w = daily.loc[s-pd.Timedelta(days=PRE): e+pd.Timedelta(days=POST)]
        if len(w)==0: continue
        rel = (w.index - s).days
        for d, rel_i in zip(w.index, rel):
            records.append(dict(event_id=r["event_id"], category=r["category"],
                                rel_day=int(rel_i),
                                P=w.loc[d,"P_stn"], SM=w.loc[d,"SM"], SWRI=w.loc[d,"SWRI"]))
    comp = pd.DataFrame(records)
    if comp.empty:
        log("No composite records — skipping", "WARN"); return

    grp = comp.groupby(["category","rel_day"]).agg(
        P_mean=("P","mean"), P_p90=("P", lambda s: s.quantile(0.9)),
        SM_mean=("SM","mean"), SM_p10=("SM", lambda s: s.quantile(0.1)),
        SWRI_mean=("SWRI","mean"), n=("P","size"),
    ).reset_index()
    grp.to_csv(PREP/"fig09"/"fig09_category_composites.csv", index=False)
    register(PREP/"fig09"/"fig09_category_composites.csv","Fig.9","Category-day composites")

    comp["group"] = comp["category"].map({"Flash":"Fast","Short":"Fast","Prolonged":"Prolonged"})
    grp2 = comp.groupby(["group","rel_day"]).agg(
        P_mean=("P","mean"), SM_mean=("SM","mean"),
        SWRI_mean=("SWRI","mean"), n=("P","size"),
    ).reset_index()
    grp2.to_csv(PREP/"fig09"/"fig09_fast_vs_prolonged.csv", index=False)
    register(PREP/"fig09"/"fig09_fast_vs_prolonged.csv","Fig.9","Fast vs Prolonged")
    log(f"Composites saved: {len(grp)} category-rows", "OK")


# =============================================================================
# SECTION 9 — Fig.10 synoptic (verify)
# =============================================================================
def prepare_fig10_synoptic():
    log("="*70, "SEC"); log("SECTION 9: Fig.10 synoptic verify", "SEC"); log("="*70, "SEC")
    expected = ["MM_pr_wtr_composite_event.nc","MM_pr_wtr_anomaly_event.nc",
                "MM_hgt_500_composite_event.nc","MM_hgt_500_anomaly_event.nc",
                "MM_uwnd_850_composite_event.nc","MM_vwnd_850_composite_event.nc",
                "MM_pr_wtr_daily.nc"]
    for f in expected:
        p = NCEP_DIR / f
        if p.exists():
            register(p,"Fig.10",f"NCEP {f}")
            log(f"Verified: {f}", "OK")
        else:
            log(f"Missing: {p}", "WARN")


# =============================================================================
# SECTION 10 — Fig.11 climate linkage (RF target = flood-year)
# =============================================================================
def prepare_fig11_climate(annual, ev):
    log("="*70, "SEC"); log("SECTION 10: Fig.11 climate linkage", "SEC"); log("="*70, "SEC")

    ev_year = ev.copy(); ev_year["year"] = ev_year["start_date"].dt.year
    yearly = ev_year.groupby("year").size().rename("event_days").reset_index()

    def _load_idx(fname, months):
        p = RAW / fname
        if not p.exists(): return None
        d = pd.read_csv(p, sep=r"\s+", engine="python")
        d.columns = [c.upper() for c in d.columns]
        d["YEAR"] = d["YEAR"].astype(int)
        d = d[(d["YEAR"]>=START_YEAR)&(d["YEAR"]<=END_YEAR)]
        d["SEASONAL"] = d[months].mean(axis=1)
        return d[["YEAR","SEASONAL"]]

    specs = [("ONI_2D.txt","ONI",["APR","MAY"]),
             ("DMI_2D.txt","DMI",["APR","MAY"]),
             ("WYI_2D.txt","WYI",["JUN","JUL","AUG","SEP"]),
             ("SAMI_2D.txt","SAMI",["JUN","JUL","AUG","SEP"])]

    idx = None
    for f,lbl,m in specs:
        d = _load_idx(f,m)
        if d is None:
            log(f"Missing {f}", "WARN"); continue
        d = d.rename(columns={"SEASONAL":lbl})
        idx = d if idx is None else idx.merge(d, on="YEAR", how="outer")
    if idx is None:
        log("No climate indices available.", "WARN"); return

    merged = yearly.merge(idx, left_on="year", right_on="YEAR", how="left")
    merged["is_flood_year"] = merged["year"].isin(FLOOD_YEARS).astype(int)
    merged.to_csv(PREP/"fig11"/"fig11_index_annual.csv", index=False)
    register(PREP/"fig11"/"fig11_index_annual.csv","Fig.11","Annual indices vs events")

    corr_rows = []
    for lbl in ["ONI","DMI","WYI","SAMI"]:
        if lbl not in merged: continue
        x = merged[lbl]; y = merged["event_days"]
        ok = x.notna() & y.notna()
        if ok.sum() < 5: continue
        r, p = pearsonr(x[ok], y[ok])
        rho, ps = spearmanr(x[ok], y[ok])
        corr_rows.append(dict(index=lbl, n=int(ok.sum()),
                              pearson_r=r, pearson_p=p,
                              spearman_rho=rho, spearman_p=ps))
    pd.DataFrame(corr_rows).to_csv(PREP/"tables"/"table08_climate_correlations.csv", index=False)
    register(PREP/"tables"/"table08_climate_correlations.csv","Fig.11, Table 8","Climate correlations")

    # RF importance — target = flood-year label (10 vs 35)
    feats = [c for c in ["ONI","DMI","WYI","SAMI"] if c in merged]
    X = merged[feats].fillna(0).values
    y = merged["is_flood_year"].values
    if len(np.unique(y)) > 1:
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(X, y)
        imp = pd.DataFrame({"feature":feats,
                            "importance":rf.feature_importances_}
                           ).sort_values("importance", ascending=False)
        imp.to_csv(PREP/"fig11"/"fig11_rf_importance.csv", index=False)
        register(PREP/"fig11"/"fig11_rf_importance.csv","Fig.11","RF importance — climate")
        log(f"RF importance (flood-year target): {dict(zip(imp.feature, imp.importance.round(3)))}", "OK")
    else:
        log("Only one class present — skipping RF importance", "WARN")


# =============================================================================
# SECTION 11 — Fig.12 operational curve + daily RF (dropna before fit)
# =============================================================================
def prepare_fig12_operational(daily, final_thr):
    log("="*70, "SEC"); log("SECTION 11: Fig.12 operational + RF", "SEC"); log("="*70, "SEC")

    # ---- Binned P(Flood | Rx5day) ----
    bins = np.arange(0, 250, 15)
    df = daily.copy()
    df["bin"] = pd.cut(df["Rx5_stn"], bins)
    b = df.groupby("bin", observed=True).agg(
        n=("flood_day","size"), nf=("flood_day","sum")
    ).reset_index()
    b["mid"] = b["bin"].apply(lambda x: x.mid)
    b["p"]   = b["nf"] / b["n"]
    b.to_csv(PREP/"fig12"/"fig12_binned_prob.csv", index=False)
    register(PREP/"fig12"/"fig12_binned_prob.csv","Fig.12","Binned P(Flood|Rx5day)")
    log(f"Binned probability: {len(b)} bins", "OK")

    # ---- Logistic fit (dropna) ----
    sub = daily[["Rx5_stn","flood_day"]].dropna()
    X = sub["Rx5_stn"].values.reshape(-1,1)
    y = sub["flood_day"].values
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
    a, c = lr.coef_[0][0], lr.intercept_[0]
    xg = np.linspace(0, 250, 300)
    pg = lr.predict_proba(xg.reshape(-1,1))[:,1]
    pd.DataFrame({"x_mm":xg,"p":pg}).to_csv(
        PREP/"fig12"/"fig12_logistic_curve.csv", index=False)
    register(PREP/"fig12"/"fig12_logistic_curve.csv","Fig.12","Logistic trigger curve")

    def _rx5_at(p):
        return float((np.log(p/(1-p)) - c) / a)
    op = {"P0_30_mm":_rx5_at(0.30), "P0_60_mm":_rx5_at(0.60),
          "T_RX1":final_thr["Rx1day_Stn"], "T_RX5":final_thr["Rx5day_Stn"],
          "T_SM":final_thr["SM"], "T_SWRI":final_thr["SWRI"]}
    with open(PREP/"fig12"/"operational_tiers.json","w") as f:
        json.dump(op, f, indent=2)
    register(PREP/"fig12"/"operational_tiers.json","Fig.12","Operational risk tiers")
    log(f"Operational tiers: {op}", "OK")

    # ---- Daily RF with SM features (dropna) ----
    d = daily.copy()
    d["Rx5_lag1"] = d["Rx5_stn"].shift(1)
    d["Rx5_lag3"] = d["Rx5_stn"].shift(3)
    d["ante5"]    = d["P_stn"].rolling(5).sum()
    d["SM_lag1"]  = d["SM"].shift(1)
    d["SM_anom"]  = d["SM"] - d["SM"].rolling(30, min_periods=15).mean()
    feats = ["Rx5_stn","Rx5_lag1","Rx5_lag3","ante5","SM","SM_lag1","SM_anom"]
    data = d.dropna(subset=feats).copy()
    X = data[feats].values
    y = data["flood_day"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr, te = next(sss.split(X, y))
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    prob = rf.predict_proba(X[te])[:,1]
    pred = (prob>=0.5).astype(int)

    fpr, tpr, _ = roc_curve(y[te], prob)
    a_rf = auc(fpr, tpr)
    tn,fp,fn,tp = confusion_matrix(y[te], pred, labels=[0,1]).ravel()
    pod = tp/(tp+fn) if (tp+fn) else 0
    far = fp/(fp+tn) if (fp+tn) else 0
    bs  = brier_score_loss(y[te], prob)

    pd.DataFrame([dict(
        model="RF_daily_with_SM",
        n_train=len(tr), n_test=len(te),
        n_pos_train=int(y[tr].sum()), n_pos_test=int(y[te].sum()),
        AUC=a_rf, POD=pod, FAR=far, brier=bs,
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
    )]).to_csv(PREP/"tables"/"table09_rf_daily_performance.csv", index=False)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(
        PREP/"fig12"/"fig12_rf_roc.csv", index=False)
    pd.DataFrame({"feature":feats,"importance":rf.feature_importances_}).to_csv(
        PREP/"fig12"/"fig12_rf_importance_daily.csv", index=False)
    register(PREP/"tables"/"table09_rf_daily_performance.csv","Fig.12, Table 9","RF daily perf")
    register(PREP/"fig12"/"fig12_rf_roc.csv","Fig.12","RF ROC")
    log(f"RF daily: AUC={a_rf:.3f} POD={pod:.3f} FAR={far:.3f} Brier={bs:.4f}", "OK")


# =============================================================================
# SECTION 12 — Fig.13 risk mosaic (box 90-105, 20-26.5)
# =============================================================================
def prepare_fig13_risk(final_thr):
    log("="*70, "SEC"); log("SECTION 12: Fig.13 risk mosaic", "SEC"); log("="*70, "SEC")

    if not CHIRPS_DAILY.exists():
        log(f"Missing CHIRPS: {CHIRPS_DAILY}", "ERR"); return

    ds = xr.open_dataset(CHIRPS_DAILY)
    for c in ["latitude","longitude"]:
        if c in ds.dims: ds = ds.rename({c: c[:3]})
    years = pd.to_datetime(ds.time.values).year
    ds = ds.isel(time=(years>=START_YEAR)&(years<=END_YEAR))
    ds = ds.sortby("lat").sortby("lon").sel(
        lat=slice(STUDY_LAT_MIN, STUDY_LAT_MAX),
        lon=slice(STUDY_LON_MIN, STUDY_LON_MAX))
    var = "precip" if "precip" in ds else list(ds.data_vars)[0]
    P = ds[var]
    lat = P.lat.values; lon = P.lon.values
    log(f"CHIRPS grid for risk: {P.shape}", "OK")

    T_RX5  = final_thr.get("Rx5day_Stn", 99.5)
    T_RAIN = final_thr.get("Rx1day_Stn", 43.4)
    log(f"Risk thresholds: Rx5≥{T_RX5:.1f} mm, Rain≥{T_RAIN:.1f} mm", "OK")

    rx5 = P.rolling(time=5, min_periods=5).sum()
    n_trig = (rx5 >= T_RX5).sum(dim="time")
    n_ext  = (P   >= T_RAIN).sum(dim="time")
    sens = (n_trig / n_ext * 100).where(n_ext >= 5).clip(0, 100).values
    ds.close()

    log("Loading ETOPO1...", "INFO")
    dd = xr.open_dataset(ETOPO_PATH)
    if "x" in dd.coords: dd = dd.rename({"x":"lon"})
    if "y" in dd.coords: dd = dd.rename({"y":"lat"})
    dem = dd["z"].squeeze().sel(
        lon=slice(STUDY_LON_MIN, STUDY_LON_MAX),
        lat=slice(STUDY_LAT_MIN, STUDY_LAT_MAX))
    dem_i = dem.sortby("lat").sortby("lon").interp(
        lat=lat, lon=lon, method="linear").values
    dd.close()

    dem_hr = (dem >= LAND_MIN_Z).astype(float).sortby("lat").sortby("lon")
    land = dem_hr.interp(lat=lat, lon=lon, method="nearest").values
    land = uniform_filter(land, size=3, mode="constant") >= 0.75
    dem_land = np.where(land, dem_i, np.nan)

    dy = np.gradient(dem_land, axis=0) * 111000
    dx = np.gradient(dem_land, axis=1) * 111000
    slope = np.sqrt(dy**2 + dx**2)
    sc = np.nanpercentile(slope, 95)
    slope_n = np.clip(slope/sc, 0, 1)

    el = dem_land[~np.isnan(dem_land)]
    el_f, el_c = np.percentile(el, 5), np.nanmax(el)
    elev_n = np.clip((dem_land-el_f)/(el_c-el_f), 0, 1)
    elev_w = 1 - elev_n

    terrain = 0.5*slope_n + 0.5*elev_w
    terrain = np.where(np.isnan(dem_land), np.nan, terrain)

    risk = sens * terrain
    risk = np.where((risk >= RISK_MIN) & np.isfinite(risk), risk, np.nan)

    out = xr.Dataset(
        {"sensitivity":   (("lat","lon"), sens.astype("float32")),
         "terrain_weight":(("lat","lon"), terrain.astype("float32")),
         "risk_index":    (("lat","lon"), risk.astype("float32"))},
        coords={"lat":lat, "lon":lon})
    fp = PREP/"risk"/"fig13_risk_mosaic.nc"
    out.to_netcdf(fp, engine="netcdf4")
    register(fp,"Fig.13","Risk mosaic grid")
    log(f"Risk mosaic saved: {int(np.isfinite(risk).sum())} cells ≥ {RISK_MIN:.0f}", "OK")


# =============================================================================
# MANIFEST + MAIN
# =============================================================================
def write_manifest():
    pd.DataFrame(MANIFEST).to_csv(PREP/"manifest.csv", index=False)
    log(f"Manifest: {len(MANIFEST)} outputs", "OK")

def main():
    log("="*70, "STEP"); log("REVISED PREP v2 — START", "STEP"); log("="*70, "STEP")
    log(f"Domain: lon {STUDY_LON_MIN}–{STUDY_LON_MAX}°E, lat {STUDY_LAT_MIN}–{STUDY_LAT_MAX}°N", "INFO")
    log(f"Output root: {PREP}", "INFO")

    prepare_fig01_metadata()
    master = build_master_daily()
    prepare_fig03_climatology()
    annual, daily_lab = prepare_fig04_annual_and_trends(master)
    prepare_fig05_spatial_trends()
    annual_thr, daily_thr, final_thr = prepare_fig06_thresholds(annual, daily_lab)
    ev, cat, trigger = prepare_events(daily_lab, final_thr)
    prepare_fig09_composites(daily_lab, ev)
    prepare_fig10_synoptic()
    prepare_fig11_climate(annual, ev)
    prepare_fig12_operational(daily_lab, final_thr)
    prepare_fig13_risk(final_thr)
    write_manifest()

    log("="*70, "STEP"); log(f"✅ PREP COMPLETE in {time.time()-_T0:.1f}s", "OK"); log("="*70, "STEP")
    log(f"Output: {PREP}", "OK")
    log(f"Manifest: {PREP/'manifest.csv'}", "OK")

if __name__ == "__main__":
    main()
```

    [17:10:50] 🔹 ======================================================================
    [17:10:50] 🔹 REVISED PREP v2 — START
    [17:10:50] 🔹 ======================================================================
    [17:10:50] ℹ️ Domain: lon 90.0–105.0°E, lat 20.0–26.5°N
    [17:10:50] ℹ️ Output root: G:\NUIST_YTU_Water\prep_revised_v2
    [17:10:50] 📦 ======================================================================
    [17:10:50] 📦 SECTION 1: Fig.1 metadata
    [17:10:50] 📦 ======================================================================
    [17:10:51] ✅ Basin bounds: [94.86249999999959, 20.85416666666636, 100.22916666666632, 26.5]
    [17:10:51] ✅ Stations retained: 14 | excluded: ['Hkamti']
    [17:10:51] 📦 ======================================================================
    [17:10:51] 📦 SECTION 2: Master daily table
    [17:10:51] 📦 ======================================================================
    [17:10:51] ℹ️ Clipping SM to basin and computing daily mean...
    [17:10:54] ✅ Master daily rows: 16436 | SM non-NaN: 16436
    [17:10:54] ✅ Years incomplete: []
    [17:10:54] 📦 ======================================================================
    [17:10:54] 📦 SECTION 3: Fig.3 climatology
    [17:10:54] 📦 ======================================================================
    [17:10:55] ✅ Saved: fig03_jjas_rain.nc | shape {'lon': 300, 'lat': 130}
    [17:10:55] ✅ Saved: fig03_jjas_temp.nc | shape {'lon': 61, 'lat': 27}
    [17:10:55] ✅ Saved: fig03_jjas_sm.nc | shape {'lon': 60, 'lat': 26}
    [17:10:55] ✅ Saved: fig03_monthly_climatology.csv | cols=['Month', 'Rainfall_mm', 'Temperature_C', 'SoilMoisture_m3m3']
    [17:10:55] 📦 ======================================================================
    [17:10:55] 📦 SECTION 4: Fig.4 annual + trends
    [17:10:55] 📦 ======================================================================
    [17:10:55] ✅ Annual table: 45 years, 17 cols
    [17:10:55] ✅ Flood-labelled days: 96
    [17:10:55] 📦 ======================================================================
    [17:10:55] 📦 SECTION 5: Fig.5 spatial trends
    [17:10:55] 📦 ======================================================================
    [17:10:55] ℹ️   Rx1day source md5-head: 287bc70a79  (Rx1day_MM_chirps.nc)
    [17:10:55] ℹ️   Rx5day source md5-head: 24c7f81855  (Rx5day_MM_chirps.nc)
    [17:10:55] ✅ Rx1day: grid (45, 130, 300), var='rx1dayETCCDI'
    [17:10:59] ✅ Saved: fig05_rx1day_trend_grid.nc | slope -13.61–18.67
    [17:10:59] ✅ Rx5day: grid (45, 130, 300), var='rx5dayETCCDI'
    [17:11:02] ✅ Saved: fig05_rx5day_trend_grid.nc | slope -13.61–18.67
    [17:11:02] ✅   Correlation of Rx1 vs Rx5 trend fields: r = 1.000
    [17:11:02] ⚠️   ⚠️ Rx1/Rx5 trends nearly identical — check source NetCDFs!
    [17:11:02] 📦 ======================================================================
    [17:11:02] 📦 SECTION 6: Fig.6 ML thresholds
    [17:11:02] 📦 ======================================================================
    [17:11:02] ✅ Annual thresholds: 11 variables
    [17:11:02] ✅   SWRI_annual_max      AUC=0.666 thr=26.95
    [17:11:02] ✅   SWRI_JJAS_mean       AUC=0.629 thr=5.17
    [17:11:02] ✅   R95p_Stn             AUC=0.580 thr=11.00
    [17:11:02] ✅   SM_JJAS_mean         AUC=0.574 thr=0.34
    [17:11:02] ✅   Rx1day_Stn           AUC=0.557 thr=41.12
    [17:11:02] ✅   R99p_Stn             AUC=0.537 thr=3.00
    [17:11:02] ✅   Rx1day_CHIRPS        AUC=0.529 thr=73.88
    [17:11:02] ✅   Rx5day_Stn           AUC=0.471 thr=124.18
    [17:11:02] ✅   Rx5day_CHIRPS        AUC=0.434 thr=135.51
    [17:11:02] ✅   Rx5day_20kmBuffer    AUC=0.423 thr=167.94
    [17:11:02] ✅   SM_annual_max        AUC=0.377 thr=0.37
    [17:11:02] ℹ️ Daily ROC uses wet-day filter P≥5 mm for rainfall vars
    [17:11:02] ✅   SM               AUC=0.810 thr=0.323 POD=0.92 FAR=0.37
    [17:11:02] ✅   Rx5day_Stn       AUC=0.604 thr=52.969 POD=0.54 FAR=0.35
    [17:11:02] ✅   Rx5day_CHIRPS    AUC=0.595 thr=49.731 POD=0.58 FAR=0.43
    [17:11:02] ✅   SWRI             AUC=0.551 thr=3.102 POD=1.00 FAR=0.89
    [17:11:02] ✅   Rx1day_Stn       AUC=0.516 thr=23.235 POD=0.18 FAR=0.12
    [17:11:03] ✅ RF importance: {'SWRI_max': 0.296, 'Rx1_stn': 0.269, 'Rx5_stn': 0.23, 'SM_max': 0.205}
    [17:11:03] ✅ Final operating thresholds: {'Rx1day_Stn': 41.124201142857146, 'Rx5day_Stn': 124.17829128571428, 'SM': 0.3227272033691406, 'SWRI': 3.101592151984086}
    [17:11:03] 📦 ======================================================================
    [17:11:03] 📦 SECTION 7: Fig.7–8 event catalog
    [17:11:03] 📦 ======================================================================
    [17:11:03] ✅ Trigger: Rx5≥124.2 mm & SM≥0.323 & SWRI≥3.102
    [17:11:03] ✅ Trigger days: 25
    [17:11:03] ✅ Raw windows: 11 → merged: 11
    [17:11:03] ✅ Events: 11 | matched: 1
    [17:11:03] ✅ Categories: {'Prolonged': 6, 'Short': 5}
    [17:11:03] ✅ Trigger daily saved.
    [17:11:03] 📦 ======================================================================
    [17:11:03] 📦 SECTION 8: Fig.9 category composites
    [17:11:03] 📦 ======================================================================
    [17:11:04] ✅ Composites saved: 77 category-rows
    [17:11:04] 📦 ======================================================================
    [17:11:04] 📦 SECTION 9: Fig.10 synoptic verify
    [17:11:04] 📦 ======================================================================
    [17:11:04] ✅ Verified: MM_pr_wtr_composite_event.nc
    [17:11:04] ✅ Verified: MM_pr_wtr_anomaly_event.nc
    [17:11:04] ✅ Verified: MM_hgt_500_composite_event.nc
    [17:11:04] ✅ Verified: MM_hgt_500_anomaly_event.nc
    [17:11:04] ✅ Verified: MM_uwnd_850_composite_event.nc
    [17:11:04] ✅ Verified: MM_vwnd_850_composite_event.nc
    [17:11:04] ✅ Verified: MM_pr_wtr_daily.nc
    [17:11:04] 📦 ======================================================================
    [17:11:04] 📦 SECTION 10: Fig.11 climate linkage
    [17:11:04] 📦 ======================================================================
    [17:11:04] ✅ RF importance (flood-year target): {'ONI': 0.326, 'WYI': 0.283, 'DMI': 0.199, 'SAMI': 0.192}
    [17:11:04] 📦 ======================================================================
    [17:11:04] 📦 SECTION 11: Fig.12 operational + RF
    [17:11:04] 📦 ======================================================================
    [17:11:04] ✅ Binned probability: 12 bins
    [17:11:05] ✅ Operational tiers: {'P0_30_mm': 15.309794178453188, 'P0_60_mm': 43.04076686700014, 'T_RX1': 41.124201142857146, 'T_RX5': 124.17829128571428, 'T_SM': 0.3227272033691406, 'T_SWRI': 3.101592151984086}
    [17:11:05] ✅ RF daily: AUC=0.823 POD=0.000 FAR=0.000 Brier=0.0057
    [17:11:05] 📦 ======================================================================
    [17:11:05] 📦 SECTION 12: Fig.13 risk mosaic
    [17:11:05] 📦 ======================================================================
    [17:11:05] ✅ CHIRPS grid for risk: (16436, 130, 300)
    [17:11:05] ✅ Risk thresholds: Rx5≥124.2 mm, Rain≥41.1 mm
    [17:11:39] ℹ️ Loading ETOPO1...
    [17:11:39] ✅ Risk mosaic saved: 5214 cells ≥ 55
    [17:11:40] ✅ Manifest: 41 outputs
    [17:11:40] 🔹 ======================================================================
    [17:11:40] ✅ ✅ PREP COMPLETE in 49.3s
    [17:11:40] 🔹 ======================================================================
    [17:11:40] ✅ Output: G:\NUIST_YTU_Water\prep_revised_v2
    [17:11:40] ✅ Manifest: G:\NUIST_YTU_Water\prep_revised_v2\manifest.csv
    


```python

```


```python
# ============================================================
# FIGURE 1: Shwe Li River Basin — revised
# Panel A: Southeast Asia context (ETOPO1 elevation + red box)
# Panel B: Study area — rivers & boundary      (from make_custom_basin_final.py panel a)
# Panel C: Masked AW3D30 elevation             (from make_custom_basin_final.py panel b)
#
# Panels B/C follow the study-area definition:
#   basin  = custom_basin.shp
#   rivers = clipped_rivers.shp  (Sub_Name split: Shweli / Irrawaddy / others)
#   DEM    = AW3D30_DEM_Shweli.tif (÷5 downsample, "90 m")
#   border = Natural Earth 50m admin_0_boundary_lines_land, clipped to basin.buffer(0.02)
#   stations = all_15_stations.csv with "Hkamti" dropped (14 stations)
# ============================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
import xarray as xr
import rasterio
from shapely.geometry import box

# --------------------------------------------
# 1. CONFIGURATION - FILE PATHS
# --------------------------------------------
etopo_path    = r"G:\ETOPO1_Bed_g_gmt4.nc"
DEM_PATH      = r"G:\NUIST_YTU_Water\data\AW3D30_DEM_Shweli.tif"
BASIN_SHP     = r"G:\NUIST_YTU_Water\output_shapefiles\custom_basin.shp"
RIVER_SHP     = r"G:\NUIST_YTU_Water\output_shapefiles\clipped_rivers.shp"
STATION_CSV   = r"G:\NUIST_YTU_Water\data\rainfall\all_15_stations.csv"
OUT_DIR       = r"G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------
# 2. LOAD PRE-BUILT BASIN + RIVERS
# --------------------------------------------
print("Loading pre-built Shweli basin and rivers...")
custom_basin   = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
clipped_rivers = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")

minx, miny, maxx, maxy = custom_basin.total_bounds
print(f"[i] custom_basin.total_bounds = ({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f})")

basin_geom   = custom_basin.geometry.union_all()
basin_buffer = basin_geom.buffer(0.02)   # ~2 km buffer for clean clip

# Stations: 14 retained (Hkamti dropped)
df = pd.read_csv(STATION_CSV)
df = df[df["Station"].str.strip() != "Hkamti"].copy()
df["Station"] = df["Station"].str.strip()
stations = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326"
)
print(f"✅ {len(stations)} stations retained (Hkamti excluded).")

# --------------------------------------------
# 3. LOAD AW3D30 DEM (memory-optimised, ÷5)
# --------------------------------------------
print("Loading High-Res AW3D30 DEM (downsampling for memory)...")
with rasterio.open(DEM_PATH) as src:
    elev_data = src.read(1, out_shape=(src.height // 5, src.width // 5))
    elev_data = elev_data.astype('float32')
    bounds = src.bounds
    lon_min_dem, lat_min_dem, lon_max_dem, lat_max_dem = (
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )

lons_1d = np.linspace(lon_min_dem, lon_max_dem, elev_data.shape[1])
lats_1d = np.linspace(lat_min_dem, lat_max_dem, elev_data.shape[0])

print("Computing Hillshade...")
ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elev_data, vert_exag=2)

# Separate rivers for styling
shweli = clipped_rivers[
    clipped_rivers["Sub_Name"].astype(str).str.contains("Shweli", case=False, na=False)
]
irrawaddy = clipped_rivers[
    clipped_rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)
]
others = clipped_rivers[
    ~clipped_rivers.index.isin(shweli.index)
    & ~clipped_rivers.index.isin(irrawaddy.index)
]

# --------------------------------------------
# 4. PRE-CLIP COUNTRY BORDERS TO STUDY AREA
# --------------------------------------------
print("Clipping country borders to study area...")
borders_path = shapereader.natural_earth(
    resolution='50m', category='cultural', name='admin_0_boundary_lines_land'
)
borders_gdf = gpd.read_file(borders_path).to_crs("EPSG:4326")
borders_clip = borders_gdf[borders_gdf.intersects(basin_buffer)].copy()
borders_clip["geometry"] = borders_clip.geometry.intersection(basin_buffer)
borders_clip = borders_clip[~borders_clip.is_empty]
if borders_clip.empty:
    print("[!] No country border intersects the basin — border layer will be empty.")

# --------------------------------------------
# 5. EXACT BASIN EXTENT (same as reference script)
# --------------------------------------------
pad = 0.1
extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
lon_min, lon_max = extent[0], extent[1]
lat_min, lat_max = extent[2], extent[3]
print(f"Study domain: lon {lon_min:.2f}–{lon_max:.2f}, lat {lat_min:.2f}–{lat_max:.2f}")

# --------------------------------------------
# 6. LOAD ETOPO1 (used only for Panel A context)
# --------------------------------------------
print("Loading ETOPO1 elevation...")
ds = xr.open_dataset(etopo_path)
if 'x' in ds.coords and 'y' in ds.coords:
    ds = ds.rename({'x': 'lon', 'y': 'lat'})
elev_full = ds['z'].squeeze()
ds.close()

sea_lon_min, sea_lon_max = 85, 110
sea_lat_min, sea_lat_max = 5, 35

# --------------------------------------------
# 7. CREATE FIGURE (1 context + 2 study panels)
# --------------------------------------------
print("Plotting...")
fig = plt.figure(figsize=(24, 8), dpi=300)

# helper: shared overlays
def add_border_clip(ax, zorder=5):
    if not borders_clip.empty:
        borders_clip.plot(ax=ax, linestyle='--', edgecolor='magenta',
                          linewidth=2.0, transform=ccrs.PlateCarree(),
                          zorder=zorder)

def add_country_labels(ax, zorder=6):
    ax.text(maxx - 0.6, maxy - 0.6, 'CHINA', fontsize=18, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(),
            zorder=zorder, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor='black'))
    ax.text(minx + 1.0, miny + 0.8, 'MYANMAR', fontsize=18, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(),
            zorder=zorder, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor='black'))

# =============================================================================
# PANEL (a) — Southeast Asia context (unchanged)
# =============================================================================
ax_sea = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
ax_sea.set_extent([sea_lon_min, sea_lon_max, sea_lat_min, sea_lat_max])

elev_sea = elev_full.sel(lon=slice(sea_lon_min, sea_lon_max),
                         lat=slice(sea_lat_min, sea_lat_max))
elev_data_sea = elev_sea.values
ls_sea = LightSource(azdeg=315, altdeg=45)
hillshade_sea = ls_sea.hillshade(elev_data_sea, vert_exag=2, dx=1, dy=1)

im_sea = ax_sea.imshow(elev_data_sea, origin='upper',
                       extent=[sea_lon_min, sea_lon_max, sea_lat_min, sea_lat_max],
                       cmap='terrain', alpha=0.7,
                       transform=ccrs.PlateCarree(), zorder=1)
ax_sea.imshow(hillshade_sea, origin='upper',
              extent=[sea_lon_min, sea_lon_max, sea_lat_min, sea_lat_max],
              cmap='gray', alpha=0.3, transform=ccrs.PlateCarree(), zorder=2)
ax_sea.add_feature(cfeature.COASTLINE, linewidth=1.5)
ax_sea.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1.5, alpha=0.7)

# Red rectangle now follows custom_basin bounds
rect = Rectangle((minx, miny), maxx - minx, maxy - miny,
                 facecolor='none', edgecolor='red', linewidth=2,
                 transform=ccrs.PlateCarree(), zorder=10)
ax_sea.add_patch(rect)

ax_sea.set_title('(a) Southeast Asia', fontsize=16, fontweight='bold')
gl_sea = ax_sea.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl_sea.top_labels = False
gl_sea.right_labels = False
gl_sea.xlabel_style = {'size': 14, 'weight': 'bold'}
gl_sea.ylabel_style = {'size': 14, 'weight': 'bold'}

# =============================================================================
# PANEL (b) — Rivers & boundary  (from make_custom_basin_final.py panel a)
# =============================================================================
ax1 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
ax1.set_extent(extent, crs=ccrs.PlateCarree())

custom_basin.plot(ax=ax1, facecolor='#e0f3f8', edgecolor='navy',
                  linewidth=2.5, transform=ccrs.PlateCarree(), zorder=1)

if not others.empty:
    others.plot(ax=ax1, color='dodgerblue', linewidth=1.5, alpha=0.7,
                transform=ccrs.PlateCarree(), zorder=2)
if not shweli.empty:
    shweli.plot(ax=ax1, color='crimson', linewidth=3.5,
                transform=ccrs.PlateCarree(), zorder=3)
if not irrawaddy.empty:
    irrawaddy.plot(ax=ax1, color='darkorange', linewidth=3.5,
                   transform=ccrs.PlateCarree(), zorder=3)

stations.plot(ax=ax1, color='red', marker='^', markersize=70,
              edgecolor='black', linewidth=0.8,
              transform=ccrs.PlateCarree(), zorder=4)

add_border_clip(ax1, zorder=5)
add_country_labels(ax1, zorder=6)

ax1.set_title('(b) Shweli Basin — Rivers & Boundary',
              fontsize=16, fontweight='bold')
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': 14, 'weight': 'bold'}
gl1.ylabel_style = {'size': 14, 'weight': 'bold'}

handles1 = [
    Line2D([0], [0], color='navy', lw=2.5, label='Basin Boundary'),
    Line2D([0], [0], color='magenta', linestyle='--', lw=2.0,
           label='Myanmar-China Border (clipped)'),
    Line2D([0], [0], color='crimson', lw=3.5, label='Shweli River'),
    Line2D([0], [0], color='darkorange', lw=3.5, label='Irrawaddy River'),
    Line2D([0], [0], color='dodgerblue', lw=1.5, label='Tributaries'),
    Line2D([0], [0], marker='^', color='red', markeredgecolor='black',
           linestyle='None', markersize=10, label='Stations'),
]
ax1.legend(handles=handles1, loc='lower right', fontsize=12,
           frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')

# =============================================================================
# PANEL (c) — Masked AW3D30 elevation  (from make_custom_basin_final.py panel b)
# =============================================================================
ax2 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
ax2.set_extent(extent, crs=ccrs.PlateCarree())

levels = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200]
contour = ax2.contourf(lons_1d, lats_1d, elev_data, levels=levels,
                       cmap='terrain', transform=ccrs.PlateCarree(),
                       zorder=1, alpha=0.8, extend='max')

ax2.imshow(hillshade, origin='upper',
           extent=[lon_min_dem, lon_max_dem, lat_min_dem, lat_max_dem],
           cmap='gray', alpha=0.3,
           transform=ccrs.PlateCarree(), zorder=2)

# Mask outside the pre-built Shweli polygon
bbox = box(lon_min_dem, lat_min_dem, lon_max_dem, lat_max_dem)
inverse_mask = bbox.difference(basin_geom)
gpd.GeoSeries([inverse_mask], crs="EPSG:4326").plot(
    ax=ax2, color='white', alpha=1.0,
    transform=ccrs.PlateCarree(), zorder=3)

if not others.empty:
    others.plot(ax=ax2, color='dodgerblue', linewidth=1.5, alpha=0.7,
                transform=ccrs.PlateCarree(), zorder=4)
if not shweli.empty:
    shweli.plot(ax=ax2, color='crimson', linewidth=3.5,
                transform=ccrs.PlateCarree(), zorder=5)
if not irrawaddy.empty:
    irrawaddy.plot(ax=ax2, color='darkorange', linewidth=3.5,
                   transform=ccrs.PlateCarree(), zorder=5)

stations.plot(ax=ax2, color='red', marker='^', markersize=70,
              edgecolor='black', linewidth=0.8,
              transform=ccrs.PlateCarree(), zorder=6)
custom_basin.plot(ax=ax2, facecolor='none', edgecolor='navy',
                  linewidth=2.5, transform=ccrs.PlateCarree(), zorder=7)

add_border_clip(ax2, zorder=8)
add_country_labels(ax2, zorder=9)

ax2.set_title('(c) Masked Elevation (AW3D30 90 m)',
              fontsize=16, fontweight='bold')
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 14, 'weight': 'bold'}
gl2.ylabel_style = {'size': 14, 'weight': 'bold'}

handles2 = [
    Line2D([0], [0], color='navy', lw=2.5, label='Basin Boundary'),
    Line2D([0], [0], color='magenta', linestyle='--', lw=2.0,
           label='Myanmar-China Border (clipped)'),
    Line2D([0], [0], color='crimson', lw=3.5, label='Shweli River'),
    Line2D([0], [0], color='darkorange', lw=3.5, label='Irrawaddy River'),
    Line2D([0], [0], color='dodgerblue', lw=1.5, label='Tributaries'),
    Line2D([0], [0], marker='^', color='red', markeredgecolor='black',
           linestyle='None', markersize=10, label='Stations'),
]
ax2.legend(handles=handles2, loc='lower right', fontsize=12,
           frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')

# --------------------------------------------
# 8. COLORBARS
# --------------------------------------------
# ETOPO1 elevation (Panel A)
cbar_ax1 = fig.add_axes([0.06, 0.08, 0.16, 0.02])
cbar1 = fig.colorbar(im_sea, cax=cbar_ax1, orientation='horizontal')
cbar1.set_label('ETOPO1 Elevation (m)', fontsize=12, fontweight='bold')
cbar1.ax.tick_params(labelsize=10)

# AW3D30 elevation (Panel C)
cbar_ax2 = fig.add_axes([0.72, 0.08, 0.20, 0.02])
cbar2 = fig.colorbar(contour, cax=cbar_ax2, orientation='horizontal', ticks=levels)
cbar2.set_label('Elevation (m)', fontsize=12, fontweight='bold')
cbar2.ax.tick_params(labelsize=10)

plt.tight_layout(rect=[0, 0.10, 1, 1])

# --------------------------------------------
# 9. SAVE
# --------------------------------------------
out_fig = os.path.join(OUT_DIR, "Fig01_Basin_AW3D30_Map_3panel.png")
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
print(f"✅ Saved {out_fig}")
plt.show()
```

    Loading pre-built Shweli basin and rivers...
    [i] custom_basin.total_bounds = (94.8625, 20.8542, 100.2292, 26.5000)
    ✅ 14 stations retained (Hkamti excluded).
    Loading High-Res AW3D30 DEM (downsampling for memory)...
    Computing Hillshade...
    Clipping country borders to study area...
    Study domain: lon 94.76–100.33, lat 20.75–26.60
    Loading ETOPO1 elevation...
    Plotting...
    ✅ Saved G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig01_Basin_AW3D30_Map_3panel.png
    


    
![png](output_3_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_custom_basin.py (FINAL SAVE + SPLIT RIVERS)
- Cuts the north at 26.5N
- Fills all interior gaps (Sealed)
- Clips rivers so ONLY those inside the basin are saved.
- SAVES:
    * custom_basin.shp      (single basin polygon, NO rivers)
    * clipped_rivers.shp    (all rivers inside basin)
    * shweli_river.shp      (Shweli only)
    * irrawaddy_river.shp   (Irrawaddy only)
    * other_rivers.shp      (everything else)
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.ops import unary_union
from shapely.geometry import Polygon, box

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASIN_SHP   = r"G:\NUIST_YTU_Water\data\Hydrobasin_L6_SE-ASIA.shp"
RIVER_SHP   = r"G:\NUIST_YTU_Water\data\gms_river.shp"
STATION_CSV = r"G:\NUIST_YTU_Water\data\rainfall\all_15_stations.csv"
OUT_DIR     = r"G:\NUIST_YTU_Water\output_shapefiles"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load and filter stations
# -----------------------------------------------------------------------------
df = pd.read_csv(STATION_CSV)
df = df[df["Station"].str.strip() != "Hkamti"].copy()
stations = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326"
)

# -----------------------------------------------------------------------------
# 2. Load basins and merge
# -----------------------------------------------------------------------------
gdf_basins = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
joined = gpd.sjoin(gdf_basins, stations, how="inner", predicate="contains")
basins_with_stations = joined["SUB_NAME"].unique()
selected_basins = gdf_basins[gdf_basins["SUB_NAME"].isin(basins_with_stations)].copy()
merged_geometry = unary_union(selected_basins.geometry.values)

# -----------------------------------------------------------------------------
# 3. APPLY CUSTOM CROP & FILL
# -----------------------------------------------------------------------------
# A. Northern Cut
crop_polygon = box(90.0, 20.0, 105.0, 26.5)
cropped_geometry = merged_geometry.intersection(crop_polygon)

# B. Fill the central/eastern gap and upper Bhamo
fill_coords = [
    (96.8, 22.8), (97.6, 22.8), (98.5, 22.8), (99.2, 23.8),
    (99.2, 24.8), (98.8, 25.6), (98.0, 25.6), (97.2, 25.4),
    (96.8, 24.5), (96.8, 22.8)
]
fill_poly = Polygon(fill_coords)

# Union and seal holes
combined_geom = cropped_geometry.union(fill_poly).buffer(0)
if combined_geom.geom_type == 'Polygon':
    final_geometry = Polygon(combined_geom.exterior)
elif combined_geom.geom_type == 'MultiPolygon':
    clean_parts = [Polygon(part.exterior) for part in combined_geom.geoms]
    final_geometry = unary_union(clean_parts)
else:
    final_geometry = combined_geom

custom_gdf = gpd.GeoDataFrame(
    {"basin_name": ["Custom_Study_Basin"]},
    geometry=[final_geometry],
    crs="EPSG:4326",
)

# -----------------------------------------------------------------------------
# 4. LOAD, CLIP & SPLIT RIVERS
# -----------------------------------------------------------------------------
gdf_rivers = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")

# Clip rivers strictly to the final basin geometry
clipped_rivers = gpd.clip(gdf_rivers, custom_gdf)

# Split into the three groups
shweli = clipped_rivers[
    clipped_rivers["Sub_Name"].astype(str).str.contains("Shweli", case=False, na=False)
].copy()

irrawaddy = clipped_rivers[
    clipped_rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)
].copy()

others = clipped_rivers[
    ~clipped_rivers.index.isin(shweli.index)
    & ~clipped_rivers.index.isin(irrawaddy.index)
].copy()

# -----------------------------------------------------------------------------
# 5. SAVE ALL SHAPEFILES
# -----------------------------------------------------------------------------
shp_basin_out    = os.path.join(OUT_DIR, "custom_basin.shp")
shp_river_out    = os.path.join(OUT_DIR, "clipped_rivers.shp")
shp_shweli_out   = os.path.join(OUT_DIR, "shweli_river.shp")
shp_irra_out     = os.path.join(OUT_DIR, "irrawaddy_river.shp")
shp_others_out   = os.path.join(OUT_DIR, "other_rivers.shp")

custom_gdf.to_file(shp_basin_out)          # basin only, no rivers
clipped_rivers.to_file(shp_river_out)      # all rivers inside basin
shweli.to_file(shp_shweli_out)             # Shweli only
irrawaddy.to_file(shp_irra_out)            # Irrawaddy only
others.to_file(shp_others_out)             # everything else

print(f"✅ Saved Custom Basin     -> {shp_basin_out}")
print(f"✅ Saved All Rivers       -> {shp_river_out}")
print(f"✅ Saved Shweli River     -> {shp_shweli_out}  ({len(shweli)} features)")
print(f"✅ Saved Irrawaddy River  -> {shp_irra_out}  ({len(irrawaddy)} features)")
print(f"✅ Saved Other Rivers     -> {shp_others_out}  ({len(others)} features)")

# -----------------------------------------------------------------------------
# 6. FINAL QA MAP
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

custom_gdf.plot(ax=ax, facecolor="#a6cee3", edgecolor="navy",
                linewidth=2.5, alpha=0.4, label="Custom Basin")

if not others.empty:
    others.plot(ax=ax, color="dodgerblue", linewidth=1, alpha=0.5,
                label="Other Rivers")
if not shweli.empty:
    shweli.plot(ax=ax, color="crimson", linewidth=3, label="Shweli River")
if not irrawaddy.empty:
    irrawaddy.plot(ax=ax, color="darkorange", linewidth=3,
                   label="Irrawaddy River (Clipped)")

stations.plot(ax=ax, color="red", marker="^", markersize=80,
              edgecolor="black", linewidth=0.8, zorder=10, label="Stations")
for _, row in stations.iterrows():
    ax.text(row["Lon"] + 0.05, row["Lat"] + 0.05, row["Station"],
            fontsize=8, zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8))

ax.set_title("FINAL SAVED: Custom Basin & Split Rivers",
             fontsize=14, fontweight="bold", color="darkgreen")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

handles = [
    Line2D([0], [0], color="navy", lw=2.5, label="Basin Boundary"),
    Line2D([0], [0], color="crimson", lw=3, label="Shweli River"),
    Line2D([0], [0], color="darkorange", lw=3, label="Irrawaddy (Clipped)"),
    Line2D([0], [0], color="dodgerblue", lw=1, label="Other Rivers"),
    Line2D([0], [0], marker="^", color="red", linestyle="None",
           markersize=8, label="Stations"),
]
ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True)

minx, miny, maxx, maxy = custom_gdf.total_bounds
pad = 0.5
ax.set_xlim(minx - pad, maxx + pad)
ax.set_ylim(miny - pad, maxy + pad)
ax.grid(alpha=0.3)

plt.show()
```

    ✅ Saved Custom Basin     -> G:\NUIST_YTU_Water\output_shapefiles\custom_basin.shp
    ✅ Saved All Rivers       -> G:\NUIST_YTU_Water\output_shapefiles\clipped_rivers.shp
    ✅ Saved Shweli River     -> G:\NUIST_YTU_Water\output_shapefiles\shweli_river.shp  (32 features)
    ✅ Saved Irrawaddy River  -> G:\NUIST_YTU_Water\output_shapefiles\irrawaddy_river.shp  (58 features)
    ✅ Saved Other Rivers     -> G:\NUIST_YTU_Water\output_shapefiles\other_rivers.shp  (182 features)
    


    
![png](output_5_1.png)
    



```python

```


```python
import os
import glob
import zipfile
import shutil
import rasterio
from rasterio.merge import merge

# --- CONFIGURATION ---
ZIP_DIR = r"G:\NUIST_YTU_Water\data\DEM"
OUTPUT_TIF = r"G:\NUIST_YTU_Water\data\AW3D30_Merged.tif"
TEMP_EXTRACT_DIR = os.path.join(ZIP_DIR, "_temp_extract")

os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

# 1. Find and extract all zip files
zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
print(f"Found {len(zip_files)} zip files. Extracting...")

for zf in zip_files:
    print(f"  Extracting: {os.path.basename(zf)}")
    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(TEMP_EXTRACT_DIR)

# 2. Find all extracted .tif files (JAXA usually puts them in subfolders)
tif_files = glob.glob(os.path.join(TEMP_EXTRACT_DIR, "**", "*.tif"), recursive=True)
print(f"\nFound {len(tif_files)} .tif files. Merging...")

# 3. Merge all .tif files
src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
mosaic, out_trans = merge(src_files_to_mosaic)

# 4. Update metadata and save
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "crs": "EPSG:4326"
})

with rasterio.open(OUTPUT_TIF, "w", **out_meta) as dest:
    dest.write(mosaic)

# 5. Close files and clean up
for src in src_files_to_mosaic:
    src.close()

print(f"\n✅ Successfully merged all DEM tiles into: {OUTPUT_TIF}")

# Clean up temporary extraction folder to save space
shutil.rmtree(TEMP_EXTRACT_DIR)
print("Cleaned up temporary files.")
```

    Found 5 zip files. Extracting...
      Extracting: N020E095_N025E100.zip
      Extracting: N020E100_N025E105.zip
      Extracting: N025E090_N030E095.zip
      Extracting: N025E095_N030E100.zip
    


```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 — Study area map, Shweli–Ayeyarwady basin
=========================================================================
Triangle layout with LARGE, CLEAR typography:
    Row 1:                (a) Southeast Asia context  (full width, centred)
    Row 2:  (b) basin + rivers + stations | (c) masked AW3D30 elevation

Reading order: zoom from SE Asia → basin → elevation detail.

Reads:
    prep_revised_v2/fig01/fig01_metadata.json
    prep_revised_v2/fig01/fig01_stations_retained14.csv
    output_shapefiles/custom_basin.shp
    output_shapefiles/clipped_rivers.shp
    data/AW3D30_DEM_Shweli.tif
    G:\\ETOPO1_Bed_g_gmt4.nc

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig01_Basin_Map_v2.png / .pdf
"""
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
import xarray as xr
import rasterio
from shapely.geometry import box

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
SHP_DIR = ROOT / "output_shapefiles"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASIN_SHP   = SHP_DIR / "custom_basin.shp"
RIVER_SHP   = SHP_DIR / "clipped_rivers.shp"
STN_CSV     = PREP / "fig01" / "fig01_stations_retained14.csv"
META_JSON   = PREP / "fig01" / "fig01_metadata.json"
DEM_PATH    = ROOT / "data" / "AW3D30_DEM_Shweli.tif"
ETOPO_PATH  = Path(r"G:\ETOPO1_Bed_g_gmt4.nc")

# -----------------------------------------------------------------------------
# Publication style — LARGE typography
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 13, 'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# 1. Load basin, rivers, stations, metadata
# -----------------------------------------------------------------------------
meta = json.load(open(META_JSON))
n_retained = meta["n_stations_retained"]
excluded   = meta["stations_excluded"]

basin   = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
rivers  = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")
minx, miny, maxx, maxy = basin.total_bounds

basin_geom   = basin.geometry.union_all()
basin_buffer = basin_geom.buffer(0.02)

# Stations (14 retained, from prep output)
stn_df = pd.read_csv(STN_CSV)
stn_df["Station"] = stn_df["Station"].astype(str).str.strip()
stations = gpd.GeoDataFrame(
    stn_df, geometry=gpd.points_from_xy(stn_df["Lon"], stn_df["Lat"]),
    crs="EPSG:4326",
)
print(f"✅ {len(stations)} stations retained; excluded: {excluded}")

# Split rivers by role
shweli    = rivers[rivers["Sub_Name"].astype(str).str.contains("Shweli",    case=False, na=False)]
irrawaddy = rivers[rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)]
others    = rivers[~rivers.index.isin(shweli.index) & ~rivers.index.isin(irrawaddy.index)]

# Country borders — clipped to basin buffer
borders_path = shapereader.natural_earth(
    resolution='50m', category='cultural', name='admin_0_boundary_lines_land')
borders_gdf  = gpd.read_file(borders_path).to_crs("EPSG:4326")
borders_clip = borders_gdf[borders_gdf.intersects(basin_buffer)].copy()
borders_clip["geometry"] = borders_clip.geometry.intersection(basin_buffer)
borders_clip = borders_clip[~borders_clip.is_empty]
if borders_clip.empty:
    print("[!] No border intersects basin — border layer will be empty.")

# -----------------------------------------------------------------------------
# 2. Load AW3D30 DEM (downsampled ÷5 for memory)
# -----------------------------------------------------------------------------
print("Loading AW3D30 DEM...")
with rasterio.open(DEM_PATH) as src:
    elev_data = src.read(1, out_shape=(src.height // 5, src.width // 5)).astype('float32')
    bounds = src.bounds
    lon_min_dem, lat_min_dem, lon_max_dem, lat_max_dem = (
        bounds.left, bounds.bottom, bounds.right, bounds.top)

lons_1d = np.linspace(lon_min_dem, lon_max_dem, elev_data.shape[1])
lats_1d = np.linspace(lat_min_dem, lat_max_dem, elev_data.shape[0])

ls = LightSource(azdeg=315, altdeg=45)
hillshade = ls.hillshade(elev_data, vert_exag=2)

# -----------------------------------------------------------------------------
# 3. Load ETOPO1 for Panel (a)
# -----------------------------------------------------------------------------
print("Loading ETOPO1...")
ds = xr.open_dataset(ETOPO_PATH)
if 'x' in ds.coords and 'y' in ds.coords:
    ds = ds.rename({'x': 'lon', 'y': 'lat'})
elev_full = ds['z'].squeeze()
ds.close()

SEA_LON = (85, 110); SEA_LAT = (5, 35)

# Study extent with padding
pad = 0.1
extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
print(f"Study extent: lon {extent[0]:.2f}–{extent[1]:.2f}, "
      f"lat {extent[2]:.2f}–{extent[3]:.2f}")

# -----------------------------------------------------------------------------
# 4. Shared overlay helpers
# -----------------------------------------------------------------------------
def add_border_clip(ax, zorder=5):
    if not borders_clip.empty:
        borders_clip.plot(ax=ax, linestyle='--', edgecolor='magenta',
                          linewidth=2.4, transform=ccrs.PlateCarree(), zorder=zorder)

def add_country_labels(ax, zorder=6):
    ax.text(maxx - 0.6, maxy - 0.6, 'CHINA', fontsize=17, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(),
            zorder=zorder, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.85, edgecolor='black', linewidth=0.9))
    ax.text(minx + 1.0, miny + 0.8, 'MYANMAR', fontsize=17, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(),
            zorder=zorder, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.85, edgecolor='black', linewidth=0.9))

def add_station_labels(ax, zorder=7, fontsize=11):
    """Add short labels for the 14 stations."""
    for _, row in stations.iterrows():
        name = row["Station"].split("(")[0].strip()
        ax.text(row["Lon"] + 0.06, row["Lat"] + 0.05, name,
                fontsize=fontsize, fontweight='bold',
                transform=ccrs.PlateCarree(),
                zorder=zorder,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.85, edgecolor="#666", linewidth=0.5))

# -----------------------------------------------------------------------------
# 5. Figure — (a) alone on top, (b) + (c) below
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(24, 17), dpi=300)
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[0.85, 1.15],      # row 1 shorter (wide context map)
    wspace=0.06, hspace=0.10,
    left=0.045, right=0.945, top=0.945, bottom=0.06,
)

# Row 1: (a) spans both columns → centred full-width top
# Row 2: (b) left column, (c) right column
ax_a = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())

# ==========================================================
# (a) Southeast Asia context — top row, full width
# ==========================================================
ax_a.set_extent([SEA_LON[0], SEA_LON[1], SEA_LAT[0], SEA_LAT[1]],
                crs=ccrs.PlateCarree())

elev_sea = elev_full.sel(lon=slice(*SEA_LON), lat=slice(*SEA_LAT))
arr_sea = elev_sea.values
hs_sea  = ls.hillshade(arr_sea, vert_exag=2, dx=1, dy=1)

im_a = ax_a.imshow(arr_sea, origin='upper',
                    extent=[SEA_LON[0], SEA_LON[1], SEA_LAT[0], SEA_LAT[1]],
                    cmap='terrain', alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=1)
ax_a.imshow(hs_sea, origin='upper',
            extent=[SEA_LON[0], SEA_LON[1], SEA_LAT[0], SEA_LAT[1]],
            cmap='gray', alpha=0.3, transform=ccrs.PlateCarree(), zorder=2)
ax_a.add_feature(cfeature.COASTLINE, linewidth=1.4, zorder=3)
ax_a.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1.1,
                 alpha=0.7, zorder=4)

ax_a.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny,
                         facecolor='none', edgecolor='red', linewidth=2.6,
                         transform=ccrs.PlateCarree(), zorder=10))

ax_a.set_title('(a) Southeast Asia context',
               loc='left', pad=12, fontsize=18, fontweight='bold')
gl_a = ax_a.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                       alpha=0.5, linestyle=':')
gl_a.top_labels = False; gl_a.right_labels = False
gl_a.xlabel_style = {'size': 14, 'weight': 'bold'}
gl_a.ylabel_style = {'size': 14, 'weight': 'bold'}

# ==========================================================
# (b) Rivers, basin, 14 stations, border — bottom left
# ==========================================================
ax_b.set_extent(extent, crs=ccrs.PlateCarree())

basin.plot(ax=ax_b, facecolor='#e0f3f8', edgecolor='none',
           transform=ccrs.PlateCarree(), zorder=1)

if not others.empty:
    others.plot(ax=ax_b, color='dodgerblue', linewidth=1.4, alpha=0.7,
                transform=ccrs.PlateCarree(), zorder=2)
if not shweli.empty:
    shweli.plot(ax=ax_b, color='crimson', linewidth=3.2,
                transform=ccrs.PlateCarree(), zorder=3)
if not irrawaddy.empty:
    irrawaddy.plot(ax=ax_b, color='darkorange', linewidth=3.2,
                   transform=ccrs.PlateCarree(), zorder=3)

basin.plot(ax=ax_b, facecolor='none', edgecolor='navy', linewidth=2.6,
           transform=ccrs.PlateCarree(), zorder=4)

stations.plot(ax=ax_b, color='red', marker='^', markersize=130,
              edgecolor='black', linewidth=1.2,
              transform=ccrs.PlateCarree(), zorder=6)
add_station_labels(ax_b, zorder=7, fontsize=11)

add_border_clip(ax_b, zorder=5)
add_country_labels(ax_b, zorder=8)

ax_b.set_title('(b) Shweli–Ayeyarwady basin — rivers, basin, stations',
               loc='left', pad=12, fontsize=18, fontweight='bold')
gl_b = ax_b.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                       alpha=0.5, linestyle=':')
gl_b.top_labels = False; gl_b.right_labels = False
gl_b.xlabel_style = {'size': 14, 'weight': 'bold'}
gl_b.ylabel_style = {'size': 14, 'weight': 'bold'}

legend_b = [
    Line2D([0], [0], color='navy',       lw=2.6, label='Basin boundary'),
    Line2D([0], [0], color='magenta',    lw=2.4, ls='--',
           label='China–Myanmar border (clipped)'),
    Line2D([0], [0], color='crimson',    lw=3.2, label='Shweli River'),
    Line2D([0], [0], color='darkorange', lw=3.2, label='Ayeyarwady River'),
    Line2D([0], [0], color='dodgerblue', lw=1.4, label='Tributaries'),
    Line2D([0], [0], marker='^', color='red', markeredgecolor='black',
           linestyle='None', markersize=14, label=f'Stations (n = {n_retained})'),
]
ax_b.legend(handles=legend_b, loc='lower right', fontsize=13,
            framealpha=0.95, edgecolor='0.3', ncol=1)

# ==========================================================
# (c) Masked AW3D30 elevation — bottom right
# ==========================================================
ax_c.set_extent(extent, crs=ccrs.PlateCarree())

levels = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3600, 4200]
cf = ax_c.contourf(lons_1d, lats_1d, elev_data, levels=levels,
                   cmap='terrain', transform=ccrs.PlateCarree(),
                   zorder=1, alpha=0.9, extend='max')

ax_c.imshow(hillshade, origin='upper',
            extent=[lon_min_dem, lon_max_dem, lat_min_dem, lat_max_dem],
            cmap='gray', alpha=0.30,
            transform=ccrs.PlateCarree(), zorder=2)

bbox = box(lon_min_dem, lat_min_dem, lon_max_dem, lat_max_dem)
inverse_mask = bbox.difference(basin_geom)
gpd.GeoSeries([inverse_mask], crs="EPSG:4326").plot(
    ax=ax_c, color='white', alpha=1.0,
    transform=ccrs.PlateCarree(), zorder=3)

if not others.empty:
    others.plot(ax=ax_c, color='dodgerblue', linewidth=1.4, alpha=0.7,
                transform=ccrs.PlateCarree(), zorder=4)
if not shweli.empty:
    shweli.plot(ax=ax_c, color='crimson', linewidth=3.2,
                transform=ccrs.PlateCarree(), zorder=5)
if not irrawaddy.empty:
    irrawaddy.plot(ax=ax_c, color='darkorange', linewidth=3.2,
                   transform=ccrs.PlateCarree(), zorder=5)

stations.plot(ax=ax_c, color='red', marker='^', markersize=130,
              edgecolor='black', linewidth=1.2,
              transform=ccrs.PlateCarree(), zorder=6)
add_station_labels(ax_c, zorder=7, fontsize=11)

basin.plot(ax=ax_c, facecolor='none', edgecolor='navy', linewidth=2.6,
           transform=ccrs.PlateCarree(), zorder=8)

add_border_clip(ax_c, zorder=8)
add_country_labels(ax_c, zorder=9)

ax_c.set_title('(c) Masked AW3D30 elevation (~90 m)',
               loc='left', pad=12, fontsize=18, fontweight='bold')
gl_c = ax_c.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                       alpha=0.5, linestyle=':')
gl_c.top_labels = False; gl_c.right_labels = False
gl_c.left_labels = False
gl_c.xlabel_style = {'size': 14, 'weight': 'bold'}
gl_c.ylabel_style = {'size': 14, 'weight': 'bold'}

# ==========================================================
# Colorbars — anchored dynamically to each panel
# ==========================================================
# ETOPO1 colorbar (panel a) — vertical, far right of top panel
pos_a = ax_a.get_position()
cax_a = fig.add_axes([pos_a.x1 + 0.006, pos_a.y0 + 0.02,
                      0.012, pos_a.height - 0.04])
cb_a  = fig.colorbar(im_a, cax=cax_a, orientation='vertical')
cb_a.set_label('ETOPO1 elevation (m)', fontsize=15, fontweight='bold', labelpad=10)
cb_a.ax.tick_params(labelsize=13)

# DEM colorbar (panel c) — vertical, far right of panel (c)
pos_c = ax_c.get_position()
cax_c = fig.add_axes([pos_c.x1 + 0.006, pos_c.y0 + 0.02,
                      0.012, pos_c.height - 0.04])
cb_c  = fig.colorbar(cf, cax=cax_c, orientation='vertical',
                     ticks=[0, 600, 1200, 1800, 2400, 3000, 3600, 4200])
cb_c.set_label('Elevation (m)', fontsize=15, fontweight='bold', labelpad=10)
cb_c.ax.tick_params(labelsize=13)

# ==========================================================
# QC note — centred at very bottom
# ==========================================================
fig.text(0.5, 0.012,
         f"Station network: 15 original → {n_retained} retained after QC.  "
         f"Excluded: {', '.join(excluded)} (Hkamti outside basin buffer; "
         f"no DMH record for threshold period).",
         ha='center', va='bottom', fontsize=13, color='#333',
         bbox=dict(boxstyle="round,pad=0.40", facecolor="#FFF8E7",
                   edgecolor="#B08000", linewidth=1.0, alpha=0.95))

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig01_Basin_Map_v2.png"
out_pdf = FIG_DIR / "Fig01_Basin_Map_v2.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")
plt.show()
```

    ✅ 14 stations retained; excluded: ['Hkamti']
    Loading AW3D30 DEM...
    Loading ETOPO1...
    Study extent: lon 94.76–100.33, lat 20.75–26.60
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig01_Basin_Map_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig01_Basin_Map_v2.pdf
    


    
![png](output_9_1.png)
    



```python

```


```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 — Data availability and quality control for the revised analysis
=========================================================================
Panels:
  (a) Daily basin-average rainfall time series (missing periods shaded;
      documented flood years shaded red)
  (b) Annual total rainfall with completeness flags and flood-year markers
  (c) Basin-mean daily soil moisture with ML-derived SM threshold
  (d) Multi-source data availability timeline

Typography: all fonts enlarged (base 15, titles 19, ticks 15, legends 12.5)

Reads:
    prep_revised_v2/fig02/master_daily.csv
    prep_revised_v2/fig04/fig04_annual_indices.csv
    prep_revised_v2/thresholds/final_operational_thresholds.json

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig02_Data_Availability_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# ---- Paths ----
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load prep outputs ----
master = pd.read_csv(PREP/"fig02"/"master_daily.csv",
                     parse_dates=["date"], index_col="date")
annual = pd.read_csv(PREP/"fig04"/"fig04_annual_indices.csv")
thr    = json.load(open(PREP/"thresholds"/"final_operational_thresholds.json"))
T_SM   = float(thr["SM"])

FLOOD_YEARS = [1991,1998,2004,2010,2015,2018,2019,2020,2023,2024]

# ---- Completeness per year from master ----
annual = annual.merge(
    master.groupby("year")["P_stn"].apply(lambda s: s.notna().sum())
          .rename("valid_days").reset_index(),
    on="year", how="left")
annual["complete"] = (annual["valid_days"] >= 360).astype(int)
n_incomplete = int((annual["complete"] == 0).sum())

# ---- Publication style — LARGE typography ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 7, 'ytick.major.size': 7,
    'legend.fontsize': 13, 'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

fig = plt.figure(figsize=(22, 16), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.85],
                       hspace=0.50, wspace=0.22,
                       left=0.065, right=0.975, top=0.955, bottom=0.06)

# ==========================================================
# (a) Daily rainfall time series
# ==========================================================
ax_a = fig.add_subplot(gs[0, :])

# Flood-year shading (drawn first, behind line)
for yr in FLOOD_YEARS:
    ax_a.axvspan(pd.Timestamp(f"{yr}-01-01"),
                 pd.Timestamp(f"{yr}-12-31"),
                 color="#d32f2f", alpha=0.08, zorder=0)

# Missing-period shading
p = master["P_stn"]
is_nan = p.isna()
if is_nan.any():
    idx = np.where(is_nan.values)[0]
    blocks, s, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k != prev + 1:
            blocks.append((s, prev)); s = k
        prev = k
    blocks.append((s, prev))
    for s_, e_ in blocks:
        ax_a.axvspan(p.index[s_], p.index[e_], color="#bdbdbd",
                     alpha=0.55, zorder=1)

# Main line
ax_a.plot(p.index, p.values, color="#2e7d32", lw=0.55, alpha=0.9, zorder=2)

# P95 envelope
wet = p[p >= 1.0]
p95 = float(wet.quantile(0.95))
ax_a.axhline(p95, color="#1b5e20", ls="--", lw=1.6, alpha=0.8, zorder=3)

ax_a.set_xlim(pd.Timestamp("1981-01-01"), pd.Timestamp("2025-12-31"))
ax_a.set_ylim(0, max(p.max()*1.08, 85))
ax_a.set_ylabel("Daily rainfall (mm)", fontweight="bold")
ax_a.set_title("(a) Daily basin-average rainfall, 1981–2025 "
               "(red bands = documented flood years; grey = missing days)",
               loc="left", pad=12)
ax_a.grid(axis="y", ls="--", alpha=0.35)

legend_a = [
    Line2D([0], [0], color="#2e7d32", lw=2.0, label="Daily basin-mean rainfall"),
    Line2D([0], [0], color="#1b5e20", ls="--", lw=1.8,
           label=f"Wet-day P95 = {p95:.1f} mm"),
    Patch(facecolor="#d32f2f", alpha=0.20, label="Documented flood year"),
    Patch(facecolor="#bdbdbd", alpha=0.55, label="Missing data period"),
]
ax_a.legend(handles=legend_a, loc="upper left", ncol=4,
            fontsize=12.5, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (b) Annual total rainfall with completeness
# ==========================================================
ax_b = fig.add_subplot(gs[1, 0])

rain = annual["rain_total"].values
colors_b = ["#81c784" if c else "#bdbdbd" for c in annual["complete"]]
edge_b   = ["#2e7d32" if c else "#757575" for c in annual["complete"]]
bars = ax_b.bar(annual["year"], rain, color=colors_b,
                edgecolor=edge_b, linewidth=0.9, width=0.8, zorder=2)

# Flood-year red outline
for i, is_flood in enumerate(annual["is_flood_year"]):
    if is_flood:
        bars[i].set_edgecolor("#d32f2f")
        bars[i].set_linewidth(2.2)

# Mean line
mean_total = float(np.nanmean(rain))
ax_b.axhline(mean_total, color="black", ls="--", lw=1.8, zorder=3,
             label=f"Long-term mean = {mean_total:.0f} mm")

# Asterisk flag for incomplete years
for yr, comp, val in zip(annual["year"], annual["complete"], rain):
    if not comp and np.isfinite(val):
        ax_b.text(yr, val + 40, "*", ha="center", va="bottom",
                  fontsize=18, color="#c62828", fontweight="bold")

ax_b.set_xlim(1979, 2027)
ax_b.set_ylim(0, np.nanmax(rain) * 1.16)
ax_b.set_ylabel("Annual total (mm)")
ax_b.set_xlabel("Year")
ax_b.set_title("(b) Annual total rainfall (basin-average)", loc="left", pad=12)
ax_b.grid(axis="y", ls="--", alpha=0.35)

legend_b = [
    Patch(facecolor="#81c784", edgecolor="#2e7d32", label="Complete year"),
    Patch(facecolor="#bdbdbd", edgecolor="#757575", label="Incomplete year (*)"),
    Patch(facecolor="none", edgecolor="#d32f2f", linewidth=2.2,
          label="Documented flood year"),
    Line2D([0], [0], color="black", ls="--", lw=1.8,
           label=f"Long-term mean = {mean_total:.0f} mm"),
]
ax_b.legend(handles=legend_b, loc="lower right",
            fontsize=11.5, ncol=2, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (c) Daily soil moisture + ML threshold
# ==========================================================
ax_c = fig.add_subplot(gs[1, 1])

sm = master["SM"]

# JJAS shading (back)
for yr in range(1981, 2026):
    ax_c.axvspan(pd.Timestamp(f"{yr}-06-01"),
                 pd.Timestamp(f"{yr}-09-30"),
                 color="#a5d6a7", alpha=0.10, zorder=0)

# No-native-SM shading beyond 2024-12-31
sm_end = pd.Timestamp("2024-12-31")
ax_c.axvspan(sm_end, pd.Timestamp("2025-12-31"),
             color="#e0e0e0", alpha=0.85, zorder=1)

# SM time series
ax_c.plot(sm.index, sm.values, color="#5d4037", lw=0.70, alpha=0.95, zorder=2)

# ML-derived threshold
ax_c.axhline(T_SM, color="#d32f2f", ls="--", lw=2.2, zorder=3)

pct_above = float((sm >= T_SM).mean() * 100)

ax_c.set_xlim(pd.Timestamp("1981-01-01"), pd.Timestamp("2025-12-31"))
ax_c.set_ylim(sm.min()*0.95, sm.max()*1.05)
ax_c.set_ylabel("Soil moisture (m³/m³)")
ax_c.set_xlabel("Year")
ax_c.set_title("(c) Basin-mean daily soil moisture "
               "(green = JJAS; grey = no native data)",
               loc="left", pad=12)
ax_c.grid(axis="y", ls="--", alpha=0.35)

ax_c.text(pd.Timestamp("2025-03-01"), sm.max()*0.93,
          "no native\nSM (2025)",
          ha="center", va="top", fontsize=11.5, color="#555",
          bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                    edgecolor="#888", alpha=0.9))

legend_c = [
    Line2D([0], [0], color="#5d4037", lw=2.0, label="Basin-mean SM"),
    Line2D([0], [0], color="#d32f2f", ls="--", lw=2.2,
           label=f"ML SM threshold = {T_SM:.3f}"),
    Patch(facecolor="#a5d6a7", alpha=0.4, label="JJAS season"),
    Patch(facecolor="#e0e0e0", alpha=0.85, label="No native SM"),
]
ax_c.legend(handles=legend_c, loc="lower right",
            fontsize=11.5, ncol=2, framealpha=0.95, edgecolor='0.3')

ax_c.text(0.02, 0.97,
          f"{pct_above:.1f}% of days ≥ ML SM threshold",
          transform=ax_c.transAxes, va="top", ha="left",
          fontsize=12, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                    edgecolor="#666", alpha=0.9))

# ==========================================================
# (d) Multi-source data availability timeline
# ==========================================================
ax_d = fig.add_subplot(gs[2, :])

sources = [
    ("Station network (14 retained)",  "1981-01-01", "2025-12-31", "#2e7d32"),
    ("CHIRPS 0.05° daily rainfall",    "1981-01-01", "2025-12-31", "#1f77b4"),
    ("C3S satellite soil moisture",    "1981-01-01", "2024-12-31", "#8d6e63"),
    ("ERA5 / NCEP-NCAR reanalysis",    "1981-01-01", "2025-12-31", "#6a1b9a"),
    ("AW3D30 / ETOPO1 topography",     "1981-01-01", "2025-12-31", "#f9a825"),
]

for i, (label, s, e, color) in enumerate(sources):
    y = len(sources) - 1 - i
    start = pd.Timestamp(s); end = pd.Timestamp(e)
    ax_d.barh(y, (end - start).days, left=start, height=0.65,
              color=color, alpha=0.82, edgecolor="black",
              linewidth=1.1, zorder=2)
    ax_d.text(start + pd.Timedelta(days=60), y, label,
              va="center", ha="left", fontsize=13.5,
              fontweight="bold", color="white", zorder=3)

# SM 2025 gap indicator
ax_d.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"),
             color="#c62828", alpha=0.10, zorder=1)
ax_d.text(pd.Timestamp("2025-07-01"), -0.55,
          "SM gap", ha="center", va="top",
          fontsize=12, color="#c62828", fontweight="bold")

ax_d.set_yticks([])
ax_d.set_xlim(pd.Timestamp("1980-06-01"), pd.Timestamp("2026-06-30"))
ax_d.set_ylim(-0.7, len(sources) - 0.3)
ax_d.set_xlabel("Year")
ax_d.set_title("(d) Multi-source data availability "
               "(all sources overlap 1981–2024; SM extends only to 2024)",
               loc="left", pad=12)
ax_d.grid(axis="x", ls="--", alpha=0.35)
ax_d.set_axisbelow(True)

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig02_Data_Availability_v2.png"
out_pdf = FIG_DIR / "Fig02_Data_Availability_v2.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf,           bbox_inches="tight", facecolor="white")
print(f"✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")
print(f"   Incomplete years flagged with *: "
      f"{annual.loc[annual['complete']==0, 'year'].tolist()}")
plt.show()
```

    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig02_Data_Availability_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig02_Data_Availability_v2.pdf
       Incomplete years flagged with *: []
    


    
![png](output_12_1.png)
    



```python

```


```python

```


```python
# ==========================================================
# FIGURE 2 (with soil moisture): LARGE FONT VERSION
# ==========================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import cartopy.crs as ccrs
from cartopy.io import shapereader
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

# ----------------------------------------------------------
# 0. PUBLICATION STYLE (LARGE FONTS)
# ----------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,                       # ← increased
    'axes.titlesize': 16, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'axes.linewidth': 1.0,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 12, 'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# ----------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------
calculated_dir = r'G:\NUIST_YTU_Water\trend\Calculated_Data'
output_dir     = r'G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles'
os.makedirs(output_dir, exist_ok=True)

BASIN_SHP   = r"G:\NUIST_YTU_Water\output_shapefiles\custom_basin.shp"
RIVER_SHP   = r"G:\NUIST_YTU_Water\output_shapefiles\clipped_rivers.shp"
STATION_CSV = r"G:\NUIST_YTU_Water\data\rainfall\all_15_stations.csv"

# ----------------------------------------------------------
# 2. LOAD BASIN, RIVERS, STATIONS
# ----------------------------------------------------------
print("Loading shapefiles...")
custom_basin   = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
clipped_rivers = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")

minx, miny, maxx, maxy = custom_basin.total_bounds
pad = 0.1
extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
print(f"Map extent set to: {extent}")

basin_geom   = custom_basin.geometry.union_all()
basin_buffer = basin_geom.buffer(0.02)

df = pd.read_csv(STATION_CSV)
df = df[df["Station"].str.strip() != "Hkamti"].copy()
df["Station"] = df["Station"].str.strip()
stations = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326"
)

shweli    = clipped_rivers[clipped_rivers["Sub_Name"].astype(str).str.contains("Shweli",    case=False, na=False)]
irrawaddy = clipped_rivers[clipped_rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)]
others    = clipped_rivers[~clipped_rivers.index.isin(shweli.index) & ~clipped_rivers.index.isin(irrawaddy.index)]

borders_path = shapereader.natural_earth(resolution='50m', category='cultural', name='admin_0_boundary_lines_land')
borders_gdf  = gpd.read_file(borders_path).to_crs("EPSG:4326")
borders_clip = borders_gdf[borders_gdf.intersects(basin_buffer)].copy()
borders_clip["geometry"] = borders_clip.geometry.intersection(basin_buffer)
borders_clip = borders_clip[~borders_clip.is_empty]

# ----------------------------------------------------------
# 3. LOAD CLIMATOLOGY (rain, temp, SM)
# ----------------------------------------------------------
print("\nLoading NetCDF files...")
ds_rain    = xr.open_dataset(os.path.join(calculated_dir, 'Calculated_JJAS_Total_Rainfall.nc'))
ds_temp    = xr.open_dataset(os.path.join(calculated_dir, 'Calculated_JJAS_Mean_Temp.nc'))
ds_sm      = xr.open_dataset(os.path.join(calculated_dir, 'Calculated_JJAS_Mean_SoilMoisture.nc'))
df_monthly = pd.read_csv(os.path.join(calculated_dir, 'Calculated_Monthly_Climatology.csv'))

def extract_field(ds, var_name):
    da = ds[var_name]
    drop = [c for c in da.coords if c not in da.dims]
    if drop: da = da.drop_vars(drop, errors='ignore')
    lon_name = 'lon' if 'lon' in da.dims else ('x' if 'x' in da.dims else None)
    lat_name = 'lat' if 'lat' in da.dims else ('y' if 'y' in da.dims else None)
    for d in [d for d in da.dims if d not in (lon_name, lat_name)]:
        if da.sizes[d] == 1: da = da.isel({d: 0})
        else: da = da.mean(dim=d, skipna=True)
    da = da.transpose(lat_name, lon_name)
    lons = np.asarray(da[lon_name].values).ravel().astype('float64')
    lats = np.asarray(da[lat_name].values).ravel().astype('float64')
    Z    = np.asarray(da.values, dtype='float64')
    if Z.shape != (lats.size, lons.size):
        Z = Z.T if Z.shape == (lons.size, lats.size) else Z
    print(f"[{var_name}]  lats({lats.size}) × lons({lons.size})  Z{Z.shape}")
    return lons, lats, Z

lons_r, lats_r, Z_rain = extract_field(ds_rain, 'JJAS_Total_Rainfall')
lons_t, lats_t, Z_temp = extract_field(ds_temp, 'JJAS_Mean_Temp')
lons_s, lats_s, Z_sm   = extract_field(ds_sm,   'JJAS_Mean_SoilMoisture')

# ----------------------------------------------------------
# 4. SMOOTH REGRIDDING — 1-D Boolean Mask
# ----------------------------------------------------------
def smooth_regrid(lons, lats, Z, n=400):
    lons = np.asarray(lons, dtype='float64').ravel()
    lats = np.asarray(lats, dtype='float64').ravel()
    Z    = np.asarray(Z, dtype='float64').squeeze()
    if Z.shape != (lats.size, lons.size):
        if Z.shape == (lons.size, lats.size): Z = Z.T
        else: raise ValueError(f"Z{Z.shape} vs lat{lats.size}×lon{lons.size}")
    if lons[0] > lons[-1]: lons = lons[::-1]; Z = Z[:, ::-1]
    if lats[0] > lats[-1]: lats = lats[::-1]; Z = Z[::-1, :]
    mask_nan = ~np.isfinite(Z)
    if mask_nan.any():
        pts  = np.column_stack(np.meshgrid(lons, lats)).reshape(-1, 2)
        vals = Z.ravel()
        good = np.isfinite(vals)
        nn = NearestNDInterpolator(pts[good], vals[good])
        Z_fill = Z.copy().ravel()
        Z_fill[~good] = nn(pts[~good])
        Z = Z_fill.reshape(Z.shape)
    interp = RegularGridInterpolator((lats, lons), Z, bounds_error=False, fill_value=np.nan)
    fine_lon = np.linspace(extent[0], extent[1], n)
    fine_lat = np.linspace(extent[2], extent[3], n)
    LON, LAT = np.meshgrid(fine_lon, fine_lat)
    pts_fine = np.column_stack([LAT.ravel(), LON.ravel()])
    Z_fine   = interp(pts_fine).reshape(LAT.shape)
    return fine_lon, fine_lat, Z_fine

lon_r_fine, lat_r_fine, rain_fine = smooth_regrid(lons_r, lats_r, Z_rain)
lon_t_fine, lat_t_fine, temp_fine = smooth_regrid(lons_t, lats_t, Z_temp)
lon_s_fine, lat_s_fine, sm_fine   = smooth_regrid(lons_s, lats_s, Z_sm)

# ----------------------------------------------------------
# 5. FIXED CARTOPY MASKING
# ----------------------------------------------------------
def mask_contour_by_polygon(ax, contour, polygon):
    if polygon.geom_type == 'MultiPolygon':
        vertices, codes = [], []
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            vertices.extend(zip(x, y))
            codes.extend([Path.MOVETO] + [Path.LINETO]*(len(x)-2) + [Path.CLOSEPOLY])
    else:
        x, y = polygon.exterior.xy
        vertices = list(zip(x, y))
        codes = [Path.MOVETO] + [Path.LINETO]*(len(x)-2) + [Path.CLOSEPOLY]
    path  = Path(vertices, codes)
    trans = contour.get_transform()
    patch = PathPatch(path, transform=trans, facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    contour.set_clip_path(patch)

# ----------------------------------------------------------
# 6. OVERLAY HELPERS
# ----------------------------------------------------------
def add_border_clip(ax, zorder=8):
    if not borders_clip.empty:
        borders_clip.plot(ax=ax, linestyle='--', edgecolor='magenta',
                          linewidth=2.2, transform=ccrs.PlateCarree(), zorder=zorder)

def add_country_labels(ax, zorder=9):
    ax.text(maxx - 0.6, maxy - 0.6, 'CHINA', fontsize=18, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(), zorder=zorder,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.9, edgecolor='0.3', linewidth=1.0))
    ax.text(minx + 1.0, miny + 0.8, 'MYANMAR', fontsize=18, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(), zorder=zorder,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.9, edgecolor='0.3', linewidth=1.0))

def add_overlays(ax):
    if not others.empty:
        others.plot(ax=ax, color='dodgerblue', linewidth=1.5, alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=5)
    if not shweli.empty:
        shweli.plot(ax=ax, color='crimson', linewidth=3.5,
                    transform=ccrs.PlateCarree(), zorder=5.1)
    if not irrawaddy.empty:
        irrawaddy.plot(ax=ax, color='darkorange', linewidth=3.5,
                       transform=ccrs.PlateCarree(), zorder=5.1)
    custom_basin.plot(ax=ax, facecolor='none', edgecolor='navy',
                      linewidth=2.5, transform=ccrs.PlateCarree(), zorder=5.5)
    stations.plot(ax=ax, color='red', marker='^', markersize=90,
                  edgecolor='black', linewidth=1.0,
                  transform=ccrs.PlateCarree(), zorder=6)
    add_border_clip(ax, zorder=7)
    add_country_labels(ax, zorder=8)

def style_geo_axes(ax, title, show_left_labels=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, pad=12, fontsize=16, fontweight='bold')
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='0.4',
                      alpha=0.5, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.xlabel_style = {'size': 13, 'weight': 'bold'}
    gl.ylabel_style = {'size': 13, 'weight': 'bold'}
    for s in ax.spines.values():
        s.set_linewidth(1.0); s.set_edgecolor('0.2')

# ----------------------------------------------------------
# 7. FIGURE LAYOUT (2×3, LARGE FONT)
# ----------------------------------------------------------
fig = plt.figure(figsize=(26, 14))                    # ← bigger canvas
gs  = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0],
                       hspace=0.35, wspace=0.22,
                       left=0.05, right=0.97, top=0.94, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[1, :])

# ---------------- Panel (a): JJAS rainfall ----------------
rain_levels = np.linspace(np.nanmin(rain_fine), np.nanmax(rain_fine), 21)
rain_cmap   = plt.get_cmap('YlGnBu')
rain_norm   = BoundaryNorm(rain_levels, ncolors=rain_cmap.N, clip=True)
cf1 = ax1.contourf(lon_r_fine, lat_r_fine, rain_fine, levels=rain_levels,
                   cmap=rain_cmap, norm=rain_norm, extend='both',
                   transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax1, cf1, basin_geom)
add_overlays(ax1)
style_geo_axes(ax1, '(a) Total monsoon (JJAS) rainfall', show_left_labels=True)
cb1 = fig.colorbar(cf1, ax=ax1, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb1.set_label('Total rainfall (mm)', fontsize=14, fontweight='bold')
cb1.ax.tick_params(labelsize=12, width=1.0)
cb1.outline.set_linewidth(0.8)

# ---------------- Panel (b): JJAS temperature ----------------
temp_levels = np.linspace(np.nanmin(temp_fine), np.nanmax(temp_fine), 21)
temp_cmap   = plt.get_cmap('YlOrRd')
temp_norm   = BoundaryNorm(temp_levels, ncolors=temp_cmap.N, clip=True)
cf2 = ax2.contourf(lon_t_fine, lat_t_fine, temp_fine, levels=temp_levels,
                   cmap=temp_cmap, norm=temp_norm, extend='both',
                   transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax2, cf2, basin_geom)
add_overlays(ax2)
style_geo_axes(ax2, '(b) Mean monsoon (JJAS) temperature', show_left_labels=False)
cb2 = fig.colorbar(cf2, ax=ax2, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb2.set_label('Temperature (°C)', fontsize=14, fontweight='bold')
cb2.ax.tick_params(labelsize=12, width=1.0)
cb2.outline.set_linewidth(0.8)

# ---------------- Panel (c): JJAS soil moisture ----------------
sm_levels = np.linspace(np.nanmin(sm_fine), np.nanmax(sm_fine), 21)
sm_cmap   = plt.get_cmap('YlGn')
sm_norm   = BoundaryNorm(sm_levels, ncolors=sm_cmap.N, clip=True)
cf3 = ax3.contourf(lon_s_fine, lat_s_fine, sm_fine, levels=sm_levels,
                   cmap=sm_cmap, norm=sm_norm, extend='both',
                   transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax3, cf3, basin_geom)
add_overlays(ax3)
style_geo_axes(ax3, '(c) Mean monsoon (JJAS) soil moisture', show_left_labels=False)
cb3 = fig.colorbar(cf3, ax=ax3, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb3.set_label('Volumetric soil moisture (m³ m⁻³)', fontsize=14, fontweight='bold')
cb3.ax.tick_params(labelsize=12, width=1.0)
cb3.outline.set_linewidth(0.8)

# Shared legend
legend_elements = [
    Line2D([0], [0], color='navy',       lw=2.5, label='Basin boundary'),
    Line2D([0], [0], color='magenta',    lw=2.2, ls='--', label='Myanmar–China border (clipped)'),
    Line2D([0], [0], color='crimson',    lw=3.5, label='Shweli River'),
    Line2D([0], [0], color='darkorange', lw=3.5, label='Irrawaddy River'),
    Line2D([0], [0], color='dodgerblue', lw=1.5, label='Tributaries'),
    Line2D([0], [0], marker='^', color='red', markeredgecolor='black',
           ls='None', markersize=12, label='Stations (n = 14)'),
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=11,
           frameon=True, framealpha=0.95, edgecolor='0.3',
           borderpad=0.6, handlelength=2.2, labelspacing=0.4)

# ---------------- Panel (d): Monthly climatology ----------------
months       = df_monthly['Month']
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax4.axvspan(5.5, 9.5, color='0.85', alpha=0.35, zorder=0, label='JJAS monsoon')

bar_color = '#2166ac'
ax4.bar(months, df_monthly['Rainfall_mm'], color=bar_color, alpha=0.9,
        width=0.65, edgecolor='0.2', linewidth=0.8,
        label='Mean rainfall', zorder=2)
ax4.set_ylabel('Rainfall (mm)', fontsize=14, fontweight='bold', color=bar_color)
ax4.tick_params(axis='y', labelcolor=bar_color, labelsize=13, width=1.0)
ax4.set_xticks(months); ax4.set_xticklabels(month_labels, fontsize=13, fontweight='bold')
ax4.set_xlabel('Month', fontsize=15, fontweight='bold'); ax4.set_ylim(bottom=0)
ax4.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.55)
ax4.set_axisbelow(True); ax4.spines['top'].set_visible(False)

ax4_t = ax4.twinx()
temp_color = '#b2182b'
ax4_t.plot(months, df_monthly['Temperature_C'], color=temp_color,
           linewidth=2.8, marker='o', markersize=7,
           markeredgecolor='black', markeredgewidth=0.6,
           label='Mean temperature', zorder=3)
ax4_t.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold', color=temp_color)
ax4_t.tick_params(axis='y', labelcolor=temp_color, labelsize=13, width=1.0)
ax4_t.spines['top'].set_visible(False)

ax4_s = ax4.twinx()
ax4_s.spines['right'].set_position(('outward', 70))
sm_color = '#1b7837'
ax4_s.plot(months, df_monthly['SoilMoisture_m3m3'], color=sm_color,
           linewidth=2.8, marker='D', markersize=6,
           markeredgecolor='black', markeredgewidth=0.5,
           label='Mean soil moisture', zorder=4)
ax4_s.set_ylabel('Soil moisture (m³ m⁻³)', fontsize=14, fontweight='bold', color=sm_color)
ax4_s.tick_params(axis='y', labelcolor=sm_color, labelsize=13, width=1.0)
ax4_s.spines['top'].set_visible(False)

h1, l1 = ax4.get_legend_handles_labels()
h2, l2 = ax4_t.get_legend_handles_labels()
h3, l3 = ax4_s.get_legend_handles_labels()
ax4.legend(h1 + h2 + h3, l1 + l2 + l3, loc='upper left',
           fontsize=12, frameon=True, framealpha=0.95,
           edgecolor='0.3', borderpad=0.6)
ax4.set_title('(d) Monthly climatology (January–December)', pad=12, fontsize=16, fontweight='bold')

# ----------------------------------------------------------
# 8. SAVE
# ----------------------------------------------------------
output_fig = os.path.join(output_dir, 'Figure2_Spatial_Climatology_withSM.png')
plt.savefig(output_fig, dpi=600, bbox_inches='tight')
print(f"✅ Figure saved to: {output_fig}")
plt.show()
```

    Loading shapefiles...
    Map extent set to: [np.float64(94.76249999999959), np.float64(100.32916666666631), np.float64(20.754166666666357), np.float64(26.6)]
    
    Loading NetCDF files...
    [JJAS_Total_Rainfall]  lats(140) × lons(300)  Z(140, 300)
    [JJAS_Mean_Temp]  lats(29) × lons(61)  Z(29, 61)
    [JJAS_Mean_SoilMoisture]  lats(28) × lons(60)  Z(28, 60)
    ✅ Figure saved to: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Figure2_Spatial_Climatology_withSM.png
    


    
![png](output_15_1.png)
    



```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 3 — Monsoon climatology and pre-conditioning, 1981–2025
=========================================================================
Combined layout:
  Spatial (from Script 1, smoothed + basin-masked contourf):
    (a) JJAS total rainfall
    (b) JJAS mean temperature
    (c) JJAS mean soil moisture
  Temporal (from Script 2, large-font matplotlib):
    (d) Monthly rainfall + temperature climatology
    (e) Monthly soil-moisture climatology
    (f) Seasonal SM distribution (DJF / MAM / JJAS / ON)

Output:
    CMIP6_Evaluation/output_shapefiles/Fig03_Monsoon_Climatology_combined.png/.pdf
"""
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
SHP_DIR = ROOT / "output_shapefiles"
CALC    = ROOT / "trend" / "Calculated_Data"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASIN_SHP   = SHP_DIR / "custom_basin.shp"
RIVER_SHP   = SHP_DIR / "clipped_rivers.shp"
STATION_CSV = ROOT / "data" / "rainfall" / "all_15_stations.csv"
RAIN_NC     = CALC / "Calculated_JJAS_Total_Rainfall.nc"
TEMP_NC     = CALC / "Calculated_JJAS_Mean_Temp.nc"
SM_NC       = CALC / "Calculated_JJAS_Mean_SoilMoisture.nc"
MCLIM_CSV   = CALC / "Calculated_Monthly_Climatology.csv"
MASTER      = PREP / "fig02" / "master_daily.csv"
THR_JSON    = PREP / "thresholds" / "final_operational_thresholds.json"

# -----------------------------------------------------------------------------
# Style — LARGE typography (from Script 1)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 15,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.1,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 12.5, 'legend.frameon': True, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# 1. Load basin, rivers, stations
# -----------------------------------------------------------------------------
print("Loading shapefiles...")
basin  = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
rivers = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")

shweli    = rivers[rivers["Sub_Name"].astype(str).str.contains("Shweli",    case=False, na=False)]
irrawaddy = rivers[rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)]
others    = rivers[~rivers.index.isin(shweli.index) & ~rivers.index.isin(irrawaddy.index)]

minx, miny, maxx, maxy = basin.total_bounds
pad = 0.1
EXTENT = [minx - pad, maxx + pad, miny - pad, maxy + pad]

basin_geom   = basin.geometry.union_all()
basin_buffer = basin_geom.buffer(0.02)

# Stations — drop Hkamti
stn = pd.read_csv(STATION_CSV)
stn["Station"] = stn["Station"].astype(str).str.strip()
stn = stn[stn["Station"] != "Hkamti"].copy()
stations = gpd.GeoDataFrame(
    stn, geometry=gpd.points_from_xy(stn["Lon"], stn["Lat"]), crs="EPSG:4326")
print(f"✅ {len(stations)} stations retained (Hkamti excluded)")

# Country borders — clipped to basin buffer
borders_path = shapereader.natural_earth(
    resolution='50m', category='cultural', name='admin_0_boundary_lines_land')
borders_gdf  = gpd.read_file(borders_path).to_crs("EPSG:4326")
borders_clip = borders_gdf[borders_gdf.intersects(basin_buffer)].copy()
borders_clip["geometry"] = borders_clip.geometry.intersection(basin_buffer)
borders_clip = borders_clip[~borders_clip.is_empty]

# -----------------------------------------------------------------------------
# 2. Load climatology NetCDFs
# -----------------------------------------------------------------------------
print("\nLoading climatology NetCDFs...")
ds_rain = xr.open_dataset(RAIN_NC)
ds_temp = xr.open_dataset(TEMP_NC)
ds_sm   = xr.open_dataset(SM_NC)

def extract_field(ds, var_name):
    da = ds[var_name]
    drop = [c for c in da.coords if c not in da.dims]
    if drop: da = da.drop_vars(drop, errors='ignore')
    lon_name = 'lon' if 'lon' in da.dims else ('x' if 'x' in da.dims else None)
    lat_name = 'lat' if 'lat' in da.dims else ('y' if 'y' in da.dims else None)
    for d in [d for d in da.dims if d not in (lon_name, lat_name)]:
        if da.sizes[d] == 1: da = da.isel({d: 0})
        else: da = da.mean(dim=d, skipna=True)
    da = da.transpose(lat_name, lon_name)
    lons = np.asarray(da[lon_name].values).ravel().astype('float64')
    lats = np.asarray(da[lat_name].values).ravel().astype('float64')
    Z    = np.asarray(da.values, dtype='float64')
    if Z.shape != (lats.size, lons.size):
        Z = Z.T if Z.shape == (lons.size, lats.size) else Z
    print(f"[{var_name}]  lats({lats.size}) × lons({lons.size})  Z{Z.shape}")
    return lons, lats, Z

lons_r, lats_r, Z_rain = extract_field(ds_rain, 'JJAS_Total_Rainfall')
lons_t, lats_t, Z_temp = extract_field(ds_temp, 'JJAS_Mean_Temp')
lons_s, lats_s, Z_sm   = extract_field(ds_sm,   'JJAS_Mean_SoilMoisture')

# -----------------------------------------------------------------------------
# 3. Smooth regrid
# -----------------------------------------------------------------------------
def smooth_regrid(lons, lats, Z, n=400):
    lons = np.asarray(lons, dtype='float64').ravel()
    lats = np.asarray(lats, dtype='float64').ravel()
    Z    = np.asarray(Z, dtype='float64').squeeze()
    if Z.shape != (lats.size, lons.size):
        if Z.shape == (lons.size, lats.size): Z = Z.T
        else: raise ValueError(f"Z{Z.shape} vs lat{lats.size}×lon{lons.size}")
    if lons[0] > lons[-1]: lons = lons[::-1]; Z = Z[:, ::-1]
    if lats[0] > lats[-1]: lats = lats[::-1]; Z = Z[::-1, :]
    mask_nan = ~np.isfinite(Z)
    if mask_nan.any():
        pts  = np.column_stack(np.meshgrid(lons, lats)).reshape(-1, 2)
        vals = Z.ravel()
        good = np.isfinite(vals)
        nn = NearestNDInterpolator(pts[good], vals[good])
        Z_fill = Z.copy().ravel()
        Z_fill[~good] = nn(pts[~good])
        Z = Z_fill.reshape(Z.shape)
    interp = RegularGridInterpolator((lats, lons), Z, bounds_error=False, fill_value=np.nan)
    fine_lon = np.linspace(EXTENT[0], EXTENT[1], n)
    fine_lat = np.linspace(EXTENT[2], EXTENT[3], n)
    LON, LAT = np.meshgrid(fine_lon, fine_lat)
    pts_fine = np.column_stack([LAT.ravel(), LON.ravel()])
    Z_fine   = interp(pts_fine).reshape(LAT.shape)
    return fine_lon, fine_lat, Z_fine

lon_r_f, lat_r_f, rain_f = smooth_regrid(lons_r, lats_r, Z_rain)
lon_t_f, lat_t_f, temp_f = smooth_regrid(lons_t, lats_t, Z_temp)
lon_s_f, lat_s_f, sm_f   = smooth_regrid(lons_s, lats_s, Z_sm)

# -----------------------------------------------------------------------------
# 4. Basin mask helper (from Script 1)
# -----------------------------------------------------------------------------
def mask_contour_by_polygon(ax, contour, polygon):
    if polygon.geom_type == 'MultiPolygon':
        vertices, codes = [], []
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            vertices.extend(zip(x, y))
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(x)-2) + [MplPath.CLOSEPOLY])
    else:
        x, y = polygon.exterior.xy
        vertices = list(zip(x, y))
        codes = [MplPath.MOVETO] + [MplPath.LINETO]*(len(x)-2) + [MplPath.CLOSEPOLY]
    path  = MplPath(vertices, codes)
    trans = contour.get_transform()
    patch = PathPatch(path, transform=trans, facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    contour.set_clip_path(patch)

# -----------------------------------------------------------------------------
# 5. Overlay helpers (from Script 1)
# -----------------------------------------------------------------------------
def add_border_clip(ax, zorder=8):
    if not borders_clip.empty:
        borders_clip.plot(ax=ax, linestyle='--', edgecolor='magenta',
                          linewidth=2.4, transform=ccrs.PlateCarree(), zorder=zorder)

def add_country_labels(ax, zorder=9):
    ax.text(maxx - 0.6, maxy - 0.6, 'CHINA', fontsize=17, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(), zorder=zorder,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.9, edgecolor='0.3', linewidth=1.0))
    ax.text(minx + 1.0, miny + 0.8, 'MYANMAR', fontsize=17, fontweight='bold',
            color='black', transform=ccrs.PlateCarree(), zorder=zorder,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.9, edgecolor='0.3', linewidth=1.0))

def add_overlays(ax):
    if not others.empty:
        others.plot(ax=ax, color='dodgerblue', linewidth=1.6, alpha=0.7,
                    transform=ccrs.PlateCarree(), zorder=5)
    if not shweli.empty:
        shweli.plot(ax=ax, color='crimson', linewidth=3.6,
                    transform=ccrs.PlateCarree(), zorder=5.1)
    if not irrawaddy.empty:
        irrawaddy.plot(ax=ax, color='darkorange', linewidth=3.6,
                       transform=ccrs.PlateCarree(), zorder=5.1)
    basin.plot(ax=ax, facecolor='none', edgecolor='navy',
               linewidth=2.6, transform=ccrs.PlateCarree(), zorder=5.5)
    stations.plot(ax=ax, color='red', marker='^', markersize=100,
                  edgecolor='black', linewidth=1.1,
                  transform=ccrs.PlateCarree(), zorder=6)
    add_border_clip(ax, zorder=7)
    add_country_labels(ax, zorder=8)

def style_geo_axes(ax, title, show_left_labels=True):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title, loc='left', pad=12, fontsize=18, fontweight='bold')
    gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='0.4',
                      alpha=0.5, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.xlabel_style = {'size': 14, 'weight': 'bold'}
    gl.ylabel_style = {'size': 14, 'weight': 'bold'}
    for s in ax.spines.values():
        s.set_linewidth(1.1); s.set_edgecolor('0.2')

# -----------------------------------------------------------------------------
# 6. Load monthly climatology + master (for temporal panels)
# -----------------------------------------------------------------------------
print("\nLoading monthly climatology and master...")
mclim = pd.read_csv(MCLIM_CSV) if MCLIM_CSV.exists() else None
master = pd.read_csv(MASTER, parse_dates=["date"], index_col="date")
master["month"] = master.index.month

sm_mon = (master.loc[master.index.year <= 2024]
          .groupby("month")["SM"].agg(["mean", "std", "min", "max"]))
sm_mon = sm_mon.reindex(range(1, 13))

if mclim is not None and len(mclim) == 12:
    rain_m = mclim["Rainfall_mm"].values
    temp_m = mclim["Temperature_C"].values
else:
    rain_m = master.groupby("month")["P_stn"].mean().reindex(range(1, 13)).values
    temp_m = np.full(12, np.nan)

months = np.arange(1, 13)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# Seasonal SM
def season_of(m):
    if m in [12, 1, 2]:  return "DJF"
    if m in [3, 4, 5]:   return "MAM"
    if m in [6, 7, 8, 9]:return "JJAS"
    return "ON"

master["season"] = master.index.month.map(season_of)
season_order = ["DJF", "MAM", "JJAS", "ON"]
season_colors = {"DJF": "#90caf9", "MAM": "#f9a825",
                 "JJAS": "#2e7d32", "ON": "#8d6e63"}
season_data = {s: master.loc[master["season"] == s, "SM"].dropna().values
               for s in season_order}

# ==========================================================
# 7. FIGURE — 2 rows × 3 cols, LARGE
# ==========================================================
fig = plt.figure(figsize=(26, 15),dpi=300)
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.55, 1.0],
    hspace=0.38, wspace=0.22,
    left=0.045, right=0.975, top=0.94, bottom=0.075,
)

# ==========================================================
# Row 1 — Spatial maps (a, b, c) from Script 1
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())

# ---- (a) JJAS rainfall ----
rain_levels = np.linspace(np.nanmin(rain_f), np.nanmax(rain_f), 21)
rain_cmap   = plt.get_cmap('YlGnBu')
rain_norm   = BoundaryNorm(rain_levels, ncolors=rain_cmap.N, clip=True)
cf1 = ax_a.contourf(lon_r_f, lat_r_f, rain_f, levels=rain_levels,
                    cmap=rain_cmap, norm=rain_norm, extend='both',
                    transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax_a, cf1, basin_geom)
add_overlays(ax_a)
style_geo_axes(ax_a, '(a) Total monsoon (JJAS) rainfall', show_left_labels=True)
cb1 = fig.colorbar(cf1, ax=ax_a, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb1.set_label('Total rainfall (mm)', fontsize=15, fontweight='bold')
cb1.ax.tick_params(labelsize=13, width=1.0)
cb1.outline.set_linewidth(0.9)

# ---- (b) JJAS temperature ----
temp_levels = np.linspace(np.nanmin(temp_f), np.nanmax(temp_f), 21)
temp_cmap   = plt.get_cmap('YlOrRd')
temp_norm   = BoundaryNorm(temp_levels, ncolors=temp_cmap.N, clip=True)
cf2 = ax_b.contourf(lon_t_f, lat_t_f, temp_f, levels=temp_levels,
                    cmap=temp_cmap, norm=temp_norm, extend='both',
                    transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax_b, cf2, basin_geom)
add_overlays(ax_b)
style_geo_axes(ax_b, '(b) Mean monsoon (JJAS) temperature', show_left_labels=False)
cb2 = fig.colorbar(cf2, ax=ax_b, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb2.set_label('Temperature (°C)', fontsize=15, fontweight='bold')
cb2.ax.tick_params(labelsize=13, width=1.0)
cb2.outline.set_linewidth(0.9)

# ---- (c) JJAS soil moisture ----
sm_levels = np.linspace(np.nanmin(sm_f), np.nanmax(sm_f), 21)
sm_cmap   = plt.get_cmap('YlGn')
sm_norm   = BoundaryNorm(sm_levels, ncolors=sm_cmap.N, clip=True)
cf3 = ax_c.contourf(lon_s_f, lat_s_f, sm_f, levels=sm_levels,
                    cmap=sm_cmap, norm=sm_norm, extend='both',
                    transform=ccrs.PlateCarree(), zorder=1)
mask_contour_by_polygon(ax_c, cf3, basin_geom)
add_overlays(ax_c)
style_geo_axes(ax_c, '(c) Mean monsoon (JJAS) soil moisture', show_left_labels=False)
cb3 = fig.colorbar(cf3, ax=ax_c, orientation='vertical',
                   fraction=0.046, pad=0.03, shrink=0.95)
cb3.set_label(r'Volumetric soil moisture (m$^3$ m$^{-3}$)',
              fontsize=15, fontweight='bold')
cb3.ax.tick_params(labelsize=13, width=1.0)
cb3.outline.set_linewidth(0.9)

# Shared spatial legend — anchored to (c)
legend_spatial = [
    Line2D([0], [0], color='navy',       lw=2.6, label='Basin boundary'),
    Line2D([0], [0], color='magenta',    lw=2.4, ls='--',
           label='China–Myanmar border (clipped)'),
    Line2D([0], [0], color='crimson',    lw=3.6, label='Shweli River'),
    Line2D([0], [0], color='darkorange', lw=3.6, label='Ayeyarwady River'),
    Line2D([0], [0], color='dodgerblue', lw=1.6, label='Tributaries'),
    Line2D([0], [0], marker='^', color='red', markeredgecolor='black',
           ls='None', markersize=13, label=f'Stations (n = {len(stations)})'),
]
ax_c.legend(handles=legend_spatial, loc='lower right', fontsize=12,
            frameon=True, framealpha=0.95, edgecolor='0.3',
            borderpad=0.6, handlelength=2.2, labelspacing=0.4)

# ==========================================================
# Row 2 — Temporal panels (d, e, f) from Script 2
# ==========================================================
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

# ---- (d) Monthly rain + temperature ----
ax_d.axvspan(5.5, 9.5, color='#e0e0e0', alpha=0.35, zorder=0,
             label='JJAS monsoon')

rain_color = "#2e7d32"
ax_d.bar(months, rain_m, color=rain_color, alpha=0.85,
         edgecolor='black', linewidth=0.8, width=0.65,
         label=r'Mean rainfall (mm month$^{-1}$)', zorder=2)
ax_d.set_xlabel('Month', fontsize=16, fontweight='bold')
ax_d.set_ylabel(r'Rainfall (mm month$^{-1}$)',
                color=rain_color, fontweight='bold', fontsize=16)
ax_d.tick_params(axis='y', labelcolor=rain_color, labelsize=14)
ax_d.set_xticks(months)
ax_d.set_xticklabels(month_labels, fontsize=14, fontweight='bold')
ax_d.set_ylim(bottom=0)
ax_d.grid(axis='y', ls='--', alpha=0.30, zorder=1)
ax_d.set_axisbelow(True)
ax_d.spines['top'].set_visible(False)

temp_color = "#c62828"
ax_d2 = ax_d.twinx()
if np.isfinite(temp_m).any():
    ax_d2.plot(months, temp_m, color=temp_color, lw=2.8,
               marker='o', markersize=7, markeredgecolor='black',
               markeredgewidth=0.7,
               label=r'Mean temperature ($^\circ$C)', zorder=3)
ax_d2.set_ylabel(r'Temperature ($^\circ$C)',
                 color=temp_color, fontweight='bold', fontsize=16)
ax_d2.tick_params(axis='y', labelcolor=temp_color, labelsize=14)
ax_d2.spines['top'].set_visible(False)

ax_d.set_title('(d) Monthly rainfall and temperature climatology',
               loc='left', pad=12, fontsize=18, fontweight='bold')

h1, l1 = ax_d.get_legend_handles_labels()
h2, l2 = ax_d2.get_legend_handles_labels()
ax_d.legend(h1 + h2, l1 + l2, loc='upper left',
            fontsize=12.5, framealpha=0.95, edgecolor='0.3')

# ---- (e) Monthly SM cycle ----
ax_e.axvspan(5.5, 9.5, color='#e0e0e0', alpha=0.35, zorder=0)

sm_mean = sm_mon["mean"].values
sm_std  = sm_mon["std"].values
sm_color = "#1565c0"

ax_e.fill_between(months, sm_mean - sm_std, sm_mean + sm_std,
                  color=sm_color, alpha=0.20, label='±1 SD', zorder=1)
ax_e.plot(months, sm_mean, color=sm_color, lw=2.8,
          marker='D', markersize=8, markeredgecolor='black',
          markeredgewidth=0.6, label='Mean SM', zorder=3)

try:
    thr = json.load(open(THR_JSON))
    T_SM = float(thr["SM"])
    ax_e.axhline(T_SM, color='#d32f2f', ls='--', lw=2.0, zorder=2,
                 label=f"ML SM threshold = {T_SM:.3f}")
except Exception:
    pass

ax_e.set_xlabel('Month', fontsize=16, fontweight='bold')
ax_e.set_ylabel(r'Soil moisture (m$^3$ m$^{-3}$)',
                fontsize=16, fontweight='bold')
ax_e.set_xticks(months)
ax_e.set_xticklabels(month_labels, fontsize=14, fontweight='bold')
ax_e.set_ylim(sm_mean.min() - 3*sm_std.max(),
              sm_mean.max() + 2*sm_std.max())
ax_e.grid(axis='y', ls='--', alpha=0.30)
ax_e.spines['top'].set_visible(False)
ax_e.set_title('(e) Monthly soil-moisture climatology (1981–2024)',
               loc='left', pad=12, fontsize=18, fontweight='bold')
ax_e.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# ---- (f) Seasonal SM distribution ----
bp = ax_f.boxplot(
    [season_data[s] for s in season_order],
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color='black', linewidth=2.2),
    whiskerprops=dict(linewidth=1.3),
    capprops=dict(linewidth=1.3),
    boxprops=dict(linewidth=1.1),
)
for patch, s in zip(bp["boxes"], season_order):
    patch.set_facecolor(season_colors[s])
    patch.set_alpha(0.75)

for i, s in enumerate(season_order, start=1):
    m = float(np.nanmean(season_data[s]))
    ax_f.scatter([i], [m], marker='D', s=110, color='black',
                 edgecolor='white', linewidth=1.1, zorder=5)

ax_f.set_xticks(range(1, len(season_order)+1))
ax_f.set_xticklabels([f"{s}\n(n={len(season_data[s])})"
                       for s in season_order], fontsize=14, fontweight='bold')
ax_f.set_ylabel(r'Soil moisture (m$^3$ m$^{-3}$)',
                fontsize=16, fontweight='bold')
ax_f.grid(axis='y', ls='--', alpha=0.30)
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)
ax_f.set_title('(f) Seasonal SM distribution (diamonds = mean)',
               loc='left', pad=12, fontsize=18, fontweight='bold')

ax_f.legend(handles=[
    Patch(facecolor=season_colors["DJF"], alpha=0.75, label="DJF (dry)"),
    Patch(facecolor=season_colors["MAM"], alpha=0.75, label="MAM (pre-monsoon)"),
    Patch(facecolor=season_colors["JJAS"], alpha=0.75, label="JJAS (monsoon)"),
    Patch(facecolor=season_colors["ON"],  alpha=0.75, label="ON (post-monsoon)"),
], loc='lower right', fontsize=11.5, framealpha=0.95, edgecolor='0.3', ncol=2)

# ==========================================================
# Suptitle
# ==========================================================
#fig.suptitle(
 #   "Figure 3: Monsoon climatology and pre-conditioning variables, "
#    "Shweli–Ayeyarwady basin (1981–2025)",
#    fontsize=19, fontweight='bold', y=0.975)

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig03_Monsoon_Climatology_combined.png"
out_pdf = FIG_DIR / "Fig03_Monsoon_Climatology_combined.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# Summary
print("\nClimatology summary:")
print(f"  JJAS rain : {float(np.nanmin(rain_f)):.0f} – {float(np.nanmax(rain_f)):.0f} mm")
print(f"  JJAS temp : {float(np.nanmin(temp_f)):.1f} – {float(np.nanmax(temp_f)):.1f} °C")
print(f"  JJAS SM   : {float(np.nanmin(sm_f)):.3f} – {float(np.nanmax(sm_f)):.3f} m³/m³")
print(f"  Monthly SM mean: {np.nanmean(sm_mean):.3f} | "
      f"peak month = {month_labels[int(np.nanargmax(sm_mean))]}")
for s in season_order:
    print(f"  {s:<5s} SM mean: {float(np.nanmean(season_data[s])):.3f} "
          f"(n = {len(season_data[s])})")

plt.show()
```

    Loading shapefiles...
    ✅ 14 stations retained (Hkamti excluded)
    
    Loading climatology NetCDFs...
    [JJAS_Total_Rainfall]  lats(140) × lons(300)  Z(140, 300)
    [JJAS_Mean_Temp]  lats(29) × lons(61)  Z(29, 61)
    [JJAS_Mean_SoilMoisture]  lats(28) × lons(60)  Z(28, 60)
    
    Loading monthly climatology and master...
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig03_Monsoon_Climatology_combined.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig03_Monsoon_Climatology_combined.pdf
    
    Climatology summary:
      JJAS rain : 405 – 2696 mm
      JJAS temp : 15.9 – 30.1 °C
      JJAS SM   : 0.205 – 0.986 m³/m³
      Monthly SM mean: 0.287 | peak month = Aug
      DJF   SM mean: 0.243 (n = 4061)
      MAM   SM mean: 0.236 (n = 4140)
      JJAS  SM mean: 0.341 (n = 5490)
      ON    SM mean: 0.319 (n = 2745)
    


    
![png](output_16_1.png)
    



```python

```


```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4 — Temporal trends in extreme rainfall and flood-event frequency
=========================================================================
Nature-journal style — REVISED with much larger, clearer typography.

Changes vs. original:
  • Base font 13 → 16, axis labels 13.5 → 18, titles 14.5 → 19
  • Tick labels 12 → 15.5, legend 10 → 12.5
  • Tick marks longer/thicker (size 5→7, width 0.9→1.2)
  • Larger markers, thicker lines, heavier spines
  • More figure real estate (22×17 in) so nothing collides
  • Wider spacing (hspace 0.48→0.55, wspace 0.24→0.30)
  • Suptitle / note fonts bumped up
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp, theilslopes, kendalltau

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
annual = pd.read_csv(PREP / "fig04" / "fig04_annual_indices.csv")
yearly = pd.read_csv(PREP / "fig08" / "fig08_yearly_counts.csv")
trends = pd.read_csv(PREP / "tables" / "table07_trends_annual.csv")
ks     = pd.read_csv(PREP / "tables" / "table07_ks_early_late.csv")
thr    = json.load(open(PREP / "thresholds" / "final_operational_thresholds.json"))

T_RX5  = float(thr["Rx5day_Stn"])
T_SM   = float(thr["SM"])
T_SWRI = float(thr["SWRI"])

FLOOD_YEARS = [1991, 1998, 2004, 2010, 2015, 2018, 2019, 2020, 2023, 2024]

annual.loc[annual["year"] == 2025, ["SM_max", "SM_JJAS_mean",
                                    "SWRI_max", "SWRI_JJAS_mean"]] = np.nan

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _get_trend(variable):
    row = trends[trends["variable"] == variable]
    if row.empty:
        return dict(slope=np.nan, p_mk=np.nan)
    return dict(slope=float(row["slope"].iloc[0]),
                p_mk=float(row["p_mk"].iloc[0]))

def _get_ks(variable):
    row = ks[ks["variable"] == variable]
    return None if row.empty else row.iloc[0]

def _p_str(p):
    if not np.isfinite(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"

def _add_trend_line(ax, x, y, slope, p_mk, color, label_prefix):
    x = np.asarray(x); y = np.asarray(y)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    xs = np.array([x[ok].min(), x[ok].max()])
    y0 = np.nanmedian(y[ok] - slope / 10.0 * x[ok])
    ys = y0 + slope / 10.0 * xs
    line, = ax.plot(xs, ys, color=color, lw=3.0, ls='-',
                    label=f"{label_prefix}: {slope:+.2f} /dec ({_p_str(p_mk)})",
                    zorder=4)
    return line

def _add_flood_shading(ax):
    for yr in FLOOD_YEARS:
        ax.axvspan(yr - 0.5, yr + 0.5, color="#d32f2f", alpha=0.10, zorder=0)

# -----------------------------------------------------------------------------
# Nature-style publication settings — LARGE, CLEAR, unicode-safe
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',

    # ---- BASE / TITLES / LABELS ----
    'font.size': 16,
    'axes.titlesize': 19,   'axes.titleweight': 'bold',
    'axes.labelsize': 18,   'axes.labelweight': 'bold',

    # ---- TICKS (the main visibility fix) ----
    'xtick.labelsize': 15.5, 'ytick.labelsize': 15.5,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8, 'ytick.minor.width': 0.8,
    'xtick.major.size': 7,    'ytick.major.size': 7,
    'xtick.minor.size': 4,    'ytick.minor.size': 4,
    'xtick.direction': 'out', 'ytick.direction': 'out',

    # ---- SPINES ----
    'axes.linewidth': 1.3,

    # ---- LEGEND ----
    'legend.fontsize': 12.5,
    'legend.title_fontsize': 13,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'legend.handlelength': 2.2,
    'legend.borderpad': 0.5,

    # ---- SAVE ----
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure — bigger canvas so larger fonts fit comfortably
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 17),dpi=500)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                       hspace=0.55, wspace=0.30,
                       left=0.06, right=0.975, top=0.925, bottom=0.06)

# ==========================================================
# (a) Rx1day trends
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0])
_add_flood_shading(ax_a)

yrs = annual["year"].values
ax_a.plot(yrs, annual["Rx1_stn"], 'o-', color="#1f4e8c",
          lw=2.2, ms=8, mec="black", mew=0.8,
          label="Station area-average", zorder=3)
ax_a.plot(yrs, annual["Rx1_chirps"], 's--', color="#2e7d32",
          lw=2.0, ms=7, mec="black", mew=0.6, alpha=0.85,
          label="CHIRPS basin-average", zorder=2)

t_stn = _get_trend("Rx1_stn")
t_ch  = _get_trend("Rx1_chirps")
_add_trend_line(ax_a, yrs, annual["Rx1_stn"], t_stn["slope"], t_stn["p_mk"],
                color="#0d47a1", label_prefix="Stn trend")
_add_trend_line(ax_a, yrs, annual["Rx1_chirps"], t_ch["slope"], t_ch["p_mk"],
                color="#1b5e20", label_prefix="CHIRPS trend")

ax_a.set_xlabel("Year")
ax_a.set_ylabel("Rx1day (mm)")
ax_a.set_title("(a) Annual maximum 1-day rainfall (Rx1day)", loc="left", pad=10)
ax_a.grid(ls=':', alpha=0.45)
ax_a.set_xlim(1979, 2027)
ax_a.set_ylim(annual["Rx1_stn"].min() * 0.85, annual["Rx1_stn"].max() * 1.22)
ax_a.legend(loc="upper left", fontsize=12, ncol=1, framealpha=0.95)

# ==========================================================
# (b) Rx5day trends + ML threshold
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1])
_add_flood_shading(ax_b)

ax_b.plot(yrs, annual["Rx5_stn"], 'o-', color="#1f4e8c",
          lw=2.2, ms=8, mec="black", mew=0.8,
          label="Station area-average", zorder=3)
ax_b.plot(yrs, annual["Rx5_chirps"], 's--', color="#2e7d32",
          lw=2.0, ms=7, mec="black", mew=0.6, alpha=0.85,
          label="CHIRPS basin-average", zorder=2)
ax_b.plot(yrs, annual["Rx5_buf"], '^:', color="#b26500",
          lw=1.8, ms=7, mec="black", mew=0.6, alpha=0.85,
          label="20-km buffer", zorder=1)

t5_stn = _get_trend("Rx5_stn")
t5_ch  = _get_trend("Rx5_chirps")
t5_buf = _get_trend("Rx5_buf")
_add_trend_line(ax_b, yrs, annual["Rx5_stn"],  t5_stn["slope"], t5_stn["p_mk"],
                color="#0d47a1", label_prefix="Stn")
_add_trend_line(ax_b, yrs, annual["Rx5_chirps"], t5_ch["slope"], t5_ch["p_mk"],
                color="#1b5e20", label_prefix="CHIRPS")
_add_trend_line(ax_b, yrs, annual["Rx5_buf"],   t5_buf["slope"], t5_buf["p_mk"],
                color="#7a4500", label_prefix="Buffer")

ax_b.axhline(T_RX5, color="black", ls="--", lw=2.2, zorder=4,
             label=f"ML Rx5 threshold = {T_RX5:.0f} mm")

ax_b.set_xlabel("Year")
ax_b.set_ylabel("Rx5day (mm)")
ax_b.set_title("(b) Annual maximum 5-day rainfall (Rx5day)", loc="left", pad=10)
ax_b.grid(ls=':', alpha=0.45)
ax_b.set_xlim(1979, 2027)
ax_b.set_ylim(annual["Rx5_chirps"].min() * 0.85, annual["Rx5_stn"].max() * 1.25)
ax_b.legend(loc="upper left", fontsize=12, ncol=1, framealpha=0.95)

# ==========================================================
# (c) R95p / R99p counts
# ==========================================================
ax_c = fig.add_subplot(gs[1, 0])
_add_flood_shading(ax_c)

width = 0.65
ax_c.bar(yrs, annual["R95p_stn"], width=width, color="#56b4e9",
         edgecolor="black", linewidth=0.7,
         label=r"R95p (days yr$^{-1}$)", zorder=2)
ax_c.bar(yrs, annual["R99p_stn"], width=width * 0.55, color="#d55e00",
         edgecolor="black", linewidth=0.7,
         label=r"R99p (days yr$^{-1}$)", zorder=3)

mean_r95 = float(np.nanmean(annual["R95p_stn"]))
ax_c.axhline(mean_r95, color="black", ls="--", lw=2.0, zorder=4,
             label=rf"R95p mean = {mean_r95:.1f} d yr$^{{-1}}$")

t95 = _get_trend("R95p_stn")
t99 = _get_trend("R99p_stn")
_add_trend_line(ax_c, yrs, annual["R95p_stn"], t95["slope"], t95["p_mk"],
                color="#1f78b4", label_prefix="R95p")
_add_trend_line(ax_c, yrs, annual["R99p_stn"], t99["slope"], t99["p_mk"],
                color="#b26500", label_prefix="R99p")

ax_c.set_xlabel("Year")
ax_c.set_ylabel("Days per year")
ax_c.set_title("(c) Extreme-rainfall frequency (R95p / R99p)", loc="left", pad=10)
ax_c.grid(axis='y', ls=':', alpha=0.45)
ax_c.set_xlim(1979, 2027)
ax_c.set_ylim(0, max(annual["R95p_stn"].max() * 1.30, 15))
ax_c.legend(loc="upper left", fontsize=12, ncol=1, framealpha=0.95)

# ==========================================================
# (d) Annual flood events by category
# ==========================================================
ax_d = fig.add_subplot(gs[1, 1])

year_col = "year"
cat_cols = ["Flash", "Short", "Prolonged"]
cat_colors = {"Flash": "#E66101", "Short": "#FDB863", "Prolonged": "#B2182B"}

bottom = np.zeros(len(yearly))
for cat in cat_cols:
    if cat not in yearly:
        continue
    ax_d.bar(yearly[year_col], yearly[cat], bottom=bottom, width=0.8,
             color=cat_colors[cat], edgecolor="black", linewidth=0.6,
             label=cat, zorder=2)
    bottom += yearly[cat].values

ax_d.plot(yearly[year_col], yearly["total"], color="black",
          lw=2.0, ls=':', marker='o', ms=5, zorder=5,
          label="Total events")

mean_total = float(yearly["total"].mean())
ax_d.axhline(mean_total, color="gray", ls='--', lw=1.5, zorder=1,
             label=rf"Mean = {mean_total:.1f} events yr$^{{-1}}$")

sl, ic, _, _ = theilslopes(yearly["total"].values, yearly["year"].values)
tau, p = kendalltau(yearly["year"].values, yearly["total"].values)
xs_t = np.array([yearly["year"].min(), yearly["year"].max()])
ax_d.plot(xs_t, ic + sl * xs_t, color="#0d47a1", lw=2.6, ls='-', zorder=5,
          label=f"Total trend: {sl*10:+.2f} /dec ({_p_str(p)})")

ax_d.set_xlabel("Year")
ax_d.set_ylabel("Flood-event days per year")
ax_d.set_title("(d) Annual flood-event days by category", loc="left", pad=10)
ax_d.grid(axis='y', ls=':', alpha=0.45)
ax_d.set_xlim(1979, 2027)
ax_d.set_ylim(bottom=0)
ax_d.legend(loc="upper left", fontsize=12, ncol=1, framealpha=0.95)

cat_summ = "  |  ".join([f"{c}: {int(yearly[c].sum())}"
                          for c in cat_cols if c in yearly])
ax_d.text(0.98, 0.97, cat_summ, transform=ax_d.transAxes,
          va="top", ha="right", fontsize=13, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                    edgecolor="#666", alpha=0.92))

# ==========================================================
# (e) Annual max SM and SWRI (twin axis)
# ==========================================================
ax_e = fig.add_subplot(gs[2, 0])
_add_flood_shading(ax_e)

ax_e.plot(yrs, annual["SM_max"], 'o-', color="#5d4037",
          lw=2.2, ms=7.5, mec="black", mew=0.7,
          label="Annual max SM", zorder=3)
ax_e.axhline(T_SM, color="#d32f2f", ls="--", lw=2.0, zorder=2,
             label=f"ML SM threshold = {T_SM:.3f}")

t_sm = _get_trend("SM_max")
_add_trend_line(ax_e, yrs, annual["SM_max"], t_sm["slope"], t_sm["p_mk"],
                color="#3e2723", label_prefix="SM trend")

ax_e.set_xlabel("Year")
ax_e.set_ylabel(r"Annual max soil moisture (m$^3$ m$^{-3}$)",
                color="#5d4037", fontweight="bold")
ax_e.tick_params(axis='y', labelcolor="#5d4037", labelsize=15.5)
ax_e.set_ylim(0.20, 0.55)
ax_e.grid(ls=':', alpha=0.45)
ax_e.set_xlim(1979, 2027)
ax_e.set_title("(e) Annual maximum SM and SWRI", loc="left", pad=10)

ax_e2 = ax_e.twinx()
ax_e2.plot(yrs, annual["SWRI_max"], 's-', color="#b71c1c",
           lw=2.0, ms=7, mec="black", mew=0.6,
           label="Annual max SWRI", zorder=3)
ax_e2.axhline(T_SWRI, color="#7f0000", ls=':', lw=1.8, zorder=2,
              label=f"ML SWRI threshold = {T_SWRI:.2f}")
ax_e2.set_ylabel("Annual max SWRI", color="#b71c1c", fontweight="bold")
ax_e2.tick_params(axis='y', labelcolor="#b71c1c", labelsize=15.5)
ax_e2.spines['top'].set_visible(False)

t_swri = _get_trend("SWRI_max")
_add_trend_line(ax_e2, yrs, annual["SWRI_max"], t_swri["slope"], t_swri["p_mk"],
                color="#7f0000", label_prefix="SWRI trend")

h1, l1 = ax_e.get_legend_handles_labels()
h2, l2 = ax_e2.get_legend_handles_labels()
ax_e.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11.5, ncol=1,
            framealpha=0.95)

# ==========================================================
# (f) Early vs Late KS test — ECDF overlay
# ==========================================================
ax_f = fig.add_subplot(gs[2, 1])

early = annual[annual["year"] <= 2000]
late  = annual[annual["year"] >= 2001]

vars_f = [("Rx5_stn",  "Rx5day",   "#1f4e8c"),
          ("SM_max",   "Max SM",   "#5d4037"),
          ("SWRI_max", "Max SWRI", "#b71c1c")]

for i, (var, label, color) in enumerate(vars_f):
    x_early = early[var].dropna().values
    x_late  = late[var].dropna().values
    if len(x_early) < 5 or len(x_late) < 5:
        continue
    all_x = np.concatenate([x_early, x_late])
    lo, hi = np.nanmin(all_x), np.nanmax(all_x)
    if hi <= lo:
        continue
    xe = (x_early - lo) / (hi - lo)
    xl = (x_late  - lo) / (hi - lo)

    x_axis = np.linspace(0, 1, 200)
    ecdf_e = np.array([(xe <= v).mean() for v in x_axis])
    ecdf_l = np.array([(xl <= v).mean() for v in x_axis])
    ax_f.plot(x_axis, ecdf_e, color=color, ls='--', lw=2.0, alpha=0.9)
    ax_f.plot(x_axis, ecdf_l, color=color, ls='-',  lw=3.0, label=label)

    stat, pval = ks_2samp(x_early, x_late)
    ax_f.text(0.03, 0.94 - 0.10 * i,
              f"{label:<10s}  KS={stat:.2f}  {_p_str(pval)}",
              transform=ax_f.transAxes, fontsize=12,
              family='monospace',
              bbox=dict(boxstyle="round,pad=0.30",
                        facecolor=color, alpha=0.10,
                        edgecolor=color, linewidth=1.0))

ax_f.set_xlabel("Normalised value")
ax_f.set_ylabel("Empirical CDF")
ax_f.set_title("(f) Early (1981–2000) vs Late (2001–2025) shift (KS test)",
               loc="left", pad=10)
ax_f.grid(ls=':', alpha=0.45)
ax_f.set_xlim(0, 1)
ax_f.set_ylim(0, 1.02)

legend_f = [
    Line2D([0], [0], color="black", lw=3.0, ls='-',  label="2001–2025 (late)"),
    Line2D([0], [0], color="black", lw=2.0, ls='--', label="1981–2000 (early)"),
]
ax_f.legend(handles=legend_f, loc="lower right", fontsize=12, ncol=1,
            framealpha=0.95)

# ==========================================================
# Suptitle + flood-band note
# ==========================================================
fig.suptitle(
    "Figure 4: Temporal trends in extreme rainfall and flood-event frequency "
    "(1981–2025)",
    fontsize=20, fontweight="bold", y=0.985)

#fig.text(0.99, 0.952,
##         "Red bands = documented flood years",
  #       ha="right", va="top", fontsize=13, color="#b71c1c", fontweight="bold",
  #       bbox=dict(boxstyle="round,pad=0.30", facecolor="#ffe6e6",
   #                edgecolor="#b71c1c", linewidth=1.0))

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig04_Temporal_Trends_v2_bigfont.png"
out_pdf = FIG_DIR / "Fig04_Temporal_Trends_v2_bigfont.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf,           bbox_inches="tight", facecolor="white")
print(f"✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# ==========================================================
# Summary
# ==========================================================
print("\n=== Trend summary (station series) ===")
for v in ["Rx1_stn", "Rx5_stn", "R95p_stn", "R99p_stn", "SM_max", "SWRI_max"]:
    t = _get_trend(v)
    sig = "**" if t["p_mk"] < 0.05 else "  "
    print(f"  {v:<10s}  {t['slope']:+7.3f} /dec  p={t['p_mk']:.4f} {sig}")

print("\n=== KS early vs late ===")
for v in ["Rx1_stn", "Rx5_stn", "SM_max", "SWRI_max"]:
    r = _get_ks(v)
    if r is not None:
        print(f"  {v:<10s}  KS={r['ks_stat']:.3f}  p={r['ks_p']:.4f}")

print("\n=== Category totals ===")
for c in cat_cols:
    if c in yearly:
        print(f"  {c:<10s}: {int(yearly[c].sum())} events")
print(f"  Mean = {mean_total:.2f} events/yr | trend = {sl*10:+.2f} events/dec "
      f"(p={p:.3f})")

plt.show()
```

    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig04_Temporal_Trends_v2_bigfont.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig04_Temporal_Trends_v2_bigfont.pdf
    
    === Trend summary (station series) ===
      Rx1_stn      +0.287 /dec  p=0.8602   
      Rx5_stn      -3.445 /dec  p=0.1770   
      R95p_stn     +0.000 /dec  p=0.7151   
      R99p_stn     +0.000 /dec  p=0.5975   
      SM_max       -0.012 /dec  p=0.0000 **
      SWRI_max     +0.134 /dec  p=0.8756   
    
    === KS early vs late ===
      Rx1_stn     KS=0.320  p=0.1676
      Rx5_stn     KS=0.180  p=0.8040
      SM_max      KS=0.600  p=0.0003
      SWRI_max    KS=0.190  p=0.7531
    
    === Category totals ===
      Flash     : 0 events
      Short     : 5 events
      Prolonged : 6 events
      Mean = 1.00 events/yr | trend = +0.00 events/dec (p=nan)
    


    
![png](output_19_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5 — Spatial trends of Rx1day and Rx5day, 1981–2025
=========================================================================
Only the study area (custom basin polygon) is shown; everything outside
the basin is masked to white. Nature-journal style.
Single shared horizontal colourbar at the bottom (both panels use mm/dec).

Panels:
  (a) Rx1day trend magnitude (mm decade⁻¹) with significance stippling
  (b) Rx5day trend magnitude (mm decade⁻¹) with significance stippling

Reads:
    prep_revised_v2/fig05/fig05_rx1day_trend_grid.nc
    prep_revised_v2/fig05/fig05_rx5day_trend_grid.nc
    prep_revised_v2/fig01/fig01_stations_retained14.csv
    output_shapefiles/custom_basin.shp
    output_shapefiles/clipped_rivers.shp

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig05_Spatial_Trends_v2.png / .pdf
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401  (enables .rio)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
SHP_DIR = ROOT / "output_shapefiles"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Display extent — study domain (slightly padded so basin edge is visible)
EXT = [94.0, 100.5, 20, 27.5]

# -----------------------------------------------------------------------------
# Load trend grids  (robust: set spatial dims AFTER rename)
# -----------------------------------------------------------------------------
def load_trend(tag):
    path = PREP / "fig05" / f"fig05_{tag.lower()}_trend_grid.nc"
    ds   = xr.open_dataset(path)

    slope = ds[f"{tag}_slope_mm_per_decade"]
    pval  = ds[f"{tag}_mk_pvalue"]

    # harmonise dimension names
    rename = {}
    for c in ("latitude", "longitude"):
        if c in slope.dims:
            rename[c] = c[:3]
    if rename:
        slope = slope.rename(rename)
        pval  = pval.rename(rename)

    slope = slope.sortby("lat").sortby("lon")
    pval  = pval.sortby("lat").sortby("lon")

    slope = slope.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
    pval  = pval.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")

    print(f"  {tag}: dims={slope.dims}, shape={slope.shape}, "
          f"lon={float(slope.lon.min()):.2f}..{float(slope.lon.max()):.2f}, "
          f"lat={float(slope.lat.min()):.2f}..{float(slope.lat.max()):.2f}")
    return slope, pval


print("Loading trend grids...")
slope_rx1, pval_rx1 = load_trend("Rx1day")
slope_rx5, pval_rx5 = load_trend("Rx5day")

# -----------------------------------------------------------------------------
# Load geometries
# -----------------------------------------------------------------------------
print("Loading shapefiles...")
basin  = gpd.read_file(SHP_DIR / "custom_basin.shp").to_crs("EPSG:4326")
rivers = gpd.read_file(SHP_DIR / "clipped_rivers.shp").to_crs("EPSG:4326")

shweli    = rivers[rivers["Sub_Name"].astype(str).str.contains("Shweli",    case=False, na=False)]
irrawaddy = rivers[rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)]

stn_df   = pd.read_csv(PREP / "fig01" / "fig01_stations_retained14.csv")
stations = gpd.GeoDataFrame(
    stn_df, geometry=gpd.points_from_xy(stn_df["Lon"], stn_df["Lat"]),
    crs="EPSG:4326")

basin_buffer = basin.geometry.union_all().buffer(0.03)
borders_path = shapereader.natural_earth('50m', 'cultural', 'admin_0_boundary_lines_land')
borders_gdf  = gpd.read_file(borders_path).to_crs("EPSG:4326")
border_clip  = borders_gdf[borders_gdf.intersects(basin_buffer)].copy()
border_clip["geometry"] = border_clip.geometry.intersection(basin_buffer)
border_clip = border_clip[~border_clip.is_empty]

# -----------------------------------------------------------------------------
# Mask trend grids to basin polygon (coordinate mask, no rioxarray round-trip)
# -----------------------------------------------------------------------------
print("Masking trend grids to basin polygon...")
basin_geom = basin.geometry.union_all()

def basin_mask(da, geom):
    lon = da.lon.values
    lat = da.lat.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = MplPath(np.array(geom.exterior.coords)).contains_points(pts)
    return xr.DataArray(inside.reshape(lon2d.shape),
                        coords={"lat": lat, "lon": lon},
                        dims=("lat", "lon"))

def mask_to_basin(da):
    return da.where(basin_mask(da, basin_geom))

slope_rx1 = mask_to_basin(slope_rx1)
pval_rx1  = mask_to_basin(pval_rx1)
slope_rx5 = mask_to_basin(slope_rx5)
pval_rx5  = mask_to_basin(pval_rx5)

print(f"  Rx1day valid cells inside basin: {int(np.isfinite(slope_rx1.values).sum())}")
print(f"  Rx5day valid cells inside basin: {int(np.isfinite(slope_rx5.values).sum())}")

# -----------------------------------------------------------------------------
# Shared symmetric colour scale (basin cells only)
# -----------------------------------------------------------------------------
max_abs = max(np.nanpercentile(np.abs(slope_rx1.values), 98),
              np.nanpercentile(np.abs(slope_rx5.values), 98))
max_abs = np.ceil(max_abs / 2) * 2
levels  = np.linspace(-max_abs, max_abs, 17)
cmap    = plt.get_cmap('RdBu_r')
norm    = Normalize(vmin=-max_abs, vmax=max_abs)
cb_ticks = [-max_abs, -max_abs/2, 0, max_abs/2, max_abs]
print(f"Symmetric colour scale: ±{max_abs:.1f} mm/dec, {len(levels)} levels")

# -----------------------------------------------------------------------------
# Nature-style configuration
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 13,
    'axes.titlesize': 14.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 13.5,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11.5, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'hatch.linewidth': 0.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure (tight 2-panel layout, room for shared colourbar + legend at bottom)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8.6), dpi=500)

# Panels — leave bottom (0 – 0.30) for colourbar + legend
ax_a = fig.add_axes([0.055, 0.30, 0.40, 0.63], projection=ccrs.PlateCarree())
ax_b = fig.add_axes([0.535, 0.30, 0.40, 0.63], projection=ccrs.PlateCarree())

# -----------------------------------------------------------------------------
# Drawing function
# -----------------------------------------------------------------------------
def draw_panel(ax, slope_da, pval_da, panel_letter, title):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
    ax.set_facecolor('white')

    lon2d, lat2d = np.meshgrid(slope_da.lon.values, slope_da.lat.values)
    cf = ax.contourf(lon2d, lat2d, slope_da.values,
                     levels=levels, cmap=cmap, norm=norm,
                     extend='both',
                     transform=ccrs.PlateCarree(), zorder=2)

    # Significance stippling (p < 0.05) — only inside basin
    sig_mask = np.where(pval_da.values < 0.05, 1, np.nan)
    ax.contourf(lon2d, lat2d, sig_mask,
                levels=[0.5, 1.5], colors='none',
                hatches=['...'],
                transform=ccrs.PlateCarree(), zorder=3)

    basin.plot(ax=ax, facecolor='none', edgecolor='black',
               linewidth=1.8, transform=ccrs.PlateCarree(), zorder=5)

    if not shweli.empty:
        shweli.plot(ax=ax, color='#6a1b9a', linewidth=2.2,
                    transform=ccrs.PlateCarree(), zorder=6)
    if not irrawaddy.empty:
        irrawaddy.plot(ax=ax, color='#0d47a1', linewidth=2.0,
                       transform=ccrs.PlateCarree(), zorder=6)

    if not border_clip.empty:
        border_clip.plot(ax=ax, color='#333333', linewidth=1.3,
                         linestyle='--',
                         transform=ccrs.PlateCarree(), zorder=7)

    stations.plot(ax=ax, color='#d32f2f', marker='^', markersize=70,
                  edgecolor='black', linewidth=1.0,
                  transform=ccrs.PlateCarree(), zorder=8)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                       alpha=0.45, linestyle=':')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = True
    gl.bottom_labels = True
    gl.xlabel_style  = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style  = {'size': 12, 'weight': 'bold'}

    # Panel letter
    ax.text(0.025, 0.975, panel_letter, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='black', linewidth=1.0, alpha=0.95),
            zorder=100)

    ax.set_title(title, loc='left', pad=10, fontsize=14.5, fontweight='bold')

    # Stats box
    vals  = slope_da.values
    pv    = pval_da.values
    ok    = np.isfinite(vals)
    mean_t   = float(np.nanmean(vals[ok]))
    pct_sig  = float(100 * np.sum((pv < 0.05) & ok) / np.sum(ok))
    ax.text(0.975, 0.03,
            f"Mean: {mean_t:+.2f} mm/dec\nSignificant: {pct_sig:.1f}%",
            transform=ax.transAxes, va='bottom', ha='right',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#444', alpha=0.95, linewidth=0.9),
            zorder=100)

    return cf

print("Plotting...")
cf_a = draw_panel(ax_a, slope_rx1, pval_rx1, "(a)",
                   "Rx1day trend (mm decade$^{-1}$)")
cf_b = draw_panel(ax_b, slope_rx5, pval_rx5, "(b)",
                   "Rx5day trend (mm decade$^{-1}$)")

# -----------------------------------------------------------------------------
# SINGLE shared horizontal colourbar — centred under both panels
# -----------------------------------------------------------------------------
cax = fig.add_axes([0.30, 0.175, 0.40, 0.022])      # centred, 40 % figure width
cb  = fig.colorbar(cf_a, cax=cax, orientation='horizontal', extend='both')
cb.set_label("Sen's slope (mm decade$^{-1}$)",
             fontsize=13, fontweight='bold', labelpad=6)
cb.ax.tick_params(labelsize=11.5, length=4, width=0.9)
cb.outline.set_linewidth(0.9)
cb.set_ticks(cb_ticks)
cb.ax.set_xticklabels([f"{t:+.0f}" for t in cb_ticks], fontweight='bold')

# -----------------------------------------------------------------------------
# Shared legend — just below the colourbar
# -----------------------------------------------------------------------------
legend_elements = [
    Line2D([0], [0], color='black',    lw=1.8, label='Basin boundary'),
    Line2D([0], [0], color='#6a1b9a',  lw=2.2, label='Shweli River'),
    Line2D([0], [0], color='#0d47a1',  lw=2.0, label='Ayeyarwady River'),
    Line2D([0], [0], color='#333333',  lw=1.3, ls='--',
           label='China–Myanmar border'),
    Line2D([0], [0], marker='^', color='#d32f2f',
           markeredgecolor='black', linestyle='None',
           markersize=12, label='Stations (n = 14)'),
    Patch(facecolor='none', edgecolor='black', hatch='...',
          label='Significant (p < 0.05)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.025), fontsize=11.5,
           frameon=True, framealpha=0.95, edgecolor='0.3')

# -----------------------------------------------------------------------------
# Suptitle
# -----------------------------------------------------------------------------
fig.suptitle(
    "Figure 5: Spatial trends of Rx1day and Rx5day, 1981–2025 "
    "(Sen's slope; stippling = p < 0.05, Mann–Kendall)", alpha=0,
    fontsize=15.5, fontweight='bold', y=0.975)

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_png = FIG_DIR / "Fig05_Spatial_Trends_v2.png"
out_pdf = FIG_DIR / "Fig05_Spatial_Trends_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# -----------------------------------------------------------------------------
# Numeric summary (basin cells only)
# -----------------------------------------------------------------------------
print("\n=== Spatial trend summary (basin cells only) ===")
for name, slope, pval in [("Rx1day", slope_rx1, pval_rx1),
                           ("Rx5day", slope_rx5, pval_rx5)]:
    v = slope.values; p = pval.values
    ok = np.isfinite(v)
    pos = float((v[ok] > 0).mean() * 100)
    neg = float((v[ok] < 0).mean() * 100)
    sig = float(((p < 0.05) & ok).sum() / ok.sum() * 100)
    print(f"  {name}: mean {np.nanmean(v[ok]):+.2f} mm/dec, "
          f"positive cells {pos:.1f}%, negative cells {neg:.1f}%, "
          f"significant (p<0.05) {sig:.1f}%")

plt.show()
```

    Loading trend grids...
      Rx1day: dims=('lat', 'lon'), shape=(130, 300), lon=90.02..104.98, lat=20.02..26.47
      Rx5day: dims=('lat', 'lon'), shape=(130, 300), lon=90.02..104.98, lat=20.02..26.47
    Loading shapefiles...
    Masking trend grids to basin polygon...
      Rx1day valid cells inside basin: 5201
      Rx5day valid cells inside basin: 5201
    Symmetric colour scale: ±6.0 mm/dec, 17 levels
    Plotting...
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig05_Spatial_Trends_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig05_Spatial_Trends_v2.pdf
    
    === Spatial trend summary (basin cells only) ===
      Rx1day: mean +0.58 mm/dec, positive cells 61.3%, negative cells 38.7%, significant (p<0.05) 5.5%
      Rx5day: mean +0.58 mm/dec, positive cells 61.3%, negative cells 38.7%, significant (p<0.05) 5.5%
    


    
![png](output_21_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 6 — Threshold-variable evaluation and selection
=========================================================================
Nature-journal style: large clear labels, ROC curves, AUC summary,
POD vs FAR trade-off, RF feature importance, final operating thresholds.

Panels:
  (a) ROC curves — annual scale (flood-year labels), 6 curated variables
  (b) ROC curves — daily scale (flood-day labels), all 5 variables
  (c) AUC comparison — grouped bars, annual vs daily scale
  (d) POD vs FAR trade-off scatter
  (e) RF feature importance (annual features)
  (f) Final operating thresholds summary card

Reads:
    prep_revised_v2/tables/table04_thresholds_annual_scale.csv
    prep_revised_v2/tables/table04_thresholds_daily_scale.csv
    prep_revised_v2/thresholds/fig06_roc_annual.csv
    prep_revised_v2/thresholds/fig06_rf_importance_annual.csv
    prep_revised_v2/thresholds/final_operational_thresholds.json
    prep_revised_v2/fig12/fig12_daily_labelled.csv

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig06_Threshold_Evaluation_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
print("Loading prep outputs...")
annual_thr = pd.read_csv(PREP/"tables"/"table04_thresholds_annual_scale.csv")
daily_thr  = pd.read_csv(PREP/"tables"/"table04_thresholds_daily_scale.csv")
rf_imp     = pd.read_csv(PREP/"thresholds"/"fig06_rf_importance_annual.csv")
roc_annual = pd.read_csv(PREP/"thresholds"/"fig06_roc_annual.csv")
final_thr  = json.load(open(PREP/"thresholds"/"final_operational_thresholds.json"))
daily_df   = pd.read_csv(PREP/"fig12"/"fig12_daily_labelled.csv",
                         parse_dates=["date"], index_col="date")

print(f"  Annual thresholds : {len(annual_thr)} vars")
print(f"  Daily thresholds  : {len(daily_thr)} vars")
print(f"  RF importance     : {len(rf_imp)} features")
print(f"  Final thresholds  : {final_thr}")

# -----------------------------------------------------------------------------
# Recompute daily ROC curves (not saved in prep)
# -----------------------------------------------------------------------------
WET_MM = 5.0
wet_mask = daily_df["P_stn"] >= WET_MM

daily_specs = [
    ("Rx1day_Stn",    "P_stn",      True),
    ("Rx5day_Stn",    "Rx5_stn",    True),
    ("Rx5day_CHIRPS", "Rx5_chirps", True),
    ("SM",            "SM",         False),
    ("SWRI",          "SWRI",       True),
]
daily_roc = {}
for label, col, wet_filter in daily_specs:
    sub = daily_df[[col, "flood_day"]].copy()
    if wet_filter:
        sub = sub.loc[wet_mask.reindex(sub.index, fill_value=False)]
    sub = sub.dropna()
    if sub["flood_day"].sum() < 10:
        continue
    fpr, tpr, thr = roc_curve(sub["flood_day"].values, sub[col].values)
    a = auc(fpr, tpr)
    j = tpr - fpr
    k = int(np.argmax(j))
    daily_roc[label] = dict(fpr=fpr, tpr=tpr, auc=a,
                             pod=float(tpr[k]), far=float(fpr[k]),
                             opt_thr=float(thr[k]))
    print(f"  {label:<16s}  AUC={a:.3f}  POD={tpr[k]:.2f}  FAR={fpr[k]:.2f}  "
          f"opt={thr[k]:.3f}")

# -----------------------------------------------------------------------------
# Nature-family color palette
# -----------------------------------------------------------------------------
COLORS = {
    'Rx1day_Stn':        '#E64B35',   # red
    'Rx1day_CHIRPS':     '#E64B35',
    'Rx5day_Stn':        '#4DBBD5',   # cyan
    'Rx5day_CHIRPS':     '#00A087',   # teal
    'Rx5day_20kmBuffer': '#3C5488',   # deep blue
    'R95p_Stn':          '#F39B7F',   # salmon
    'R99p_Stn':          '#8491B4',   # grey-blue
    'SM_annual_max':     '#7E6148',   # brown
    'SM_JJAS_mean':      '#B09C85',
    'SM':                '#7E6148',
    'SWRI_annual_max':   '#91D1C2',   # mint
    'SWRI_JJAS_mean':    '#B5C8B2',
    'SWRI':              '#91D1C2',
}

# -----------------------------------------------------------------------------
# Nature-style configuration — LARGE, CLEAR
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 16.5, 'axes.titleweight': 'bold',
    'axes.labelsize': 15.5,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'legend.fontsize': 13, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure — 2 rows × 3 cols
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10.5),dpi=500)
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.25, 1.0],
    hspace=0.42, wspace=0.28,
    left=0.065, right=0.975, top=0.925, bottom=0.075,
)

# ==========================================================
# (a) ROC curves — ANNUAL scale
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0])

show_annual = ['SWRI_annual_max', 'SM_annual_max',
               'Rx1day_Stn', 'Rx5day_Stn',
               'R95p_Stn', 'R99p_Stn']
annual_idx = annual_thr.set_index('variable')

for var in show_annual:
    sub = roc_annual[roc_annual['variable'] == var].sort_values('fpr')
    if sub.empty or var not in annual_idx.index:
        continue
    auc_val = float(annual_idx.loc[var, 'AUC'])
    color = COLORS.get(var, '#666')
    ax_a.plot(sub['fpr'], sub['tpr'], color=color, lw=2.6,
              label=f"{var}   AUC = {auc_val:.3f}")
    # Youden-J optimum point
    pod = float(annual_idx.loc[var, 'POD_youden'])
    far = float(annual_idx.loc[var, 'FAR_youden'])
    ax_a.scatter(far, pod, s=110, color=color,
                 edgecolor='black', linewidth=1.2, zorder=6)

ax_a.plot([0, 1], [0, 1], '--', color='gray', lw=1.4,
          label='No-skill (AUC = 0.5)')
ax_a.set_xlim(-0.02, 1.02); ax_a.set_ylim(-0.02, 1.02)
ax_a.set_xlabel('False positive rate (1 − specificity)')
ax_a.set_ylabel('True positive rate (sensitivity)')
ax_a.set_title('(a) ROC — annual scale (flood-year labels)',
               loc='left', pad=8, fontsize=15)
ax_a.grid(ls=':', alpha=0.4)
ax_a.set_aspect('equal')
ax_a.legend(loc='lower right', fontsize=8.5, framealpha=0.95,
            edgecolor='0.3', borderpad=0.5, labelspacing=0.35)

# ==========================================================
# (b) ROC curves — DAILY scale
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1])

for label, info in daily_roc.items():
    color = COLORS.get(label, '#666')
    ax_b.plot(info['fpr'], info['tpr'], color=color, lw=2.6,
              label=f"{label}   AUC = {info['auc']:.3f}")
    ax_b.scatter(info['far'], info['pod'], s=110, color=color,
                 edgecolor='black', linewidth=1.2, zorder=6)

ax_b.plot([0, 1], [0, 1], '--', color='gray', lw=1.4,
          label='No-skill (AUC = 0.5)')
ax_b.set_xlim(-0.02, 1.02); ax_b.set_ylim(-0.02, 1.02)
ax_b.set_xlabel('False positive rate (1 − specificity)')
ax_b.set_ylabel('True positive rate (sensitivity)')
ax_b.set_title('(b) ROC — daily scale (flood-day windows)',
               loc='left', pad=8, fontsize=15)
ax_b.grid(ls=':', alpha=0.4)
ax_b.set_aspect('equal')
ax_b.legend(loc='lower right', fontsize=8.5, framealpha=0.95,
            edgecolor='0.3', borderpad=0.5, labelspacing=0.35)

# ==========================================================
# (c) AUC comparison — grouped horizontal bars
# ==========================================================
ax_c = fig.add_subplot(gs[0, 2])

groups = [
    ("Rx1day", "Rx1day_Stn",       "Rx1day_Stn"),
    ("Rx5day", "Rx5day_Stn",       "Rx5day_Stn"),
    ("SM",     "SM_annual_max",    "SM"),
    ("SWRI",   "SWRI_annual_max",  "SWRI"),
]

annual_aucs, daily_aucs = [], []
for _, ann_v, day_v in groups:
    a_a = float(annual_idx.loc[ann_v, 'AUC']) if ann_v in annual_idx.index else np.nan
    a_d = daily_roc.get(day_v, {}).get('auc', np.nan)
    annual_aucs.append(a_a)
    daily_aucs.append(a_d)

y_pos = np.arange(len(groups))
h = 0.35

bars_a = ax_c.barh(y_pos - h/2, annual_aucs, height=h,
                   color='#a5b4cf', edgecolor='black', linewidth=0.9,
                   label='Annual scale (flood-year labels)', zorder=2)
bars_d = ax_c.barh(y_pos + h/2, daily_aucs, height=h,
                   color='#e8b4a4', edgecolor='black', linewidth=0.9,
                   label='Daily scale (flood-day labels)', zorder=2)

# Value labels
for i, v in enumerate(annual_aucs):
    if np.isfinite(v):
        ax_c.text(v + 0.012, y_pos[i] - h/2, f"{v:.3f}",
                  va='center', ha='left', fontsize=10, fontweight='bold')
for i, v in enumerate(daily_aucs):
    if np.isfinite(v):
        ax_c.text(v + 0.012, y_pos[i] + h/2, f"{v:.3f}",
                  va='center', ha='left', fontsize=10, fontweight='bold')

ax_c.axvline(0.5, color='gray', ls='--', lw=1.2, zorder=1)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels([g[0] for g in groups], fontsize=12, fontweight='bold')
ax_c.set_xlim(0.40, 1.0)
ax_c.set_xlabel('AUC (area under ROC curve)')
ax_c.set_title('(c) AUC comparison: annual vs daily scale',
               loc='left', pad=8, fontsize=15)
ax_c.grid(axis='x', ls=':', alpha=0.4)
ax_c.set_axisbelow(True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.legend(loc='lower right', fontsize=9.5, framealpha=0.95,
            edgecolor='0.3')

# ==========================================================
# (d) POD vs FAR trade-off
# ==========================================================
ax_d = fig.add_subplot(gs[1, 0])

# Annual points
for var in annual_idx.index:
    row = annual_idx.loc[var]
    ax_d.scatter(row['FAR_youden'], row['POD_youden'],
                 s=180, color='#a5b4cf', edgecolor='black',
                 linewidth=1.2, marker='s', alpha=0.85, zorder=3)
    ax_d.annotate(var.replace('_annual_max','').replace('_Stn',''),
                  xy=(row['FAR_youden'], row['POD_youden']),
                  xytext=(5, 5), textcoords='offset points',
                  fontsize=8, color='#333',
                  bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            alpha=0.8, edgecolor='#999', linewidth=0.4))

# Daily points
for var, info in daily_roc.items():
    ax_d.scatter(info['far'], info['pod'],
                 s=220, color='#e8b4a4', edgecolor='black',
                 linewidth=1.2, marker='o', alpha=0.9, zorder=4)
    ax_d.annotate(var, xy=(info['far'], info['pod']),
                  xytext=(6, -8), textcoords='offset points',
                  fontsize=9, fontweight='bold', color='#b71c1c',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            alpha=0.9, edgecolor='#b71c1c', linewidth=0.6))

ax_d.plot([0, 1], [1, 0], '--', color='gray', lw=1.2, zorder=1)
ax_d.axhline(0.5, color='#bbb', ls=':', lw=1, zorder=1)
ax_d.axvline(0.5, color='#bbb', ls=':', lw=1, zorder=1)

ax_d.set_xlim(-0.02, 1.02); ax_d.set_ylim(-0.02, 1.02)
ax_d.set_xlabel('False alarm rate (FAR)')
ax_d.set_ylabel('Probability of detection (POD)')
ax_d.set_title('(d) POD vs FAR at Youden-J optimum',
               loc='left', pad=8, fontsize=15)
ax_d.grid(ls=':', alpha=0.4)

legend_d = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#a5b4cf',
           markeredgecolor='black', markersize=11,
           label='Annual scale'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e8b4a4',
           markeredgecolor='black', markersize=12,
           label='Daily scale'),
]
ax_d.legend(handles=legend_d, loc='lower left', fontsize=9.5,
            framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (e) RF feature importance
# ==========================================================
ax_e = fig.add_subplot(gs[1, 1])

rf_imp_sorted = rf_imp.sort_values('importance', ascending=True)
feat_colors = [COLORS.get(f, '#666')
               for f in rf_imp_sorted['feature']]
bars_e = ax_e.barh(range(len(rf_imp_sorted)),
                    rf_imp_sorted['importance'],
                    color=feat_colors,
                    edgecolor='black', linewidth=0.9, height=0.65, zorder=2)

for i, v in enumerate(rf_imp_sorted['importance']):
    ax_e.text(v + 0.008, i, f"{v:.3f}",
              va='center', ha='left',
              fontsize=10.5, fontweight='bold')

ax_e.set_yticks(range(len(rf_imp_sorted)))
ax_e.set_yticklabels([f.replace('_max',' (max)').replace('_stn',' (station)')
                       for f in rf_imp_sorted['feature']],
                      fontsize=11, fontweight='bold')
ax_e.set_xlim(0, rf_imp_sorted['importance'].max() * 1.28)
ax_e.set_xlabel('Gini importance')
ax_e.set_title('(e) Random Forest importance (annual features)',
               loc='left', pad=8, fontsize=15)
ax_e.grid(axis='x', ls=':', alpha=0.4)
ax_e.set_axisbelow(True)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# ==========================================================
# (f) Final threshold card
# ==========================================================
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1)
ax_f.axis('off')

ax_f.text(0.5, 0.965, "(f) Final operating thresholds",
          fontsize=15.5, fontweight='bold', ha='center', va='top',
          transform=ax_f.transAxes)
ax_f.text(0.5, 0.305,
          "Selected by ROC/Youden-J on documented flood labels",
          fontsize=9.5, ha='center', va='top', style='italic',
          color='#555', transform=ax_f.transAxes)

threshold_rows = [
    ("Rx1day",  f"{final_thr['Rx1day_Stn']:.1f}",  "mm",    "annual",  "Rx1day_Stn"),
    ("Rx5day",  f"{final_thr['Rx5day_Stn']:.1f}",  "mm",    "annual",  "Rx5day_Stn"),
    ("SM",      f"{final_thr['SM']:.3f}",          "m³/m³", "daily",   "SM"),
    ("SWRI",    f"{final_thr['SWRI']:.2f}",        "",      "daily",   "SWRI"),
]

y0 = 0.82
row_h = 0.14
for i, (var, val, unit, scale, color_key) in enumerate(threshold_rows):
    y = y0 - i * row_h
    color = COLORS.get(color_key, '#666')

    # Background band
    rect = Rectangle((0.03, y - row_h * 0.42), 0.94, row_h * 0.78,
                     facecolor=color, alpha=0.15,
                     edgecolor=color, linewidth=1.8,
                     transform=ax_f.transAxes, zorder=1)
    ax_f.add_patch(rect)

    # Variable name
    ax_f.text(0.06, y, var, fontsize=13, fontweight='bold',
              va='center', ha='left', transform=ax_f.transAxes, zorder=3)

    # Threshold value (big, colored)
    ax_f.text(0.36, y, val, fontsize=15, fontweight='bold',
              color=color, va='center', ha='left',
              transform=ax_f.transAxes, zorder=3)

    # Unit
    if unit:
        ax_f.text(0.62, y, unit, fontsize=10.5,
                  va='center', ha='left', color='#333',
                  transform=ax_f.transAxes, zorder=3)

    # Scale badge
    ax_f.text(0.96, y, scale, fontsize=9, style='italic',
              va='center', ha='right', color='#444',
              transform=ax_f.transAxes, zorder=3,
              bbox=dict(boxstyle='round,pad=0.2',
                        facecolor='white', alpha=0.8,
                        edgecolor='#999', linewidth=0.5))

# Footer note
ax_f.text(0.5, 0.06,
          "Rainfall variables calibrated on flood-year labels;\n"
          "state variables calibrated on flood-day windows.",
          fontsize=8.5, ha='center', va='bottom',
          style='italic', color='#666',
          transform=ax_f.transAxes,
          bbox=dict(boxstyle='round,pad=0.35', facecolor='#f7f7f7',
                    edgecolor='#ccc', linewidth=0.7, alpha=0.9))

# ==========================================================
# Suptitle
# ==========================================================
#fig.suptitle(
#    "Figure 6: Machine-learning threshold-variable evaluation "
#    "for flood identification, Shweli–Ayeyarwady basin",
#    fontsize=15, fontweight='bold', y=0.985)

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig06_Threshold_Evaluation_v2.png"
out_pdf = FIG_DIR / "Fig06_Threshold_Evaluation_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# ==========================================================
# Summary table to console
# ==========================================================
print("\n=== Annual scale (flood-year labels) ===")
print(annual_thr[["variable","AUC","threshold_youden",
                  "POD_youden","FAR_youden"]].to_string(index=False))

print("\n=== Daily scale (flood-day labels) ===")
for k, v in daily_roc.items():
    print(f"  {k:<16s} AUC={v['auc']:.3f}  POD={v['pod']:.3f}  "
          f"FAR={v['far']:.3f}  thr={v['opt_thr']:.3f}")

print("\n=== Final operating thresholds ===")
for k, v in final_thr.items():
    print(f"  {k:<12s} = {v}")

plt.show()
```

    Loading prep outputs...
      Annual thresholds : 11 vars
      Daily thresholds  : 5 vars
      RF importance     : 4 features
      Final thresholds  : {'Rx1day_Stn': 41.124201142857146, 'Rx5day_Stn': 124.17829128571428, 'SM': 0.3227272033691406, 'SWRI': 3.101592151984086}
      Rx1day_Stn        AUC=0.516  POD=0.18  FAR=0.12  opt=23.235
      Rx5day_Stn        AUC=0.604  POD=0.54  FAR=0.35  opt=52.969
      Rx5day_CHIRPS     AUC=0.595  POD=0.58  FAR=0.43  opt=49.731
      SM                AUC=0.810  POD=0.92  FAR=0.37  opt=0.323
      SWRI              AUC=0.551  POD=1.00  FAR=0.89  opt=3.102
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig06_Threshold_Evaluation_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig06_Threshold_Evaluation_v2.pdf
    
    === Annual scale (flood-year labels) ===
             variable      AUC  threshold_youden  POD_youden  FAR_youden
      SWRI_annual_max 0.665714         26.947857         0.8    0.371429
       SWRI_JJAS_mean 0.628571          5.168903         0.6    0.200000
             R95p_Stn 0.580000         11.000000         0.4    0.171429
         SM_JJAS_mean 0.574286          0.340966         0.7    0.485714
           Rx1day_Stn 0.557143         41.124201         0.9    0.685714
             R99p_Stn 0.537143          3.000000         0.3    0.228571
        Rx1day_CHIRPS 0.528571         73.878670         0.2    0.000000
           Rx5day_Stn 0.471429        124.178291         0.4    0.200000
        Rx5day_CHIRPS 0.434286        135.509841         0.3    0.114286
    Rx5day_20kmBuffer 0.422857        167.939581         0.3    0.085714
        SM_annual_max 0.377143          0.374625         1.0    0.914286
    
    === Daily scale (flood-day labels) ===
      Rx1day_Stn       AUC=0.516  POD=0.185  FAR=0.115  thr=23.235
      Rx5day_Stn       AUC=0.604  POD=0.538  FAR=0.346  thr=52.969
      Rx5day_CHIRPS    AUC=0.595  POD=0.585  FAR=0.428  thr=49.731
      SM               AUC=0.810  POD=0.917  FAR=0.372  thr=0.323
      SWRI             AUC=0.551  POD=1.000  FAR=0.886  thr=3.102
    
    === Final operating thresholds ===
      Rx1day_Stn   = 41.124201142857146
      Rx5day_Stn   = 124.17829128571428
      SM           = 0.3227272033691406
      SWRI         = 3.101592151984086
    


    
![png](output_23_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 7 — Selected detected flood events and full 1981–2025 trigger catalog
=========================================================================
Result figure (workflow schematic moved to Supplementary — Fig. S-a).
Three panels:
  (a) Selected Prolonged event
  (b) Selected Short/Flash event
  (c) Full trigger record 1981–2025 with detected-event bands

Reads:
    prep_revised_v2/fig07/fig07_trigger_daily.csv
    prep_revised_v2/events/event_catalog_full.csv
    prep_revised_v2/thresholds/final_operational_thresholds.json

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig07_Events_Catalog_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
print("Loading prep outputs...")
trig = pd.read_csv(PREP/"fig07"/"fig07_trigger_daily.csv",
                   parse_dates=["date"], index_col="date").sort_index()
ev   = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                   parse_dates=["start_date","end_date"])
thr  = json.load(open(PREP/"thresholds"/"final_operational_thresholds.json"))

T_RX5  = float(trig["T_RX5"].iloc[0])
T_SM   = float(trig["T_SM"].iloc[0])
T_SWRI = float(trig["T_SWRI"].iloc[0])

print(f"  Thresholds: Rx5 ≥ {T_RX5:.1f} mm | SM ≥ {T_SM:.3f} | SWRI ≥ {T_SWRI:.3f}")
print(f"  Trigger days: {int(trig['trigger'].sum())} / {len(trig)}")
print(f"  Events detected: {len(ev)}")
for c in ["Flash","Short","Prolonged"]:
    n = int((ev["category"] == c).sum())
    print(f"    {c:<10s}: {n}")

# -----------------------------------------------------------------------------
# Pick selected events (carried forward to WRF simulation)
# -----------------------------------------------------------------------------
prol = ev[ev["category"] == "Prolonged"].copy()
if not prol.empty:
    prol_2010 = prol[prol["start_date"].dt.year == 2010]
    ex_prol = prol_2010.iloc[0] if not prol_2010.empty else \
              prol.sort_values("duration_days", ascending=False).iloc[0]
else:
    ex_prol = ev.sort_values("duration_days", ascending=False).iloc[0]

short_pool = ev[ev["category"].isin(["Short","Flash"])].copy()
if not short_pool.empty:
    short_2023 = short_pool[short_pool["start_date"].dt.year == 2023]
    ex_short = short_2023.iloc[0] if not short_2023.empty else \
               short_pool.sort_values("start_date", ascending=False).iloc[0]
else:
    ex_short = ev.iloc[0]

print(f"\n  Selected Prolonged: {ex_prol['start_date'].date()} → "
      f"{ex_prol['end_date'].date()} ({ex_prol['duration_days']} d)")
print(f"  Selected {ex_short['category']}: {ex_short['start_date'].date()} → "
      f"{ex_short['end_date'].date()} ({ex_short['duration_days']} d)")

# -----------------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------------
COL = {
    "P":         "#1f4e8c",
    "Rx5":       "#00A087",
    "SM":        "#7E6148",
    "SWRI":      "#E64B35",
    "trigger":   "#E9C46A",
    "Flash":     "#E66101",
    "Short":     "#FDB863",
    "Prolonged": "#B2182B",
}

# -----------------------------------------------------------------------------
# Style — larger fonts for EVERYTHING (ticks, legend, labels, not just titles)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 16,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 7, 'ytick.major.size': 7,
    'legend.fontsize': 14, 'legend.title_fontsize': 14,
    'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 11),dpi=500)
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.0, 0.9],
    hspace=0.60, wspace=0.24,
    left=0.06, right=0.975, top=0.935, bottom=0.07,
)

def add_panel_letter(ax, letter):
    ax.text(0.018, 0.972, letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                      edgecolor='black', linewidth=1.2, alpha=0.0),
            zorder=100)

# ==========================================================
# draw_event helper — no "Example" in title, only "Selected"
# ==========================================================
def draw_event(ax, event_row, panel_letter, title_prefix):
    s, e = event_row["start_date"], event_row["end_date"]
    dur  = int(event_row["duration_days"])
    cat  = event_row["category"]

    w0 = s - pd.Timedelta(days=15)
    w1 = e + pd.Timedelta(days=10)
    w  = trig.loc[w0:w1]
    if len(w) == 0:
        ax.text(0.5, 0.5, "No data for window", ha='center', va='center',
                transform=ax.transAxes)
        return

    ax.bar(w.index, w["P_stn"], width=0.9, color=COL["P"],
           alpha=0.45, edgecolor="black", linewidth=0.4,
           label="Daily P (mm)", zorder=2)
    ax.plot(w.index, w["Rx5_stn"], color=COL["Rx5"], lw=2.8,
            marker='o', ms=5, markeredgecolor='black', markeredgewidth=0.6,
            label="Rx5 (mm)", zorder=4)
    ax.plot(w.index, w["SWRI"], color=COL["SWRI"], lw=2.8,
            ls='--', marker='s', ms=5, markeredgecolor='black',
            markeredgewidth=0.5, label="SWRI", zorder=4)

    ax.axhline(T_RX5,  color=COL["Rx5"],  ls=':', lw=1.9,
               label=f"$T_{{Rx5}}$ = {T_RX5:.0f} mm", zorder=3)
    ax.axhline(T_SWRI, color=COL["SWRI"], ls=':', lw=1.9,
               label=f"$T_{{SWRI}}$ = {T_SWRI:.2f}", zorder=3)

    ax.axvspan(s, e, color=COL[cat], alpha=0.15, zorder=1)
    for td in w[w["trigger"] == 1].index:
        ax.axvline(td, color=COL["trigger"], lw=0.9, alpha=0.55, zorder=1)

    ax2 = ax.twinx()
    ax2.plot(w.index, w["SM"], color=COL["SM"], lw=2.6,
             marker='^', ms=5.5, markeredgecolor='black', markeredgewidth=0.6,
             label=r"SM (m$^3$ m$^{-3}$)", zorder=5)
    ax2.axhline(T_SM, color=COL["SM"], ls=':', lw=1.9,
                label=f"$T_{{SM}}$ = {T_SM:.3f}", zorder=3)

    ax2.set_ylabel(r"Soil moisture (m$^3$ m$^{-3}$)", color=COL["SM"],
                    fontweight='bold', fontsize=16)
    ax2.tick_params(axis='y', labelcolor=COL["SM"], labelsize=15)
    ax2.set_ylim(T_SM - 0.10, 0.55)

    ax.set_ylabel("Rainfall / SWRI (mm)", fontweight='bold', fontsize=17)
    y_max = max(w["Rx5_stn"].max(), w["SWRI"].max(), T_RX5) * 1.15
    ax.set_ylim(0, y_max)

    ax.set_xlim(w0, w1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0,
             ha='center', fontsize=14)

    # Title — "Selected" only, no "Example"
    ax.set_title(f"{title_prefix}  "
                 f"[{s.strftime('%d %b %Y')} → {e.strftime('%d %b %Y')}, "
                 f"{dur} d]",
                 loc='left', pad=12, fontsize=16.5)
    add_panel_letter(ax, panel_letter)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ev_patch   = Patch(facecolor=COL[cat], alpha=0.30,
                       edgecolor=COL[cat], linewidth=1.6,
                       label=f"{cat} window ({dur} d)")
    trig_patch = Patch(facecolor=COL["trigger"], alpha=0.6,
                       label="Trigger day")
    handles = h1 + h2 + [ev_patch, trig_patch]
    labels  = l1 + l2 + [f"{cat} window ({dur} d)", "Trigger day"]

    ax.legend(handles=handles, labels=labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=13.5, framealpha=0.95, edgecolor='0.3',
              columnspacing=1.4, handlelength=2.0, handletextpad=0.7)

    ax.grid(ls=':', alpha=0.35, zorder=0)

# ==========================================================
# (a) Selected Prolonged event
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0])
draw_event(ax_a, ex_prol, 'a', "Selected Prolonged event")

# ==========================================================
# (b) Selected Short / Flash event
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1])
draw_event(ax_b, ex_short, 'b', f"Selected {ex_short['category']} event")

# ==========================================================
# (c) Full 1981–2025 record
# ==========================================================
ax_c = fig.add_subplot(gs[1, :])

ax_c.plot(trig.index, trig["Rx5_stn"], color=COL["Rx5"],
          lw=0.6, alpha=0.85, zorder=2, label="Daily Rx5 (mm)")
ax_c.axhline(T_RX5, color=COL["Rx5"], ls=':', lw=2.0, zorder=3,
             label=f"$T_{{Rx5}}$ = {T_RX5:.1f} mm")

for _, r in ev.iterrows():
    ax_c.axvspan(r["start_date"], r["end_date"],
                 color=COL[r["category"]], alpha=0.55, zorder=1)

FLOOD_YEARS = [1991,1998,2004,2010,2015,2018,2019,2020,2023,2024]
for yr in FLOOD_YEARS:
    ax_c.axvline(pd.Timestamp(f"{yr}-06-01"), color="black",
                 ls='--', lw=0.7, alpha=0.35, zorder=0)

ax_c.set_xlim(trig.index.min(), trig.index.max())
ax_c.set_ylim(0, np.nanmax(trig["Rx5_stn"]) * 1.08)
ax_c.set_xlabel("Year", fontweight='bold', fontsize=17)
ax_c.set_ylabel("Daily Rx5 (mm)", fontweight='bold', fontsize=17)
ax_c.set_title(f"(c) Full trigger record 1981–2025 — "
               f"{len(ev)} events detected "
               f"(Flash {int((ev['category']=='Flash').sum())}, "
               f"Short {int((ev['category']=='Short').sum())}, "
               f"Prolonged {int((ev['category']=='Prolonged').sum())})",
               loc='left', pad=12, fontsize=17)
ax_c.grid(axis='y', ls=':', alpha=0.35)
add_panel_letter(ax_c, 'c')

legend_c = [
    Line2D([0], [0], color=COL["Rx5"], lw=1.0, label="Daily Rx5"),
    Line2D([0], [0], color=COL["Rx5"], lw=2.0, ls=':',
           label=f"$T_{{Rx5}}$ = {T_RX5:.1f} mm"),
    Patch(facecolor=COL["Flash"], alpha=0.7, label="Flash"),
    Patch(facecolor=COL["Short"], alpha=0.7, label="Short"),
    Patch(facecolor=COL["Prolonged"], alpha=0.7, label="Prolonged"),
    Line2D([0], [0], color="black", ls='--', lw=1.0, alpha=0.5,
           label="Documented flood year"),
]
ax_c.legend(handles=legend_c, loc='upper right', ncol=6, fontsize=14,
            framealpha=0.95, edgecolor='0.3', columnspacing=1.5,
            handlelength=2.0)

# ==========================================================
# Suptitle
# ==========================================================
#fig.suptitle(
 #   "Figure 7: Selected flood events and full 1981–2025 trigger "
 #   "catalog, Shweli–Ayeyarwady basin",
 #   fontsize=18, fontweight='bold', y=0.978)

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig07_Events_Catalog_v2.png"
out_pdf = FIG_DIR / "Fig07_Events_Catalog_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# ==========================================================
# Numeric summary
# ==========================================================
print("\n=== Event catalog summary ===")
print(f"  Total events          : {len(ev)}")
print(f"  Flash (≤2 d)          : {int((ev['category']=='Flash').sum())}")
print(f"  Short (3–5 d)         : {int((ev['category']=='Short').sum())}")
print(f"  Prolonged (>5 d)      : {int((ev['category']=='Prolonged').sum())}")
print(f"  Matched to news (±3 d): {int(ev['matched'].sum())} / 12 documented")

print(f"\n=== Selected events shown ===")
print(f"  (a) Prolonged : {ex_prol['start_date'].date()} → "
      f"{ex_prol['end_date'].date()}  ({ex_prol['duration_days']} d)  "
      f"peak SWRI = {ex_prol['peak_SWRI']:.2f}")
print(f"  (b) {ex_short['category']:<8s}: {ex_short['start_date'].date()} → "
      f"{ex_short['end_date'].date()}  ({ex_short['duration_days']} d)  "
      f"peak SWRI = {ex_short['peak_SWRI']:.2f}")

plt.show()
```

    Loading prep outputs...
      Thresholds: Rx5 ≥ 124.2 mm | SM ≥ 0.323 | SWRI ≥ 3.102
      Trigger days: 25 / 16436
      Events detected: 11
        Flash     : 0
        Short     : 5
        Prolonged : 6
    
      Selected Prolonged: 2010-10-08 → 2010-10-17 (10 d)
      Selected Short: 2023-10-12 → 2023-10-14 (3 d)
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig07_Events_Catalog_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig07_Events_Catalog_v2.pdf
    
    === Event catalog summary ===
      Total events          : 11
      Flash (≤2 d)          : 0
      Short (3–5 d)         : 5
      Prolonged (>5 d)      : 6
      Matched to news (±3 d): 1 / 12 documented
    
    === Selected events shown ===
      (a) Prolonged : 2010-10-08 → 2010-10-17  (10 d)  peak SWRI = 59.20
      (b) Short   : 2023-10-12 → 2023-10-14  (3 d)  peak SWRI = 39.18
    


    
![png](output_25_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 8 — Flood-event catalog and category statistics
=========================================================================
Nature-journal style. 7 panels across 3 rows.
Optimized layout, large fonts, zero overlaps.

Data-driven candidate event dates are printed to the console BEFORE
plotting, so they can be checked against the documented record.
Panel letters live only inside the subtitles — no floating (a)/(b) boxes.
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
print("Loading prep outputs...")
ev       = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                       parse_dates=["start_date", "end_date"])
yearly   = pd.read_csv(PREP/"fig08"/"fig08_yearly_counts.csv")
cat_summ = pd.read_csv(PREP/"tables"/"table06_category_summary.csv")

ev["year"]  = ev["start_date"].dt.year
ev["month"] = ev["start_date"].dt.month

# ---- Category order: keep only categories actually present ----
ALL_CATS  = ["Flash", "Short", "Prolonged"]
CAT_ORDER = [c for c in ALL_CATS if (ev["category"] == c).any()]
COLORS = {
    "Flash":     "#E66101",
    "Short":     "#FDB863",
    "Prolonged": "#B2182B",
}

n_flash     = int((ev["category"] == "Flash").sum())
n_short     = int((ev["category"] == "Short").sum())
n_prolonged = int((ev["category"] == "Prolonged").sum())
n_matched   = int(ev["matched"].sum())

print(f"  Total events : {len(ev)}")
print(f"  Flash        : {n_flash}")
print(f"  Short        : {n_short}")
print(f"  Prolonged    : {n_prolonged}")
print(f"  Matched news : {n_matched}")

# =============================================================================
# DATA-DRIVEN CANDIDATE EVENT DATES  ← printed to console for verification
# =============================================================================
print("\n" + "=" * 78)
print(" DATA-DRIVEN CANDIDATE EVENT DATES (from event_catalog_full.csv)")
print(" Rank the top events of each year by peak SWRI; here we list the")
print(" single strongest event per year, plus every documented-match flag.")
print("=" * 78)

# 1) Strongest event per year (by peak_SWRI)
strongest = (ev.sort_values("peak_SWRI", ascending=False)
               .groupby("year", as_index=False)
               .first()
               .sort_values("year"))

print("\n--- Strongest detected event per year (peak SWRI) ---")
print(f"{'Year':<6}{'Start':<13}{'End':<13}{'Cat':<12}"
      f"{'Dur(d)':<8}{'PeakSWRI':<10}{'Matched'}")
print("-" * 78)
for _, r in strongest.iterrows():
    print(f"{int(r['year']):<6}"
          f"{r['start_date'].strftime('%Y-%m-%d'):<13}"
          f"{r['end_date'].strftime('%Y-%m-%d'):<13}"
          f"{r['category']:<12}"
          f"{int(r['duration_days']):<8}"
          f"{float(r['peak_SWRI']):<10.2f}"
          f"{'YES' if int(r['matched']) else 'no'}")

# 2) All events flagged matched (these are the ones the ML found
#    within ±3 days of a documented flood — treat as verified candidates)
matched_ev = ev[ev["matched"] == 1].sort_values("start_date")
print("\n--- Events flagged 'matched' in catalog ---")
if len(matched_ev):
    print(f"{'Start':<13}{'End':<13}{'Cat':<12}{'Dur(d)':<8}{'PeakSWRI'}")
    print("-" * 78)
    for _, r in matched_ev.iterrows():
        print(f"{r['start_date'].strftime('%Y-%m-%d'):<13}"
              f"{r['end_date'].strftime('%Y-%m-%d'):<13}"
              f"{r['category']:<12}"
              f"{int(r['duration_days']):<8}"
              f"{float(r['peak_SWRI']):.2f}")
else:
    print("  (none flagged)")

# 3) Longest event per year
longest = (ev.sort_values("duration_days", ascending=False)
             .groupby("year", as_index=False)
             .first()
             .sort_values("year"))
print("\n--- Longest detected event per year (duration) ---")
print(f"{'Year':<6}{'Start':<13}{'Cat':<12}{'Dur(d)':<8}{'PeakSWRI'}")
print("-" * 78)
for _, r in longest.iterrows():
    print(f"{int(r['year']):<6}"
          f"{r['start_date'].strftime('%Y-%m-%d'):<13}"
          f"{r['category']:<12}"
          f"{int(r['duration_days']):<8}"
          f"{float(r['peak_SWRI']):.2f}")

# 4) Suggested candidate dates (top-3 by peak SWRI per year)
top3 = (ev.sort_values("peak_SWRI", ascending=False)
          .groupby("year")
          .head(3)
          .sort_values(["year", "peak_SWRI"], ascending=[True, False]))

print("\n--- Candidate dates: top-3 events per year (peak SWRI) ---")
print(f"{'Year':<6}{'Start':<13}{'Cat':<12}{'Dur(d)':<8}{'PeakSWRI':<10}{'Matched'}")
print("-" * 78)
for _, r in top3.iterrows():
    print(f"{int(r['year']):<6}"
          f"{r['start_date'].strftime('%Y-%m-%d'):<13}"
          f"{r['category']:<12}"
          f"{int(r['duration_days']):<8}"
          f"{float(r['peak_SWRI']):<10.2f}"
          f"{'YES' if int(r['matched']) else 'no'}")

print("=" * 78 + "\n")

# =============================================================================
# USER-EDITABLE documented flood list (for panel g / shading)
# =============================================================================
# These are the *documented* news events you want to check against.
# The data-driven candidates above should be cross-checked against this list.
DOCUMENTED_DATES = {
    "1991-07-10": "1991 flood",
    "1998-07-28": "1998 flood",
    "2004-07-18": "2004 flood",
    "2010-09-01": "2010 flood",
    "2015-07-16": "2015 flood",
    "2018-07-09": "2018 flood",
    "2019-07-10": "2019 flood",
    "2020-07-02": "2020 flood",
    "2023-05-14": "2023 Cyclone Mocha",
    "2023-08-07": "2023 Aug flood",
    "2024-06-30": "2024 Jun flood",
    "2024-09-09": "2024 Sep flood",
}
FLOOD_YEARS = sorted({int(d[:4]) for d in DOCUMENTED_DATES})

# -----------------------------------------------------------------------------
# Nature-style — LARGER, CLEANER fonts for EVERYTHING
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 17,
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 18, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 15, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure Layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 14),dpi=500)
gs = fig.add_gridspec(
    3, 3,
    height_ratios=[1.2, 1.0, 1.0],
    hspace=0.45, wspace=0.32,
    left=0.055, right=0.975, top=0.935, bottom=0.06,
)

# ==========================================================
# (a) Full event timeline
# ==========================================================
ax_a = fig.add_subplot(gs[0, :])

for yr in FLOOD_YEARS:
    ax_a.axvspan(pd.Timestamp(f"{yr}-01-01"),
                 pd.Timestamp(f"{yr}-12-31"),
                 color="#d32f2f", alpha=0.06, zorder=0)

for cat in CAT_ORDER:
    sub = ev[ev["category"] == cat]
    sizes = 80 + 30 * np.clip(sub["duration_days"].values, 0, 60)
    ax_a.scatter(sub["start_date"], sub["peak_SWRI"],
                 s=sizes, c=COLORS[cat], edgecolor="black",
                 linewidth=0.8, alpha=0.85, zorder=3,
                 label=f"{cat} (n={len(sub)})")

matched = ev[ev["matched"] == 1]
if len(matched) > 0:
    ax_a.scatter(matched["start_date"], matched["peak_SWRI"],
                 s=350, facecolor="none", edgecolor="#FFB300",
                 linewidth=3.0, zorder=4,
                 label=f"Documented match (n={len(matched)})")

ax_a.set_xlim(pd.Timestamp("1980-01-01"), pd.Timestamp("2026-06-30"))
ax_a.set_ylim(0, max(ev["peak_SWRI"].max() * 1.20, 5))
ax_a.set_xlabel("Year")
ax_a.set_ylabel("Peak SWRI")
ax_a.set_title(f"(a) Event timeline 1981–2025 — {len(ev)} events detected (marker size ∝ duration)",
               loc="left", pad=15)
ax_a.grid(ls=":", alpha=0.4)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Legend — LOWER LEFT
h, l = ax_a.get_legend_handles_labels()
size_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
           markeredgecolor='black', markersize=7, alpha=0.8,
           linestyle='None', label='Short (<5 d)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
           markeredgecolor='black', markersize=14, alpha=0.8,
           linestyle='None', label='Long (>20 d)'),
]
ax_a.legend(h + size_legend, l + [s.get_label() for s in size_legend],
            loc="upper left", ncol=2, fontsize=13, framealpha=0.95,
            edgecolor='0.3', borderpad=0.6, labelspacing=0.5, columnspacing=1.5)

# ==========================================================
# (b) Annual event counts
# ==========================================================
ax_b = fig.add_subplot(gs[1, 0])

bottom = np.zeros(len(yearly))
for cat in CAT_ORDER:
    if cat not in yearly.columns:
        continue
    ax_b.bar(yearly["year"], yearly[cat], bottom=bottom, width=0.85,
             color=COLORS[cat], edgecolor="black", linewidth=0.6,
             label=cat, zorder=2)
    bottom += yearly[cat].values

ax_b.plot(yearly["year"], yearly["total"], color="black", lw=1.8,
          marker="o", ms=5, markeredgecolor="white", markeredgewidth=0.5,
          zorder=5, label="Total")

mean_total = float(yearly["total"].mean())
ax_b.axhline(mean_total, color="gray", ls="--", lw=1.5, zorder=1,
             label=f"Mean = {mean_total:.1f}")

ax_b.set_xlabel("Year")
ax_b.set_ylabel("Events per year")
ax_b.set_xlim(1979, 2027)
ax_b.set_ylim(0, yearly["total"].max() * 1.25)
ax_b.set_title("(b) Annual event counts", loc="left", pad=12)
ax_b.grid(axis="y", ls=":", alpha=0.4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.legend(loc="upper right", fontsize=12, ncol=2, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (c) Duration distribution
# ==========================================================
ax_c = fig.add_subplot(gs[1, 1])

dur_max = int(min(ev["duration_days"].max(), 65))
bins = np.arange(0.5, dur_max + 2, 2)

for cat in CAT_ORDER:
    sub = ev[ev["category"] == cat]
    ax_c.hist(sub["duration_days"], bins=bins, color=COLORS[cat],
              edgecolor="black", linewidth=0.7, alpha=0.85,
              label=f"{cat} (n={len(sub)})", zorder=2)

ymax_hist = ax_c.get_ylim()[1]
ax_c.axvline(2.5, color="black", ls="--", lw=1.8, zorder=3)
ax_c.axvline(5.5, color="black", ls="--", lw=1.8, zorder=3)

cat_pos = {"Flash": 1.5, "Short": 4.0, "Prolonged": 18.0}
for cat in CAT_ORDER:
    ax_c.text(cat_pos[cat], ymax_hist * 0.90, cat,
              fontsize=14, fontweight='bold', ha="center",
              color=COLORS[cat],
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS[cat], linewidth=1.2, alpha=0.95))

ax_c.set_xlabel("Event duration (days)")
ax_c.set_ylabel("Number of events")
ax_c.set_title("(c) Duration distribution", loc="left", pad=12)
ax_c.grid(axis="y", ls=":", alpha=0.4)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.legend(loc="upper right", fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (d) Seasonal timing
# ==========================================================
ax_d = fig.add_subplot(gs[1, 2])

monthly = (ev.groupby(["month", "category"]).size()
             .unstack(fill_value=0)
             .reindex(columns=CAT_ORDER, fill_value=0)
             .reindex(range(1, 13), fill_value=0))

n_cat = max(len(CAT_ORDER), 1)
w = 0.8 / n_cat
x = np.arange(1, 13)
for i, cat in enumerate(CAT_ORDER):
    ax_d.bar(x + (i - (n_cat - 1) / 2) * w,
             monthly[cat].values, width=w,
             color=COLORS[cat], edgecolor="black", linewidth=0.6,
             label=cat, zorder=2)

ax_d.axvspan(5.5, 9.5, color="#a5d6a7", alpha=0.20, zorder=0)
ax_d.text(7.5, ax_d.get_ylim()[1] * 0.90, "JJAS monsoon",
          ha='center', va='top', fontsize=13, style='italic',
          color='#2e7d32', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#2e7d32', linewidth=1.0, alpha=0.95))

ax_d.set_xlabel("Month of onset")
ax_d.set_ylabel("Number of events")
ax_d.set_xticks(x)
ax_d.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                      fontsize=13, fontweight='bold')
ax_d.set_title("(d) Seasonal onset timing", loc="left", pad=12)
ax_d.grid(axis="y", ls=":", alpha=0.4)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.legend(loc="upper left", fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (e) Peak SWRI distribution
# ==========================================================
ax_e = fig.add_subplot(gs[2, 0])

present_cats = [c for c in CAT_ORDER
                if ev[ev["category"] == c]["peak_SWRI"].dropna().shape[0] >= 2]
positions_all = {cat: i + 1 for i, cat in enumerate(CAT_ORDER)}

if present_cats:
    data      = [ev[ev["category"] == c]["peak_SWRI"].dropna().values for c in present_cats]
    positions = [positions_all[c] for c in present_cats]

    vp = ax_e.violinplot(data, positions=positions, widths=0.6,
                         showmeans=False, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS[present_cats[i]])
        body.set_edgecolor("black")
        body.set_linewidth(1.5)
        body.set_alpha(0.75)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(3.0)

    rng = np.random.RandomState(42)
    for i, cat in enumerate(present_cats):
        vals = ev[ev["category"] == cat]["peak_SWRI"].dropna().values
        jitter = rng.normal(0, 0.05, len(vals))
        ax_e.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                     s=50, color="white", edgecolor="black",
                     linewidth=1.0, alpha=0.85, zorder=5)
        m = float(np.median(vals))
        ax_e.scatter([positions[i]], [m], s=120, marker='D',
                     color='white', edgecolor='black', linewidth=2.0, zorder=7)

ax_e.set_xticks([positions_all[c] for c in CAT_ORDER])
ax_e.set_xticklabels(
    [f"{c}\n(n={int((ev['category']==c).sum())})" for c in CAT_ORDER],
    fontsize=14, fontweight='bold')
ax_e.set_ylabel("Peak SWRI")
ax_e.set_title("(e) Peak SWRI by category", loc="left", pad=12)
ax_e.grid(axis="y", ls=":", alpha=0.4)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# ==========================================================
# (f) Category proportions (donut)
# ==========================================================
ax_f = fig.add_subplot(gs[2, 1])

counts = [int((ev["category"] == c).sum()) for c in CAT_ORDER]
total  = sum(counts)

wedges, texts, autotexts = ax_f.pie(
    counts, labels=None,
    colors=[COLORS[c] for c in CAT_ORDER],
    autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100))})",
    startangle=90, counterclock=False,
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=3.0),
    textprops=dict(fontsize=15, fontweight='bold', color='white'),
    pctdistance=0.78,
)

ax_f.text(0, 0.10, "Total", ha='center', va='center',
          fontsize=14, fontweight='bold', color='#333')
ax_f.text(0, -0.10, f"{total}", ha='center', va='center',
          fontsize=26, fontweight='bold', color='black')
ax_f.text(0, -0.35, "events", ha='center', va='center',
          fontsize=13, color='#555')

ax_f.legend(
    labels=[f"{c} (n={n})" for c, n in zip(CAT_ORDER, counts)],
    loc='upper center', bbox_to_anchor=(0.5, -0.05),
    ncol=len(CAT_ORDER), fontsize=13, frameon=False,
)
ax_f.set_title("(f) Category proportions", loc="left", pad=12)

# ==========================================================
# (g) Documented-event coverage (uses DOCUMENTED_DATES above)
# ==========================================================
ax_g = fig.add_subplot(gs[2, 2])

tol = pd.Timedelta(days=3)
coverage_rows = []
for ds, note in DOCUMENTED_DATES.items():
    date = pd.to_datetime(ds)
    hit = ev[(ev["start_date"] <= date + tol) & (ev["end_date"] >= date - tol)]
    if len(hit) > 0:
        r = hit.iloc[0]
        coverage_rows.append(dict(date=date, matched=1,
                                  category=r["category"],
                                  duration=int(r["duration_days"]),
                                  note=note))
    else:
        coverage_rows.append(dict(date=date, matched=0, category="—",
                                  duration=0, note=note))

cov = (pd.DataFrame(coverage_rows).sort_values("date").reset_index(drop=True))
y_pos = np.arange(len(cov))

for i, r in cov.iterrows():
    if r["matched"]:
        ax_g.barh(i, r["duration"], color=COLORS[r["category"]],
                  edgecolor="black", linewidth=1.0, height=0.7)
        ax_g.text(r["duration"] + 0.5, i,
                  f"{r['category']} ({r['duration']} d)",
                  va='center', ha='left', fontsize=13, fontweight='bold',
                  color=COLORS[r["category"]])
    else:
        ax_g.barh(i, 2.0, color="#d9d9d9", edgecolor="black",
                  linewidth=1.0, height=0.7, alpha=0.7)
        ax_g.text(1.0, i, "NOT MATCHED", va='center', ha='center',
                  fontsize=13, fontweight='bold', color="#b71c1c")

ax_g.yaxis.tick_right()
ax_g.yaxis.set_label_position("right")
ax_g.set_yticks(y_pos)
ax_g.set_yticklabels([d.strftime("%Y-%m-%d") for d in cov["date"]],
                     fontsize=13, fontweight='bold')
ax_g.set_xlabel("Detected event duration (days)")
ax_g.set_xlim(0, max(cov["duration"].max() * 1.5, 25))
ax_g.set_title(f"(g) Documented coverage:\n{int(cov['matched'].sum())}/{len(cov)} matched",
               loc="left", pad=12)
ax_g.grid(axis="x", ls=":", alpha=0.4)
ax_g.spines['top'].set_visible(False)
ax_g.spines['left'].set_visible(False)
ax_g.invert_yaxis()

# ==========================================================
# Footer summary (hidden box kept as in original)
# ==========================================================
fig.text(
    0.5, 0.015,
    f"Detected: {len(ev)} events  ·  "
    f"Flash = {n_flash}  ·  Short = {n_short}  ·  Prolonged = {n_prolonged}  ·  "
    f"Documented matches = {n_matched} / {len(DOCUMENTED_DATES)}  ·  "
    f"Mean = {mean_total:.1f} events yr$^{{-1}}$",
    ha='center', va='bottom', fontsize=15, style='italic', alpha=0,
    color='#333',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
              edgecolor='#999', linewidth=1.0, alpha=0.0))

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig08_Event_Catalog_v2.png"
out_pdf = FIG_DIR / "Fig08_Event_Catalog_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")
plt.show()
```

    Loading prep outputs...
      Total events : 11
      Flash        : 0
      Short        : 5
      Prolonged    : 6
      Matched news : 1
    
    ==============================================================================
     DATA-DRIVEN CANDIDATE EVENT DATES (from event_catalog_full.csv)
     Rank the top events of each year by peak SWRI; here we list the
     single strongest event per year, plus every documented-match flag.
    ==============================================================================
    
    --- Strongest detected event per year (peak SWRI) ---
    Year  Start        End          Cat         Dur(d)  PeakSWRI  Matched
    ------------------------------------------------------------------------------
    1986  1986-10-08   1986-10-12   Short       5       33.04     no
    1989  1989-07-30   1989-08-16   Prolonged   18      17.11     no
    1990  1990-06-23   1990-08-03   Prolonged   42      22.77     no
    1992  1992-10-14   1992-10-16   Short       3       20.89     no
    2001  2001-05-31   2001-06-14   Prolonged   15      36.65     no
    2004  2004-05-18   2004-05-21   Short       4       31.75     no
    2006  2006-10-08   2006-10-11   Short       4       20.79     no
    2007  2007-05-18   2007-05-23   Prolonged   6       15.29     no
    2010  2010-10-08   2010-10-17   Prolonged   10      59.20     no
    2015  2015-07-19   2015-08-12   Prolonged   25      23.71     YES
    2023  2023-10-12   2023-10-14   Short       3       39.18     no
    
    --- Events flagged 'matched' in catalog ---
    Start        End          Cat         Dur(d)  PeakSWRI
    ------------------------------------------------------------------------------
    2015-07-19   2015-08-12   Prolonged   25      23.71
    
    --- Longest detected event per year (duration) ---
    Year  Start        Cat         Dur(d)  PeakSWRI
    ------------------------------------------------------------------------------
    1986  1986-10-08   Short       5       33.04
    1989  1989-07-30   Prolonged   18      17.11
    1990  1990-06-23   Prolonged   42      22.77
    1992  1992-10-14   Short       3       20.89
    2001  2001-05-31   Prolonged   15      36.65
    2004  2004-05-18   Short       4       31.75
    2006  2006-10-08   Short       4       20.79
    2007  2007-05-18   Prolonged   6       15.29
    2010  2010-10-08   Prolonged   10      59.20
    2015  2015-07-19   Prolonged   25      23.71
    2023  2023-10-12   Short       3       39.18
    
    --- Candidate dates: top-3 events per year (peak SWRI) ---
    Year  Start        Cat         Dur(d)  PeakSWRI  Matched
    ------------------------------------------------------------------------------
    1986  1986-10-08   Short       5       33.04     no
    1989  1989-07-30   Prolonged   18      17.11     no
    1990  1990-06-23   Prolonged   42      22.77     no
    1992  1992-10-14   Short       3       20.89     no
    2001  2001-05-31   Prolonged   15      36.65     no
    2004  2004-05-18   Short       4       31.75     no
    2006  2006-10-08   Short       4       20.79     no
    2007  2007-05-18   Prolonged   6       15.29     no
    2010  2010-10-08   Prolonged   10      59.20     no
    2015  2015-07-19   Prolonged   25      23.71     YES
    2023  2023-10-12   Short       3       39.18     no
    ==============================================================================
    
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig08_Event_Catalog_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig08_Event_Catalog_v2.pdf
    


    
![png](output_27_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 9 — Category-dependent flood-event dynamics
=========================================================================
Nature-journal style. 3x2 panels. Robust to missing categories
(e.g. no Flash events in the catalog).

Panels:
  (a) Normalized rainfall composites
  (b) Soil-moisture composites
  (c) SWRI composites
  (d) Duration vs peak SWRI scatter
  (e) Antecedent SM distribution by category (violin + strip)
  (f) Fast vs Prolonged comparison
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
print("Loading prep outputs...")
comp = pd.read_csv(PREP/"fig09"/"fig09_category_composites.csv")
fp   = pd.read_csv(PREP/"fig09"/"fig09_fast_vs_prolonged.csv")
ev   = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                   parse_dates=["start_date", "end_date"])

print(f"  Category composites: {len(comp)} rows")
print(f"  Fast vs Prolonged  : {len(fp)} rows")
print(f"  Events             : {len(ev)}")

# ---- Only keep categories actually present in the catalog ----
ALL_CATS  = ["Flash", "Short", "Prolonged"]
CAT_ORDER = [c for c in ALL_CATS if (ev["category"] == c).any()]
print(f"  Categories present : {CAT_ORDER}")

COLORS = {
    "Flash":     "#E66101",
    "Short":     "#FDB863",
    "Prolonged": "#B2182B",
    "Fast":      "#E66101",
}

# -----------------------------------------------------------------------------
# Nature-style — LARGE, CLEAN, unicode-safe
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 17,
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 18, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 15, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

#def add_panel_letter(ax, letter):
#    ax.text(0.018, 0.978, letter, transform=ax.transAxes,
 #           fontsize=22, fontweight='bold', va='top', ha='left',
  #          bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
  #                    edgecolor='black', linewidth=1.2, alpha=0.97),
   #         zorder=100)

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 15),dpi=500)
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 1.0, 1.0],
    hspace=0.42, wspace=0.22,
    left=0.065, right=0.975, top=0.925, bottom=0.06,
)

# ==========================================================
# (a) Rainfall composite
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0])

for cat in CAT_ORDER:
    sub = comp[comp["category"] == cat].sort_values("rel_day")
    if len(sub) == 0:
        continue
    ax_a.plot(sub["rel_day"], sub["P_mean"],
              color=COLORS[cat], lw=3.0,
              marker='o', ms=6, markeredgecolor='black',
              markeredgewidth=0.7,
              label=f"{cat} (n/day: {int(sub['n'].median())})",
              zorder=4)
    if cat == "Prolonged":
        ax_a.fill_between(sub["rel_day"],
                           sub["P_mean"] - (sub["P_p90"] - sub["P_mean"]).clip(lower=0),
                           sub["P_p90"],
                           color=COLORS[cat], alpha=0.15,
                           label=f"{cat} P90 envelope", zorder=2)

ax_a.axvspan(-0.5, 2.0, color='#f0f0f0', alpha=0.55, zorder=0)
ax_a.axvspan(2.0, 5.0, color='#f0f0f0', alpha=0.28, zorder=0)
ax_a.axvline(0, color='black', ls='--', lw=1.6, alpha=0.7, zorder=1)
ax_a.text(0, ax_a.get_ylim()[1], "onset",
          ha='center', va='top', fontsize=13, style='italic',
          rotation=90, color='#333')

ax_a.set_xlabel("Days relative to event onset")
ax_a.set_ylabel("Rainfall $P$ (mm)")
ax_a.set_title("(a) Rainfall composite by category", loc='left', pad=12)
ax_a.grid(ls=':', alpha=0.4)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
add_panel_letter(ax_a, 'a')
ax_a.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (b) Soil-moisture composite
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1])

for cat in CAT_ORDER:
    sub = comp[comp["category"] == cat].sort_values("rel_day")
    if len(sub) == 0:
        continue
    ax_b.plot(sub["rel_day"], sub["SM_mean"],
              color=COLORS[cat], lw=3.0,
              marker='s', ms=6, markeredgecolor='black',
              markeredgewidth=0.7,
              label=cat, zorder=4)
    if cat == "Prolonged":
        ax_b.fill_between(sub["rel_day"],
                           sub["SM_p10"],
                           sub["SM_mean"] + (sub["SM_mean"] - sub["SM_p10"]).clip(lower=0),
                           color=COLORS[cat], alpha=0.15,
                           label=f"{cat} P10–P90", zorder=2)

ax_b.axvspan(-0.5, 2.0, color='#f0f0f0', alpha=0.55, zorder=0)
ax_b.axvspan(2.0, 5.0, color='#f0f0f0', alpha=0.28, zorder=0)
ax_b.axvline(0, color='black', ls='--', lw=1.6, alpha=0.7, zorder=1)

ax_b.set_xlabel("Days relative to event onset")
ax_b.set_ylabel(r"Soil moisture (m$^3$ m$^{-3}$)")
ax_b.set_title("(b) Soil-moisture composite by category", loc='left', pad=12)
ax_b.grid(ls=':', alpha=0.4)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
add_panel_letter(ax_b, 'b')
ax_b.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (c) SWRI composite
# ==========================================================
ax_c = fig.add_subplot(gs[1, 0])

for cat in CAT_ORDER:
    sub = comp[comp["category"] == cat].sort_values("rel_day")
    if len(sub) == 0:
        continue
    ax_c.plot(sub["rel_day"], sub["SWRI_mean"],
              color=COLORS[cat], lw=3.2,
              marker='D', ms=6, markeredgecolor='black',
              markeredgewidth=0.7,
              label=cat, zorder=4)
    peak_idx = sub["SWRI_mean"].idxmax()
    ax_c.scatter([sub.loc[peak_idx, "rel_day"]],
                 [sub.loc[peak_idx, "SWRI_mean"]],
                 s=280, color=COLORS[cat],
                 edgecolor='black', linewidth=1.6, zorder=6)

ax_c.axvspan(-0.5, 2.0, color='#f0f0f0', alpha=0.55, zorder=0)
ax_c.axvspan(2.0, 5.0, color='#f0f0f0', alpha=0.28, zorder=0)
ax_c.axvline(0, color='black', ls='--', lw=1.6, alpha=0.7, zorder=1)

ax_c.set_xlabel("Days relative to event onset")
ax_c.set_ylabel("SWRI")
ax_c.set_title("(c) SWRI composite by category", loc='left', pad=12)
ax_c.grid(ls=':', alpha=0.4)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
add_panel_letter(ax_c, 'c')
ax_c.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (d) Duration vs peak SWRI
# ==========================================================
ax_d = fig.add_subplot(gs[1, 1])

for cat in CAT_ORDER:
    sub = ev[ev["category"] == cat]
    if len(sub) == 0:
        continue
    ax_d.scatter(sub["duration_days"], sub["peak_SWRI"],
                 s=140, color=COLORS[cat], edgecolor='black',
                 linewidth=1.0, alpha=0.85, marker='o',
                 label=f"{cat} (n={len(sub)})", zorder=4)

# Only draw boundaries for categories actually present
if "Flash" in CAT_ORDER or "Short" in CAT_ORDER:
    ax_d.axvline(2.5, color='black', ls='--', lw=1.6, alpha=0.55)
if "Short" in CAT_ORDER or "Prolonged" in CAT_ORDER:
    ax_d.axvline(5.5, color='black', ls='--', lw=1.6, alpha=0.55)

# Category labels only for present categories
cat_pos = {"Flash": 1.5, "Short": 4.0, "Prolonged": 18.0}
ypos = ax_d.get_ylim()[1] * 0.96
for cat in CAT_ORDER:
    ax_d.text(cat_pos[cat], ypos, cat,
              ha='center', va='top', fontsize=13, fontweight='bold',
              color=COLORS[cat],
              bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                        edgecolor=COLORS[cat], linewidth=1.2, alpha=0.95))

ax_d.set_xlabel("Event duration (days)")
ax_d.set_ylabel("Peak SWRI")
ax_d.set_title("(d) Duration vs peak SWRI", loc='left', pad=12)
ax_d.grid(ls=':', alpha=0.4)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
add_panel_letter(ax_d, 'd')
ax_d.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (e) Antecedent SM distribution — robust against empty categories
# ==========================================================
ax_e = fig.add_subplot(gs[2, 0])

positions_all = {cat: i + 1 for i, cat in enumerate(CAT_ORDER)}

# Categories with ≥ 2 finite values (violin needs KDE)
present_for_violin = [c for c in CAT_ORDER
                      if ev[ev["category"] == c]["SM_onset"].dropna().shape[0] >= 2]

if present_for_violin:
    data      = [ev[ev["category"] == c]["SM_onset"].dropna().values
                 for c in present_for_violin]
    positions = [positions_all[c] for c in present_for_violin]

    vp = ax_e.violinplot(data, positions=positions, widths=0.60,
                         showmeans=False, showmedians=True, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS[present_for_violin[i]])
        body.set_edgecolor("black")
        body.set_linewidth(1.5)
        body.set_alpha(0.75)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(3.0)

    rng = np.random.RandomState(42)
    for i, cat in enumerate(present_for_violin):
        vals = ev[ev["category"] == cat]["SM_onset"].dropna().values
        jitter = rng.normal(0, 0.05, len(vals))
        ax_e.scatter(np.full(len(vals), positions[i]) + jitter, vals,
                     s=55, color="white", edgecolor="black",
                     linewidth=1.0, alpha=0.85, zorder=5)
        m = float(np.mean(vals))
        ax_e.scatter([positions[i]], [m], marker='D', s=160,
                     color='white', edgecolor='black', linewidth=2.0,
                     zorder=8)
else:
    ax_e.text(0.5, 0.5, "Not enough data\nfor violin plot",
              ha='center', va='center', transform=ax_e.transAxes,
              fontsize=15, style='italic', color='#666')

# ML threshold
T_SM = 0.323
ax_e.axhline(T_SM, color='#d32f2f', ls='--', lw=2.2,
             label=f"$T_{{SM}}$ = {T_SM:.3f}", zorder=3)

ax_e.set_xticks([positions_all[c] for c in CAT_ORDER])
ax_e.set_xticklabels(
    [f"{c}\n(n={int((ev['category']==c).sum())})" for c in CAT_ORDER],
    fontsize=14, fontweight='bold')
ax_e.set_ylabel(r"Antecedent SM at onset (m$^3$ m$^{-3}$)")
ax_e.set_title("(e) Antecedent soil moisture by category", loc='left', pad=12)
ax_e.grid(axis='y', ls=':', alpha=0.4)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
add_panel_letter(ax_e, 'e')

legend_e = [
    Line2D([0], [0], color='#d32f2f', ls='--', lw=2.2,
           label=f"$T_{{SM}}$ = {T_SM:.3f}"),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor='black', markeredgewidth=1.6, markersize=12,
           linestyle='None', label='Category mean'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markeredgewidth=1.0, markersize=10,
           linestyle='None', label='Individual events'),
]
ax_e.legend(handles=legend_e, loc='lower right', fontsize=12,
            framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (f) Fast vs Prolonged grouped comparison
# ==========================================================
ax_f = fig.add_subplot(gs[2, 1])

fast_ev = ev[ev["category"].isin(["Flash", "Short"])]
prol_ev = ev[ev["category"] == "Prolonged"]

metrics = [
    ("Duration\n(days)",      "duration_days", None),
    ("Rain sum\n(mm)",        "rain_sum_mm",   None),
    ("Peak SWRI\n(/10)",      "peak_SWRI",     "div10"),
    ("SM onset\n(×10)",       "SM_onset",      "x10"),
]

fast_vals, prol_vals = [], []
for _, col, transform in metrics:
    if col not in ev.columns:
        fast_vals.append(0); prol_vals.append(0); continue
    f = fast_ev[col].dropna()
    p = prol_ev[col].dropna()
    fmean = float(np.nanmean(f)) if len(f) else 0
    pmean = float(np.nanmean(p)) if len(p) else 0
    if transform == "div10": fmean /= 10; pmean /= 10
    elif transform == "x10": fmean *= 10; pmean *= 10
    fast_vals.append(fmean); prol_vals.append(pmean)

x = np.arange(len(metrics))
w = 0.36
bars_f = ax_f.bar(x - w/2, fast_vals, width=w, color=COLORS["Fast"],
                   edgecolor='black', linewidth=1.1,
                   label=f"Fast — Flash + Short (n={len(fast_ev)})",
                   zorder=2)
bars_p = ax_f.bar(x + w/2, prol_vals, width=w, color=COLORS["Prolonged"],
                   edgecolor='black', linewidth=1.1,
                   label=f"Prolonged (n={len(prol_ev)})",
                   zorder=2)

for bars in [bars_f, bars_p]:
    for bar in bars:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                  f"{h:.1f}", ha='center', va='bottom',
                  fontsize=12, fontweight='bold')

ax_f.set_xticks(x)
ax_f.set_xticklabels([m[0] for m in metrics],
                      fontsize=13.5, fontweight='bold')
ax_f.set_ylabel("Mean value (unit-scaled for comparison)")
ax_f.set_title("(f) Fast vs Prolonged: mean event characteristics",
               loc='left', pad=12)
ax_f.grid(axis='y', ls=':', alpha=0.4)
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)
add_panel_letter(ax_f, 'f')
ax_f.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='0.3')

# Ratio annotations
for i, (f, p) in enumerate(zip(fast_vals, prol_vals)):
    if f > 0:
        ratio = p / f
        ax_f.text(x[i], max(f, p) * 1.22,
                  f"×{ratio:.1f}", ha='center', va='bottom',
                  fontsize=12, fontweight='bold', color='#333',
                  bbox=dict(boxstyle='round,pad=0.22',
                            facecolor='#fff9c4', edgecolor='#999',
                            linewidth=0.7, alpha=0.95))

# ==========================================================
# Suptitle + explanatory note (no overlap)
# ==========================================================
fig.suptitle(
    "Figure 9: Category-dependent flood-event dynamics — "
    "Fast (Flash+Short) vs Prolonged, Shweli–Ayeyarwady basin",
    fontsize=18, fontweight='bold', y=0.975)

fig.text(
    0.5, 0.008,
    "Grey bands: darker = typical Flash/Short window (0–2 d);  "
    "lighter = typical Short/Prolonged window (2–5 d)",
    ha='center', va='bottom', fontsize=13, style='italic', color='#555',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
              edgecolor='#999', linewidth=0.9, alpha=0.95))

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig09_Category_Dynamics_v2.png"
out_pdf = FIG_DIR / "Fig09_Category_Dynamics_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# -----------------------------------------------------------------------------
# Numeric summary
# -----------------------------------------------------------------------------
print("\n=== Event characteristics by category ===")
for cat in CAT_ORDER:
    sub = ev[ev["category"] == cat]
    if len(sub) == 0:
        continue
    print(f"  {cat:<10s}  n={len(sub):2d}  "
          f"dur(med)={sub['duration_days'].median():.1f} d  "
          f"rain_sum(med)={sub['rain_sum_mm'].median():.0f} mm  "
          f"peak_SWRI(med)={sub['peak_SWRI'].median():.2f}  "
          f"SM_onset(med)={sub['SM_onset'].median():.3f}")

print("\n=== Fast vs Prolonged ===")
print(f"  Fast       n = {len(fast_ev)}")
print(f"  Prolonged  n = {len(prol_ev)}")
for label, col, _ in metrics:
    f = fast_ev[col].dropna().mean() if col in fast_ev else np.nan
    p = prol_ev[col].dropna().mean() if col in prol_ev else np.nan
    if np.isfinite(f) and f > 0:
        print(f"  {label.replace(chr(10), ' '):<22s}  "
              f"Fast = {f:7.2f}  Prolonged = {p:7.2f}  "
              f"ratio = {p/f:.2f}")

plt.show()
```

    Loading prep outputs...
      Category composites: 77 rows
      Fast vs Prolonged  : 77 rows
      Events             : 11
      Categories present : ['Short', 'Prolonged']
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig09_Category_Dynamics_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig09_Category_Dynamics_v2.pdf
    
    === Event characteristics by category ===
      Short       n= 5  dur(med)=4.0 d  rain_sum(med)=72 mm  peak_SWRI(med)=31.75  SM_onset(med)=0.358
      Prolonged   n= 6  dur(med)=16.5 d  rain_sum(med)=237 mm  peak_SWRI(med)=23.24  SM_onset(med)=0.348
    
    === Fast vs Prolonged ===
      Fast       n = 5
      Prolonged  n = 6
      Duration (days)         Fast =    3.80  Prolonged =   19.33  ratio = 5.09
      Rain sum (mm)           Fast =   70.88  Prolonged =  257.05  ratio = 3.63
      Peak SWRI (/10)         Fast =   29.13  Prolonged =   29.12  ratio = 1.00
      SM onset (×10)          Fast =    0.34  Prolonged =    0.35  ratio = 1.02
    


    
![png](output_29_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 10 — Synoptic composites and category contrasts
=========================================================================
High-end Nature-journal style. 2x3 panels.
Optimized layout for large fonts and zero overlaps.
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
NCEP    = ROOT / "trend" / "Raw" / "NCEP" / "Processed"
SHP_DIR = ROOT / "output_shapefiles"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXT = [85.0, 108.0, 8.0, 30.0]           # synoptic view
SHWELI_BOX = [95.5, 22.0, 101.5, 26.5]   # focus region
QUIVER_STEP = 1

FLOOD_DATES = {
    1991:"1991-07-10", 1998:"1998-07-28", 2004:"2004-07-18", 2010:"2010-09-01",
    2015:"2015-07-16", 2018:"2018-07-09", 2019:"2019-07-10", 2020:"2020-07-02",
    2023:"2023-05-14", 2023.1:"2023-08-07", 2024:"2024-06-30", 2024.1:"2024-09-09",
}

# -----------------------------------------------------------------------------
# Load event catalog
# -----------------------------------------------------------------------------
print("Loading event catalog...")
ev = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                 parse_dates=["start_date","end_date"])
print(f"  Total events: {len(ev)}")
print(f"  Categories  : {dict(ev['category'].value_counts())}")

ev["group"] = ev["category"].map({"Flash":"Fast","Short":"Fast",
                                    "Prolonged":"Prolonged"})

def event_days(df):
    days = []
    for _, r in df.iterrows():
        days.extend(pd.date_range(r["start_date"], r["end_date"], freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))

fast_days = event_days(ev[ev["group"] == "Fast"])
prol_days = event_days(ev[ev["group"] == "Prolonged"])
all_days  = event_days(ev)

# -----------------------------------------------------------------------------
# Robust loader
# -----------------------------------------------------------------------------
def load_nc(path, hint=None, squeeze_time=False):
    if not path.exists():
        return None, None
    ds = xr.open_dataset(path)
    for c in ["latitude","longitude"]:
        if c in ds.dims:
            ds = ds.rename({c: c[:3]})
    ds = ds.sortby("lat").sortby("lon")

    var = None
    if hint and hint in ds.data_vars:
        var = hint
    else:
        for v in ds.data_vars:
            if hint and hint.lower() in v.lower():
                var = v; break
        if var is None:
            var = list(ds.data_vars)[0]

    da = ds[var]
    for d in list(da.dims):
        if d not in ("lat","lon","time") and da.sizes[d] == 1:
            da = da.isel({d:0})

    if squeeze_time and "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0)

    da = da.sel(lat=slice(EXT[2], EXT[3]), lon=slice(EXT[0], EXT[1]))
    return da, ds

# Load pre-computed composites
tcwv_c,  _  = load_nc(NCEP/"MM_pr_wtr_composite_event.nc",  hint="pr_wtr",  squeeze_time=True)
tcwv_a,  _  = load_nc(NCEP/"MM_pr_wtr_anomaly_event.nc",    hint="anom",    squeeze_time=True)
hgt_c,   _  = load_nc(NCEP/"MM_hgt_500_composite_event.nc", hint="hgt",     squeeze_time=True)
hgt_a,   _  = load_nc(NCEP/"MM_hgt_500_anomaly_event.nc",   hint="anom",    squeeze_time=True)
uwnd_c,  _  = load_nc(NCEP/"MM_uwnd_850_composite_event.nc",hint="uwnd",    squeeze_time=True)
vwnd_c,  _  = load_nc(NCEP/"MM_vwnd_850_composite_event.nc",hint="vwnd",    squeeze_time=True)

# Category composites
def category_composite(nc_path, hint, event_days):
    if not nc_path.exists():
        return None
    ds = xr.open_dataset(nc_path)
    for c in ["latitude","longitude"]:
        if c in ds.dims: ds = ds.rename({c: c[:3]})
    ds = ds.sortby("lat").sortby("lon")

    var = hint if hint in ds.data_vars else [v for v in ds.data_vars if hint.lower() in v.lower()][0]
    da = ds[var]
    for d in list(da.dims):
        if d not in ("lat","lon","time") and da.sizes[d] == 1:
            da = da.isel({d:0})

    da = da.sel(lat=slice(EXT[2], EXT[3]), lon=slice(EXT[0], EXT[1]))
    valid_times = pd.to_datetime(da.time.values)
    sel_days = [d for d in event_days if d in valid_times]
    if len(sel_days) < 5:
        ds.close(); return None

    sub = da.sel(time=pd.DatetimeIndex(sel_days), method="nearest")
    comp = sub.mean(dim="time", skipna=True)
    ds.close()
    return comp

tcwv_fast = category_composite(NCEP/"MM_pr_wtr_daily.nc", "pr_wtr", fast_days)
tcwv_prol = category_composite(NCEP/"MM_pr_wtr_daily.nc", "pr_wtr", prol_days)

# JJAS climatology
clim_tcwv = None
if (NCEP/"MM_pr_wtr_daily.nc").exists():
    ds_d = xr.open_dataset(NCEP/"MM_pr_wtr_daily.nc")
    for c in ["latitude","longitude"]:
        if c in ds_d.dims: ds_d = ds_d.rename({c: c[:3]})
    ds_d = ds_d.sortby("lat").sortby("lon")
    var = list(ds_d.data_vars)[0]
    da = ds_d[var]
    for d in list(da.dims):
        if d not in ("lat","lon","time") and da.sizes[d] == 1:
            da = da.isel({d:0})
    da = da.sel(lat=slice(EXT[2], EXT[3]), lon=slice(EXT[0], EXT[1]))
    jjas = da.where(da.time.dt.month.isin([6,7,8,9]), drop=True)
    clim_tcwv = jjas.mean(dim="time", skipna=True)
    ds_d.close()

tcwv_fast_anom = (tcwv_fast - clim_tcwv) if (tcwv_fast is not None and clim_tcwv is not None) else None
tcwv_prol_anom = (tcwv_prol - clim_tcwv) if (tcwv_prol is not None and clim_tcwv is not None) else None

# Daily PW
pw_series = None
if (NCEP/"MM_pr_wtr_daily.nc").exists():
    ds_pw = xr.open_dataset(NCEP/"MM_pr_wtr_daily.nc")
    for c in ["latitude","longitude"]:
        if c in ds_pw.dims: ds_pw = ds_pw.rename({c: c[:3]})
    ds_pw = ds_pw.sortby("lat").sortby("lon")
    var = list(ds_pw.data_vars)[0]
    pw_box = (ds_pw[var]
              .sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                   lon=slice(SHWELI_BOX[0], SHWELI_BOX[2]))
              .mean(dim=["lat","lon"], skipna=True))
    pw_series = pw_box.to_series()
    pw_series.index = pd.to_datetime(pw_series.index).normalize()
    ds_pw.close()

# -----------------------------------------------------------------------------
# Colour maps
# -----------------------------------------------------------------------------
tcwv_cmap = LinearSegmentedColormap.from_list(
    "tcwv",
    [(0.00,"#0d2b52"),(0.10,"#1c4c8a"),(0.25,"#4a7ab5"),(0.40,"#8bb8d8"),
     (0.50,"#e8f0f5"),(0.60,"#f5eeda"),(0.72,"#f8e6a3"),(0.84,"#f5c96b"),
     (0.94,"#f0a850"),(1.00,"#e88a3a")])

anom_cmap = LinearSegmentedColormap.from_list(
    "anom",
    [(0.00,"#08306b"),(0.15,"#2171b5"),(0.35,"#9ecae1"),(0.50,"#f7f7f7"),
     (0.65,"#fdae6b"),(0.85,"#e6550d"),(1.00,"#a50f15")])

flux_cmap = plt.get_cmap("YlGnBu")

# -----------------------------------------------------------------------------
# Nature-style — LARGER FONTS
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 17,
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 18, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.1,
    'xtick.labelsize': 18, 'ytick.labelsize': 18,
    'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 16, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

def add_panel_letter(ax, letter):
    ax.text(0.018, 0.978, letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.32', facecolor='white',
                      edgecolor='black', linewidth=1.2, alpha=0.97),
            zorder=100)

def draw_base(ax):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#e8f0f6', zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor='#f5f2ea', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#333', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#555', zorder=3)

def draw_shweli_box(ax):
    lon0, lat0, lon1, lat1 = SHWELI_BOX
    ax.add_patch(Rectangle((lon0,lat0), lon1-lon0, lat1-lat0,
                            facecolor='none', edgecolor='red',
                            linewidth=2.6, zorder=8))
    ax.text((lon0+lon1)/2, lat1+0.55, 'Shweli basin',
            fontsize=12, fontweight='bold', color='red',
            ha='center', va='bottom', transform=ccrs.PlateCarree(),
            zorder=9,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor='red', linewidth=1.0, alpha=0.95))

def style_grid(ax):
    gl = ax.gridlines(draw_labels=True, lw=0.4, color='gray',
                       alpha=0.45, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 13, 'weight': 'bold'}
    gl.ylabel_style = {'size': 13, 'weight': 'bold'}

# -----------------------------------------------------------------------------
# Figure Layout — MORE ROOM FOR BOTTOM ROW & LEGEND
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(26, 18),dpi=500)
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.0, 1.0],
    hspace=0.32, wspace=0.18,
    left=0.05, right=0.97, top=0.92, bottom=0.11,
)

# ==========================================================
# (a) All-event composite
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
draw_base(ax_a)

if tcwv_c is not None:
    lon2d, lat2d = np.meshgrid(tcwv_c.lon.values, tcwv_c.lat.values)
    lv = np.linspace(float(tcwv_c.min()), float(tcwv_c.max()), 20)
    cf_a = ax_a.contourf(lon2d, lat2d, tcwv_c.values, levels=lv,
                          cmap=tcwv_cmap, extend='both',
                          transform=ccrs.PlateCarree(), zorder=2)

if hgt_c is not None:
    cs_a = ax_a.contour(lon2d, lat2d, hgt_c.values,
                         levels=np.arange(np.floor(hgt_c.min()/3)*3,
                                          np.ceil(hgt_c.max()/3)*3 + 3, 3),
                         colors='black', linewidths=1.2,
                         transform=ccrs.PlateCarree(), zorder=5)
    ax_a.clabel(cs_a, inline=True, fontsize=11, fmt='%d', inline_spacing=3)

if uwnd_c is not None and vwnd_c is not None:
    u = uwnd_c.values; v = vwnd_c.values
    s = QUIVER_STEP
    ax_a.quiver(lon2d[::s,::s], lat2d[::s,::s], u[::s,::s], v[::s,::s],
                 color='white', scale=130, width=0.0038,
                 edgecolor='black', linewidth=0.3,
                 transform=ccrs.PlateCarree(), zorder=6)

draw_shweli_box(ax_a)
style_grid(ax_a)

ax_a.set_title("(a) Composite — TCWV · GH-500 · wind-850", loc='left', pad=12)
add_panel_letter(ax_a, 'a')

if tcwv_c is not None:
    cax_a = fig.add_axes([0.06, 0.535, 0.24, 0.015])
    cb_a = fig.colorbar(cf_a, cax=cax_a, orientation='horizontal')
    cb_a.set_label("TCWV (mm)", fontsize=14, fontweight='bold', labelpad=6)
    cb_a.ax.tick_params(labelsize=13)

# ==========================================================
# (b) All-event TCWV anomaly + GH anomaly contours
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
draw_base(ax_b)

if tcwv_a is not None:
    lon2d, lat2d = np.meshgrid(tcwv_a.lon.values, tcwv_a.lat.values)
    vmax = max(abs(float(tcwv_a.min())), abs(float(tcwv_a.max())))
    lv = np.linspace(-vmax, vmax, 21)
    cf_b = ax_b.contourf(lon2d, lat2d, tcwv_a.values, levels=lv,
                          cmap=anom_cmap, extend='both',
                          transform=ccrs.PlateCarree(), zorder=2)

if hgt_a is not None:
    cs_b = ax_b.contour(lon2d, lat2d, hgt_a.values,
                         levels=np.arange(np.floor(hgt_a.min()/2)*2,
                                          np.ceil(hgt_a.max()/2)*2 + 2, 2),
                         colors='black', linewidths=1.1,
                         transform=ccrs.PlateCarree(), zorder=5)
    ax_b.clabel(cs_b, inline=True, fontsize=10.5, fmt='%d', inline_spacing=3)
    ax_b.contour(lon2d, lat2d, hgt_a.values, levels=[0],
                  colors='magenta', linewidths=2.4,
                  transform=ccrs.PlateCarree(), zorder=6)

draw_shweli_box(ax_b)
style_grid(ax_b)

ax_b.set_title("(b) Anomaly — TCWV' · GH-500'", loc='left', pad=12)
add_panel_letter(ax_b, 'b')

if tcwv_a is not None:
    cax_b = fig.add_axes([0.38, 0.535, 0.24, 0.015])
    cb_b = fig.colorbar(cf_b, cax=cax_b, orientation='horizontal')
    cb_b.set_label("TCWV anomaly (mm)", fontsize=14, fontweight='bold', labelpad=6)
    cb_b.ax.tick_params(labelsize=13)

# ==========================================================
# (c) Moisture flux magnitude
# ==========================================================
ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
draw_base(ax_c)

if (tcwv_c is not None and uwnd_c is not None and vwnd_c is not None):
    g = 9.80665
    Qu = (tcwv_c.values * uwnd_c.values) / g
    Qv = (tcwv_c.values * vwnd_c.values) / g
    Qmag = np.sqrt(Qu**2 + Qv**2)
    lon2d, lat2d = np.meshgrid(tcwv_c.lon.values, tcwv_c.lat.values)
    lv = np.linspace(float(np.nanmin(Qmag)), float(np.nanmax(Qmag)), 20)
    cf_c = ax_c.contourf(lon2d, lat2d, Qmag, levels=lv,
                          cmap=flux_cmap, extend='both',
                          transform=ccrs.PlateCarree(), zorder=2)

    s = QUIVER_STEP
    ax_c.quiver(lon2d[::s,::s], lat2d[::s,::s],
                 uwnd_c.values[::s,::s], vwnd_c.values[::s,::s],
                 color='#111', scale=160, width=0.0028,
                 transform=ccrs.PlateCarree(), zorder=6)

    cax_c = fig.add_axes([0.70, 0.535, 0.24, 0.015])
    cb_c = fig.colorbar(cf_c, cax=cax_c, orientation='horizontal')
    cb_c.set_label(r"|Q| (kg m$^{-1}$ s$^{-1}$)", fontsize=14, fontweight='bold', labelpad=6)
    cb_c.ax.tick_params(labelsize=13)

draw_shweli_box(ax_c)
style_grid(ax_c)

ax_c.set_title(r"(c) Moisture flux magnitude |Q| = |qV|/g", loc='left', pad=12)
add_panel_letter(ax_c, 'c')

# ==========================================================
# (d) PW time series over Shweli box
# ==========================================================
ax_d = fig.add_subplot(gs[1, 0])

if pw_series is not None:
    ax_d.plot(pw_series.index, pw_series.values,
              color="#888", lw=0.5, alpha=0.8, label="Daily PW")

    jjas_mean = float(pw_series[pw_series.index.month.isin([6,7,8,9])].mean())
    ax_d.axhline(jjas_mean, color="blue", ls="--", lw=2.0,
                  label=f"JJAS mean = {jjas_mean:.1f} mm")

    all_days_in_range = [d for d in all_days if d in pw_series.index]
    if all_days_in_range:
        pw_ev = pw_series.loc[all_days_in_range]
        ax_d.scatter(pw_ev.index, pw_ev.values, color="#d32f2f",
                      s=55, edgecolor="black", linewidth=0.8,
                      zorder=5, label=f"Event days (n={len(all_days_in_range)})")

    for d in fast_days:
        if d in pw_series.index:
            ax_d.axvspan(d, d + pd.Timedelta(days=1),
                          color="#E66101", alpha=0.15, zorder=0)
    for d in prol_days:
        if d in pw_series.index:
            ax_d.axvspan(d, d + pd.Timedelta(days=1),
                          color="#B2182B", alpha=0.10, zorder=0)

ax_d.set_xlabel("Year")
ax_d.set_ylabel("Basin PW (mm)")
ax_d.set_title("(d) Precipitable water over the Shweli basin", loc='left', pad=12)
ax_d.grid(ls=':', alpha=0.4)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
add_panel_letter(ax_d, 'd')
ax_d.legend(loc='upper left', fontsize=13, ncol=2, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# (e) Fast-event anomaly composite
# ==========================================================
ax_e = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
draw_base(ax_e)

if tcwv_fast_anom is not None:
    lon2d, lat2d = np.meshgrid(tcwv_fast_anom.lon.values,
                                tcwv_fast_anom.lat.values)
    vmax = max(abs(float(tcwv_fast_anom.min())),
               abs(float(tcwv_fast_anom.max())))
    lv = np.linspace(-vmax, vmax, 21)
    cf_e = ax_e.contourf(lon2d, lat2d, tcwv_fast_anom.values, levels=lv,
                          cmap=anom_cmap, extend='both',
                          transform=ccrs.PlateCarree(), zorder=2)

    cax_e = fig.add_axes([0.38, 0.065, 0.24, 0.015])
    cb_e = fig.colorbar(cf_e, cax=cax_e, orientation='horizontal')
    cb_e.set_label("TCWV anomaly (mm)", fontsize=14, fontweight='bold', labelpad=6)
    cb_e.ax.tick_params(labelsize=13)

    box_anom = (tcwv_fast_anom
                .sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                     lon=slice(SHWELI_BOX[0], SHWELI_BOX[2]))
                .mean().values)
    ax_e.text(0.97, 0.04,
              f"Basin-mean anomaly: {float(box_anom):+.2f} mm",
              transform=ax_e.transAxes, ha='right', va='bottom',
              fontsize=13, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='#333', linewidth=1.0, alpha=0.95),
              zorder=100)

draw_shweli_box(ax_e)
style_grid(ax_e)

n_fast = len(ev[ev["group"] == "Fast"])
ax_e.set_title(f"(e) Fast-event anomaly (n={n_fast}, Flash+Short)", loc='left', pad=12)
add_panel_letter(ax_e, 'e')

# ==========================================================
# (f) Prolonged-event anomaly composite
# ==========================================================
ax_f = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
draw_base(ax_f)

if tcwv_prol_anom is not None:
    lon2d, lat2d = np.meshgrid(tcwv_prol_anom.lon.values,
                                tcwv_prol_anom.lat.values)
    vmax = max(abs(float(tcwv_prol_anom.min())),
               abs(float(tcwv_prol_anom.max())))
    lv = np.linspace(-vmax, vmax, 21)
    cf_f = ax_f.contourf(lon2d, lat2d, tcwv_prol_anom.values, levels=lv,
                          cmap=anom_cmap, extend='both',
                          transform=ccrs.PlateCarree(), zorder=2)

    cax_f = fig.add_axes([0.70, 0.065, 0.24, 0.015])
    cb_f = fig.colorbar(cf_f, cax=cax_f, orientation='horizontal')
    cb_f.set_label("TCWV anomaly (mm)", fontsize=14, fontweight='bold', labelpad=6)
    cb_f.ax.tick_params(labelsize=13)

    box_anom = (tcwv_prol_anom
                .sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                     lon=slice(SHWELI_BOX[0], SHWELI_BOX[2]))
                .mean().values)
    ax_f.text(0.97, 0.04,
              f"Basin-mean anomaly: {float(box_anom):+.2f} mm",
              transform=ax_f.transAxes, ha='right', va='bottom',
              fontsize=13, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='#333', linewidth=1.0, alpha=0.95),
              zorder=100)

draw_shweli_box(ax_f)
style_grid(ax_f)

n_prol = len(ev[ev["group"] == "Prolonged"])
ax_f.set_title(f"(f) Prolonged-event anomaly (n={n_prol})", loc='left', pad=12)
add_panel_letter(ax_f, 'f')

# ==========================================================
# Shared legend (bottom)
# ==========================================================
legend_items = [
    Line2D([0], [0], color='red', lw=2.4, label='Shweli focus box'),
    Line2D([0], [0], color='magenta', lw=2.2, label='Zero GH-500 anomaly contour'),
    Line2D([0], [0], marker=r'$\rightarrow$', color='black',
           markersize=16, linestyle='None',
           label='850-hPa wind'),
    Line2D([0], [0], color='black', lw=1.2, label='GH-500 contours (3 gpm)'),
    Patch(facecolor="#E66101", alpha=0.3, label="Fast-event days"),
    Patch(facecolor="#B2182B", alpha=0.3, label="Prolonged-event days"),
]

fig.legend(handles=legend_items, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.005), fontsize=14,
           frameon=True, framealpha=0.95, edgecolor='0.3')

# ==========================================================
# Suptitle
# ==========================================================
fig.suptitle(
    "Figure 10: Synoptic composites and category contrasts — "
    "Shweli extreme rainfall events (NCEP/NCAR Reanalysis-2)",
    fontsize=19, fontweight='bold', y=0.975)

# ==========================================================
# Save
# ==========================================================
out_png = FIG_DIR / "Fig10_Synoptic_Composites_v2.png"
out_pdf = FIG_DIR / "Fig10_Synoptic_Composites_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

plt.show()
```

    Loading event catalog...
      Total events: 11
      Categories  : {'Prolonged': np.int64(6), 'Short': np.int64(5)}
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig10_Synoptic_Composites_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig10_Synoptic_Composites_v2.pdf
    


    
![png](output_31_1.png)
    



```python

```


```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 11 — Climate-driver linkage with annual flood-event frequency
=========================================================================
Fixed data issue: regress against continuous annual max Rx5day (mm) 
instead of binary event counts.
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, linregress, t
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------
print("Loading prep outputs...")
df = pd.read_csv(PREP/"fig11"/"fig11_index_annual.csv")
corr_df = pd.read_csv(PREP/"tables"/"table08_climate_correlations.csv")

# Merge in annual max Rx5day and total rainfall
annual = pd.read_csv(PREP/"fig04"/"fig04_annual_indices.csv")
df = df.merge(annual[["year","Rx5_stn","rain_total"]], on="year", how="left")

# CLEAN DATA: Fill missing Rx5day with its mean to prevent index drop
df["Rx5_stn"] = df["Rx5_stn"].fillna(df["Rx5_stn"].mean())
# Create continuous target for correlation
df["target"] = df["Rx5_stn"]

print(f"  Index table   : {df.shape} rows")
print(f"  Flood years   : {int(df['is_flood_year'].sum())} / {len(df)}")

INDEX_SPECS = [
    ("ONI",  "(a) ONI (Niño 3.4)", "Apr–May (2-mo lead)",  "#2166ac"),
    ("DMI",  "(b) DMI (IOD)",       "Apr–May (2-mo lead)",  "#b2182b"),
    ("WYI",  "(c) WYI (Webster–Yang)", "JJAS (simultaneous)", "#1b7837"),
    ("SAMI", "(d) SAMI (South-Asian Monsoon)", "JJAS (simultaneous)", "#8073ac"),
]

# -----------------------------------------------------------------------------
# RF importance
# -----------------------------------------------------------------------------
try:
    rf_imp = pd.read_csv(PREP/"fig11"/"fig11_rf_importance.csv")
    if rf_imp["importance"].sum() < 0.01:
        raise ValueError("zero importance")
except Exception:
    feats = ["ONI","DMI","WYI","SAMI"]
    sub = df.dropna(subset=feats + ["is_flood_year"])
    X = sub[feats].values
    y = sub["is_flood_year"].astype(int).values
    if len(np.unique(y)) > 1 and len(sub) > 10:
        rf = RandomForestClassifier(n_estimators=500,
                                    class_weight="balanced",
                                    random_state=42, n_jobs=-1).fit(X, y)
        rf_imp = pd.DataFrame({
            "feature":    feats,
            "importance": rf.feature_importances_,
        })
    else:
        rf_imp = pd.DataFrame({"feature": feats,
                               "importance": np.full(4, np.nan)})

rf_imp_sorted = rf_imp.sort_values("importance", ascending=True).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Regression + CI helper
# -----------------------------------------------------------------------------
def regression_ci(x, y):
    """Ordinary least squares with 95% CI band."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return None
    slope, intercept, r, p_p, _ = linregress(x, y)
    rho, p_s = spearmanr(x, y)

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = intercept + slope * x_fit
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))
    t_crit = t.ppf(0.975, n - 2)
    se_line = s_err * np.sqrt(1/n + (x_fit - x.mean())**2 /
                              np.sum((x - x.mean())**2))
    ci = t_crit * se_line
    return dict(slope=slope, intercept=intercept, n=n,
                r=r, p_p=p_p, rho=rho, p_s=p_s,
                x_fit=x_fit, y_fit=y_fit, ci=ci)

def p_str(p):
    if not np.isfinite(p): return "p = n/a"
    if p < 0.001:          return "p < 0.001"
    return f"p = {p:.3f}"

# -----------------------------------------------------------------------------
# Gradients
# -----------------------------------------------------------------------------
rx5_cmap = LinearSegmentedColormap.from_list(
    "rx5_grad",
    [(0.00, "#2166ac"),(0.25, "#67a9cf"),(0.50, "#ffffbf"),
     (0.75, "#f46d43"),(1.00, "#a50026")])
rx5_norm = Normalize(vmin=float(df["Rx5_stn"].min()),
                      vmax=float(df["Rx5_stn"].max()))

rf_cmap = LinearSegmentedColormap.from_list(
    "rf_grad",
    [(0.00, "#1e3a8a"),(0.20, "#0891b2"),(0.45, "#16a34a"),
     (0.70, "#ca8a04"),(1.00, "#991b1b")])

# -----------------------------------------------------------------------------
# Style — Larger fonts for readability
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 18,
    'axes.titlesize': 20, 'axes.titleweight': 'bold',
    'axes.labelsize': 19, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 16.5, 'ytick.labelsize': 16.5,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6.5, 'ytick.major.size': 6.5,
    'legend.fontsize': 15.5, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'legend.borderpad': 0.6, 'legend.labelspacing': 0.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure Layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 16),dpi=500)
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[1.0, 1.0, 0.75],
    hspace=0.45, wspace=0.25,
    left=0.075, right=0.965, top=0.92, bottom=0.08,
)

# =============================================================================
# Scatter panels
# =============================================================================
panel_slots = [(0,0), (0,1), (1,0), (1,1)]

for (idx_name, title, window_label, idx_color), (row, col) in zip(INDEX_SPECS, panel_slots):
    ax = fig.add_subplot(gs[row, col])

    x = df[idx_name].values
    y = df["target"].values  # Use Rx5_stn instead of event_days
    rx5 = df["Rx5_stn"].values
    is_flood = df["is_flood_year"].astype(bool).values

    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    rx5v   = rx5[mask]
    floodv = is_flood[mask]

    # ---- Hexagon markers ----
    for xi, yi, ri, fi in zip(xv, yv, rx5v, floodv):
        c = rx5_cmap(rx5_norm(ri))
        edge = "#d32f2f" if fi else "#222"
        lw   = 2.2 if fi else 0.9
        ax.scatter(xi, yi, s=210, c=[c], marker='h',
                    edgecolor=edge, linewidth=lw,
                    alpha=0.90, zorder=4)

    # ---- Regression + 95% CI ----
    stats = regression_ci(xv, yv)
    if stats is not None:
        ax.plot(stats["x_fit"], stats["y_fit"],
                color=idx_color, lw=3.2, zorder=5, label="Linear fit")
        ax.fill_between(stats["x_fit"],
                        stats["y_fit"] - stats["ci"],
                        stats["y_fit"] + stats["ci"],
                        color=idx_color, alpha=0.16, zorder=3,
                        label="95% CI")

    # ---- Reference lines ----
    ax.axhline(0, color="#aaa", ls=":", lw=1.2, zorder=1)
    if stats is not None:
        ax.axvline(np.nanmean(xv), color="#aaa", ls=":", lw=1.2, zorder=1)

    # ---- Stats box ----
    if stats is not None:
        txt = (f"Pearson r  = {stats['r']:+.2f}   ({p_str(stats['p_p'])})\n"
               f"Spearman ρ = {stats['rho']:+.2f}   ({p_str(stats['p_s'])})\n"
               f"n = {stats['n']}")
        ax.text(0.028, 0.572, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=12.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor="white", edgecolor="#555",
                          linewidth=1.0, alpha=0.94),
                zorder=100)

    # ---- Labels and title ----
    ax.set_xlabel(f"{idx_name} index")
    ax.set_ylabel("Annual max Rx5day (mm)") # Changed y-axis label
    ax.set_title(f"{title}\n{window_label}", loc="left", pad=10, fontsize=20)
    ax.grid(ls=":", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # NO PANEL LETTER ADDED

# =============================================================================
# (e) RF importance — full-width
# =============================================================================
ax_e = fig.add_subplot(gs[2, :])

vals = rf_imp_sorted["importance"].values
names = rf_imp_sorted["feature"].values

if np.nanmax(vals) > np.nanmin(vals):
    norm_imp = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
else:
    norm_imp = np.full_like(vals, 0.5)

y_pos = np.arange(len(names))

for i, (name, val, nrm) in enumerate(zip(names, vals, norm_imp)):
    base_color = rf_cmap(nrm)
    ax_e.barh(i, val, height=0.62, color=base_color, edgecolor="black", linewidth=1.2, zorder=3)
    ax_e.barh(i, val * 0.85, height=0.30, color="white", alpha=0.35, edgecolor="none", zorder=4, left=val * 0.08)
    ax_e.barh(i, val * 0.10, height=0.62, color="black", alpha=0.18, edgecolor="none", zorder=4)

for i, val in enumerate(vals):
    if np.isfinite(val):
        ax_e.text(val + 0.008, i, f"{val:.3f}",
                  va="center", ha="left", fontsize=15, fontweight="bold", color="#111")

ax_e.set_yticks(y_pos)
ax_e.set_yticklabels(names, fontsize=16, fontweight='bold')
ax_e.set_xlabel("Random-Forest Gini importance", fontweight='bold')
ax_e.set_xlim(0, float(np.nanmax(vals)) * 1.22 if np.isfinite(vals).any() else 1.0)
ax_e.set_title("(e) Random-Forest importance of climate drivers", loc="left", pad=10, fontsize=20)
ax_e.grid(axis="x", ls=":", alpha=0.35)
ax_e.set_axisbelow(True)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

if np.isfinite(vals).any():
    top_idx = int(np.nanargmax(vals))
    top_name = names[top_idx]
    runner_idx = int(np.argsort(vals)[-2]) if len(vals) > 1 else top_idx
    runner_name = names[runner_idx]

    insight = (f"Top-ranked drivers: {top_name} ({vals[top_idx]:.3f})  and  "
               f"{runner_name} ({vals[runner_idx]:.3f}).  "
               f"Monsoon indices are weaker predictors.")
    ax_e.text(0.985, 0.56, insight, transform=ax_e.transAxes, ha="right", va="bottom",
              fontsize=1, style="italic", color="#333",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff9e0",
                        edgecolor="#c9a227", linewidth=1.2, alpha=0.94))

# =============================================================================
# Shared colorbar (moved to avoid overlap)
# =============================================================================
cax = fig.add_axes([0.32, -0.015, 0.36, 0.014])
sm_rx = ScalarMappable(norm=rx5_norm, cmap=rx5_cmap)
sm_rx.set_array([])
cb = fig.colorbar(sm_rx, cax=cax, orientation="horizontal")
cb.set_label("Colour of hexagon: Annual max Rx5day (mm)",
              fontsize=14, fontweight="bold", labelpad=6)
cb.ax.tick_params(labelsize=13)

# =============================================================================
# Shared legend at bottom
# =============================================================================
legend_items = [
    Line2D([0], [0], marker='h', color='w', markerfacecolor='#67a9cf',
           markeredgecolor='#d32f2f', markeredgewidth=2.0, markersize=16,
           linestyle='None', label='Flood year (red ring)'),
    Line2D([0], [0], marker='h', color='w', markerfacecolor='#67a9cf',
           markeredgecolor='#222', markeredgewidth=0.9, markersize=16,
           linestyle='None', label='Non-flood year'),
    Line2D([0], [0], color='black', lw=3.2, label='Linear fit'),
    Patch(facecolor='black', alpha=0.16, label='95% confidence band'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.01), fontsize=14,
           frameon=True, framealpha=0.95, edgecolor='0.3')

# =============================================================================
# Suptitle
# =============================================================================
#fig.suptitle(
 #   "Figure 11: Linkage between large-scale climate variability "
 #   "and annual maximum extreme rainfall (1981–2025)",
 #   fontsize=19, fontweight="bold", y=0.98)

# =============================================================================
# Save
# =============================================================================
out_png = FIG_DIR / "Fig11_Climate_Linkage_v2.png"
out_pdf = FIG_DIR / "Fig11_Climate_Linkage_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")
plt.show()
```

    Loading prep outputs...
      Index table   : (11, 11) rows
      Flood years   : 4 / 11
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig11_Climate_Linkage_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig11_Climate_Linkage_v2.pdf
    


    
![png](output_34_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 12 — Operational flood-trigger curve and daily RF validation
=========================================================================
Manuscript-ready, large fonts, no overlapping labels or boxes.

Panels:
  (a) Operational flood-trigger curve P(Flood | Rx5day) with risk tiers
  (b) Binned observed probabilities vs logistic fit
  (c) Daily RF — ROC curve with operating point
  (d) Confusion matrix + performance metrics card

Reads:
    prep_revised_v2/fig12/fig12_binned_prob.csv
    prep_revised_v2/fig12/fig12_logistic_curve.csv
    prep_revised_v2/fig12/operational_tiers.json
    prep_revised_v2/fig12/fig12_rf_roc.csv
    prep_revised_v2/tables/table09_rf_daily_performance.csv

Writes:
    CMIP6_Evaluation/output_shapefiles/Fig12_Operational_Validation_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Paths and load
# -----------------------------------------------------------------------------
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading prep outputs...")
binned   = pd.read_csv(PREP/"fig12"/"fig12_binned_prob.csv")
logistic = pd.read_csv(PREP/"fig12"/"fig12_logistic_curve.csv")
tiers    = json.load(open(PREP/"fig12"/"operational_tiers.json"))
roc      = pd.read_csv(PREP/"fig12"/"fig12_rf_roc.csv")
perf     = pd.read_csv(PREP/"tables"/"table09_rf_daily_performance.csv").iloc[0]

P30   = float(tiers["P0_30_mm"])
P60   = float(tiers["P0_60_mm"])
T_RX5 = float(tiers["T_RX5"])
T_SM  = float(tiers["T_SM"])
T_SWRI= float(tiers["T_SWRI"])

auc_val = float(perf["AUC"])
pod_val = float(perf["POD"])
far_val = float(perf["FAR"])
brier   = float(perf["brier"])
TP = int(perf["TP"]); FP = int(perf["FP"])
FN = int(perf["FN"]); TN = int(perf["TN"])

print(f"  P30 threshold        : {P30:.1f} mm")
print(f"  P60 threshold        : {P60:.1f} mm")
print(f"  ML Rx5 threshold     : {T_RX5:.1f} mm")
print(f"  RF AUC / POD / FAR   : {auc_val:.3f} / {pod_val:.3f} / {far_val:.3f}")
print(f"  Confusion            : TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"  Brier score          : {brier:.4f}")

# Binned: filter noisy low-count bins
bplot = binned[binned["n"] >= 5].copy()
if len(bplot) > 1:
    bar_w = min(12, (bplot["mid"].iloc[1] - bplot["mid"].iloc[0]) * 0.82)
else:
    bar_w = 12

# -----------------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------------
TIER_LOW   = "#a5d6a7"
TIER_MOD   = "#ffe082"
TIER_HIGH  = "#ef9a9a"
CURVE_COL  = "#c62828"
OBS_COL    = "#1f77b4"

# -----------------------------------------------------------------------------
# Nature-style — LARGE, CLEAR
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 18,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 17.5, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6.5, 'ytick.major.size': 6.5,
    'legend.fontsize': 13.5, 'legend.frameon': True,
    'legend.framealpha': 0.96, 'legend.edgecolor': '0.3',
    'legend.borderpad': 0.6, 'legend.labelspacing': 0.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure — 2 × 2
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 12),dpi=500)
gs = fig.add_gridspec(
    2, 2,
    hspace=0.34, wspace=0.24,
    left=0.075, right=0.965, top=0.945, bottom=0.065,
)

# =============================================================================
# (a) Operational trigger curve
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])

# Tier bands
ax_a.axhspan(0,  30, color=TIER_LOW,  alpha=0.38, zorder=0)
ax_a.axhspan(30, 60, color=TIER_MOD,  alpha=0.38, zorder=0)
ax_a.axhspan(60, 100,color=TIER_HIGH, alpha=0.38, zorder=0)

# Logistic curve
ax_a.plot(logistic["x_mm"], logistic["p"] * 100,
          color=CURVE_COL, lw=4.0, zorder=6, label="Logistic fit")

# Vertical thresholds
ax_a.axvline(P30,   color="black",   ls="--", lw=2.6, zorder=4,
              label=f"Operational (P=30%) = {P30:.0f} mm")
ax_a.axvline(P60,   color="#7f0000", ls="--", lw=2.6, zorder=4,
              label=f"Confident (P=60%) = {P60:.0f} mm")
ax_a.axvline(T_RX5, color="#1b5e20", ls=":",  lw=2.6, zorder=4,
              label=f"ML Rx5 threshold = {T_RX5:.0f} mm")

# Tier labels — right side, well away from curve
ax_a.text(244, 15, "LOW",      fontsize=17, fontweight="bold",
          color="#1b5e20", ha="right", va="center", zorder=10)
ax_a.text(244, 45, "MODERATE", fontsize=17, fontweight="bold",
          color="#ef6c00", ha="right", va="center", zorder=10)
ax_a.text(244, 80, "HIGH",     fontsize=17, fontweight="bold",
          color="#c62828", ha="right", va="center", zorder=10)

# Cross-point markers (where logistic crosses 30% and 60%)
ax_a.scatter([P30, P60], [30, 60], s=200, color="black",
             edgecolor="white", linewidth=2.0, zorder=8)

ax_a.set_xlim(0, 250)
ax_a.set_ylim(0, 100)
ax_a.set_xlabel("Rx5day (mm)")
ax_a.set_ylabel("P(Flood | Rx5day)  [%]")
ax_a.set_title("(a) Operational flood-trigger curve", loc="left", pad=12,fontsize=20)
ax_a.grid(ls=":", alpha=0.35, zorder=1)
ax_a.set_axisbelow(True)
ax_a.legend(loc="upper left", fontsize=12.5, framealpha=0.96,
             edgecolor="0.3", borderpad=0.6)

# =============================================================================
# (b) Binned observed vs logistic
# =============================================================================
ax_b = fig.add_subplot(gs[0, 1])

# Observed bars
ax_b.bar(bplot["mid"], bplot["p"] * 100,
          width=bar_w, color=OBS_COL, alpha=0.72,
          edgecolor="black", linewidth=1.2, zorder=3,
          label="Observed P(Flood | bin)")

# Logistic fit
ax_b.plot(logistic["x_mm"], logistic["p"] * 100,
          color=CURVE_COL, lw=4.0, zorder=5,
          label="Logistic fit")

# Tier reference
ax_b.axhline(30, color="gray", ls=":", lw=1.6, zorder=2)
ax_b.axhline(60, color="gray", ls=":", lw=1.6, zorder=2)
ax_b.text(246, 30, " 30%", fontsize=13, color="#555",
          va="center", ha="left", clip_on=False)
ax_b.text(246, 60, " 60%", fontsize=13, color="#555",
          va="center", ha="left", clip_on=False)

# Operational Rx5 markers on x-axis
ax_b.axvline(P30,   color="black",   ls="--", lw=1.8, zorder=2)
ax_b.axvline(P60,   color="#7f0000", ls="--", lw=1.8, zorder=2)

ax_b.set_xlim(0, 245)
ax_b.set_ylim(0, 105)
ax_b.set_xlabel("Rx5day (mm)")
ax_b.set_ylabel("P(Flood)  [%]")
ax_b.set_title("(b) Observed vs fitted probability", loc="left", pad=12,fontsize=20)
ax_b.grid(ls=":", alpha=0.35, zorder=1)
ax_b.set_axisbelow(True)
ax_b.legend(loc="upper left", fontsize=13, framealpha=0.96,
             edgecolor="0.3", borderpad=0.6)

# =============================================================================
# (c) Daily RF ROC
# =============================================================================
ax_c = fig.add_subplot(gs[1, 0])

# ROC curve
ax_c.plot(roc["fpr"], roc["tpr"],
          color=CURVE_COL, lw=4.0, zorder=5,
          label=f"Daily RF (AUC = {auc_val:.3f})")
ax_c.fill_between(roc["fpr"], roc["tpr"], alpha=0.15,
                   color=CURVE_COL, zorder=2)

# No-skill line
ax_c.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.8,
           zorder=3, label="No-skill (AUC = 0.5)")

# Operating point (P=0.5 decision threshold)
op_far = FP / (FP + TN) if (FP + TN) > 0 else 0
op_pod = TP / (TP + FN) if (TP + FN) > 0 else 0
ax_c.scatter([op_far], [op_pod], s=300, color="black",
              edgecolor="yellow", linewidth=2.5, zorder=10,
              label=f"Operating point (POD={pod_val:.2f}, FAR={far_val:.2f})")

ax_c.set_xlim(-0.01, 1.01)
ax_c.set_ylim(-0.01, 1.01)
ax_c.set_aspect("equal")
ax_c.set_xlabel("False positive rate")
ax_c.set_ylabel("True positive rate")
ax_c.set_title("(c) Daily RF — ROC curve", loc="left", pad=12,fontsize=20)
ax_c.grid(ls=":", alpha=0.35, zorder=1)
ax_c.set_axisbelow(True)
ax_c.legend(loc="lower right", fontsize=13, framealpha=0.96,
             edgecolor="0.3", borderpad=0.6)

# =============================================================================
# (d) Confusion matrix + metrics card
# =============================================================================
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_xlim(0, 10)
ax_d.set_ylim(0, 10)
ax_d.axis("off")

# --- Column header ---
ax_d.text(3.5, 9.5, "Predicted", ha="center", fontsize=15,
           fontweight="bold")
ax_d.text(2.2, 8.8, "No flood", ha="center", fontsize=13.5)
ax_d.text(4.8, 8.8, "Flood",    ha="center", fontsize=13.5)

# --- Row header ---
ax_d.text(0.15, 6.5, "Actual", ha="center", va="center",
           rotation=90, fontsize=15, fontweight="bold")
ax_d.text(0.7, 7.5, "No",    ha="center", va="center", fontsize=13.5)
ax_d.text(0.7, 5.5, "Flood", ha="center", va="center", fontsize=13.5)

# --- 2×2 cells ---
cells = [
    (2.2, 7.5, "TN", TN, "#c8e6c9"),   # correct no-flood (pale green)
    (4.8, 7.5, "FP", FP, "#ffcc80"),   # false alarm (orange)
    (2.2, 5.5, "FN", FN, "#ef5350"),   # missed flood (red — dangerous)
    (4.8, 5.5, "TP", TP, "#66bb6a"),   # correct flood (green)
]
for x, y, lab, val, col in cells:
    ax_d.add_patch(Rectangle((x - 1.05, y - 0.72), 2.10, 1.44,
                              facecolor=col, edgecolor="black",
                              linewidth=2.2, alpha=0.95, zorder=2))
    ax_d.text(x, y + 0.32, lab, ha="center", va="center",
               fontsize=15, fontweight="bold", zorder=3)
    ax_d.text(x, y - 0.32, str(val), ha="center", va="center",
               fontsize=22, fontweight="bold", zorder=3)

# --- Metrics card ---
ax_d.text(5.0, 4.1, "Performance metrics",
           ha="center", fontsize=15.5, fontweight="bold")

metric_list = [
    ("AUC",                f"{auc_val:.3f}", "#b71c1c"),
    ("POD (Hit Rate)",     f"{pod_val:.3f}", "#b71c1c"),
    ("FAR (False alarm)",  f"{far_val:.3f}", "#b71c1c"),
    ("Brier score",        f"{brier:.4f}",   "#333"),
    ("Train / Test size",  f"{int(perf['n_train'])} / {int(perf['n_test'])}", "#333"),
    ("Positive test days", f"{int(perf['n_pos_test'])}", "#333"),
]
y_start = 3.4
row_gap = 0.42
for i, (key, val, col) in enumerate(metric_list):
    y = y_start - i * row_gap
    ax_d.text(1.0, y, key, ha="left",  fontsize=13.5, fontweight="bold")
    ax_d.text(9.0, y, val, ha="right", fontsize=14,   fontweight="bold",
               color=col)
    if i < len(metric_list) - 1:
        ax_d.plot([1.0, 9.0], [y - 0.21, y - 0.21],
                   color="#e0e0e0", lw=0.8, zorder=1)

ax_d.set_title("(d) Confusion matrix & performance metrics",
                loc="left", pad=12,fontsize=20)

# =============================================================================
# Save
# =============================================================================
out_png = FIG_DIR / "Fig12_Operational_Validation_v2.png"
out_pdf = FIG_DIR / "Fig12_Operational_Validation_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf,           bbox_inches="tight", facecolor="white")
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# =============================================================================
# Console summary
# =============================================================================
print("\n=== Operational thresholds ===")
print(f"  P(Flood) = 0.30  →  Rx5day = {P30:.1f} mm")
print(f"  P(Flood) = 0.60  →  Rx5day = {P60:.1f} mm")
print(f"  ML Rx5 threshold (annual) = {T_RX5:.1f} mm")
print(f"  ML SM threshold          = {T_SM:.3f} m³/m³")
print(f"  ML SWRI threshold        = {T_SWRI:.2f}")

print("\n=== Daily RF performance ===")
print(f"  AUC       : {auc_val:.3f}")
print(f"  POD       : {pod_val:.3f}")
print(f"  FAR       : {far_val:.3f}")
print(f"  Brier     : {brier:.4f}")
print(f"  Confusion : TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"  Class imbalance: {int(perf['n_pos_test'])} positive out of "
      f"{int(perf['n_test'])} test days "
      f"({100*int(perf['n_pos_test'])/int(perf['n_test']):.2f}%)")

plt.show()
```

    Loading prep outputs...
      P30 threshold        : 15.3 mm
      P60 threshold        : 43.0 mm
      ML Rx5 threshold     : 124.2 mm
      RF AUC / POD / FAR   : 0.823 / 0.000 / 0.000
      Confusion            : TP=0  FP=0  FN=29  TN=4898
      Brier score          : 0.0057
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig12_Operational_Validation_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig12_Operational_Validation_v2.pdf
    
    === Operational thresholds ===
      P(Flood) = 0.30  →  Rx5day = 15.3 mm
      P(Flood) = 0.60  →  Rx5day = 43.0 mm
      ML Rx5 threshold (annual) = 124.2 mm
      ML SM threshold          = 0.323 m³/m³
      ML SWRI threshold        = 3.10
    
    === Daily RF performance ===
      AUC       : 0.823
      POD       : 0.000
      FAR       : 0.000
      Brier     : 0.0057
      Confusion : TP=0  FP=0  FN=29  TN=4898
      Class imbalance: 29 positive out of 4927 test days (0.59%)
    


    
![png](output_36_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 13 — Spatial flood-risk mosaic and proposed monitoring network
=========================================================================
Advanced nature-journal style:
  - Tight, balanced 2-panel layout
  - Refined colour palettes (YlOrRd risk ramp, sage-cream terrain)
  - Soft drop-shadow for basin, gold-accented peak-risk callout
  - Magenta dashed border (matches Fig 1)
  - River-constrained peak risk detection
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT     = Path(r"G:\NUIST_YTU_Water")
PREP     = ROOT / "prep_revised_v2"
SHP_DIR  = ROOT / "output_shapefiles"
FIG_DIR  = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RISK_NC   = PREP / "risk" / "fig13_risk_mosaic.nc"
BASIN_SHP = SHP_DIR / "custom_basin.shp"
RIVER_SHP = SHP_DIR / "clipped_rivers.shp"
STN_CSV   = PREP / "fig01" / "fig01_stations_retained14.csv"
ETOPO     = Path(r"G:\ETOPO1_Bed_g_gmt4.nc")

# -----------------------------------------------------------------------------
# Load risk mosaic
# -----------------------------------------------------------------------------
print("Loading risk mosaic...")
ds = xr.open_dataset(RISK_NC)
for c in ["latitude", "longitude"]:
    if c in ds.dims:
        ds = ds.rename({c: c[:3]})
lat  = ds.lat.values
lon  = ds.lon.values
risk = ds["risk_index"].values
ds.close()

# -----------------------------------------------------------------------------
# Shapefiles, stations, border, terrain
# -----------------------------------------------------------------------------
print("Loading shapefiles...")
basin  = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
rivers = gpd.read_file(RIVER_SHP).to_crs("EPSG:4326")
shweli = rivers[rivers["Sub_Name"].astype(str).str.contains("Shweli",    case=False, na=False)]
irraw  = rivers[rivers["Sub_Name"].astype(str).str.contains("Irrawaddy", case=False, na=False)]
stn_df = pd.read_csv(STN_CSV)

basin_geom  = basin.geometry.union_all()
shweli_geom = shweli.geometry.union_all()

minx, miny, maxx, maxy = basin.total_bounds
EXT = [minx - 0.35, maxx + 0.35, miny - 0.35, maxy + 0.35]

print("Loading ETOPO...")
dd = xr.open_dataset(ETOPO)
if "x" in dd.coords: dd = dd.rename({"x": "lon"})
if "y" in dd.coords: dd = dd.rename({"y": "lat"})
dem = dd["z"].squeeze().sel(lon=slice(EXT[0], EXT[1]),
                              lat=slice(EXT[2], EXT[3]))
dd.close()

dem_interp = dem.sortby("lat").sortby("lon").interp(
    lat=lat, lon=lon, method="linear").values
dem_land = np.where(dem_interp < 5, np.nan, dem_interp)

# Border (natural earth)
borders_path = shapereader.natural_earth('50m', 'cultural', 'admin_0_boundary_lines_land')
borders_gdf  = gpd.read_file(borders_path).to_crs("EPSG:4326")
basin_buf    = basin_geom.buffer(0.03)
borders_clip = borders_gdf[borders_gdf.intersects(basin_buf)].copy()
borders_clip["geometry"] = borders_clip.geometry.intersection(basin_buf)
borders_clip = borders_clip[~borders_clip.is_empty]

# -----------------------------------------------------------------------------
# Masking inside basin
# -----------------------------------------------------------------------------
print("Masking data to basin boundary...")
lon2d, lat2d = np.meshgrid(lon, lat)
pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
basin_path   = MplPath(np.array(basin_geom.exterior.coords))
inside_basin = basin_path.contains_points(pts).reshape(lon2d.shape)

risk_masked = np.where(inside_basin, risk, np.nan)
dem_masked  = np.where(inside_basin, dem_land, np.nan)

# -----------------------------------------------------------------------------
# Shweli-constrained peak risk
# -----------------------------------------------------------------------------
print("Finding peak risk cell ALONG Shweli River...")
shweli_buffer = shweli_geom.buffer(0.05)

inside_shweli = np.array([shweli_buffer.contains(Point(x, y))
                          for x, y in zip(lon2d.ravel(), lat2d.ravel())]
                         ).reshape(lon2d.shape)

risk_along_river = np.where(inside_shweli, risk_masked, np.nan)

if np.isnan(risk_along_river).all():
    print("Warning: no valid cells along river — falling back to basin.")
    risk_along_river = risk_masked

idx      = np.unravel_index(np.nanargmax(risk_along_river), risk_along_river.shape)
peak_lon = float(lon[idx[1]])
peak_lat = float(lat[idx[0]])
peak_val = float(risk_along_river[idx])
print(f"  Peak risk cell (River-constrained): "
      f"({peak_lon:.2f}°E, {peak_lat:.2f}°N) = {peak_val:.1f}")

# -----------------------------------------------------------------------------
# Monitoring sites
# -----------------------------------------------------------------------------
SITES = pd.DataFrame({
    "id":   ["C1", "C2", "C3", "C4", "M1", "M2", "M3", "M4", "M5", "M6"],
    "name": ["Longchuan", "Ruili", "Mangshi", "Lianghe",
             "Nam Kham", "Namtu", "Kyaukme", "Hsipaw", "Mongyai", "Momeik"],
    "Lon":  [97.79, 97.85, 98.57, 98.30, 97.75, 97.40, 97.03, 97.30, 98.03, 96.68],
    "Lat":  [24.30, 24.00, 24.43, 24.80, 23.90, 23.10, 22.54, 22.62, 22.35, 23.11],
    "role": ["optional", "primary", "secondary", "optional",
             "primary", "optional", "optional", "optional", "optional", "secondary"],
})

LABEL_OFFSET = {
    "C1": ( 0.10,  0.16), "C2": (-0.65,  0.16), "C3": ( 0.10,  0.16),
    "C4": ( 0.10,  0.16), "M1": ( 0.15, -0.28), "M2": (-0.65, -0.10),
    "M3": (-0.70,  0.02), "M4": ( 0.15,  0.16), "M5": ( 0.12, -0.26),
    "M6": (-0.60,  0.16),
}

# -----------------------------------------------------------------------------
# ADVANCED COLOUR PALETTES
# -----------------------------------------------------------------------------
# Risk mosaic: cream → amber → deep crimson
risk_cmap = LinearSegmentedColormap.from_list(
    "advanced_risk",
    [(0.00, "#FFF7EC"),
     (0.15, "#FDD49E"),
     (0.30, "#FDBB84"),
     (0.45, "#FC8D59"),
     (0.60, "#E34A33"),
     (0.80, "#B30000"),
     (1.00, "#4A0000")])

# Terrain: sage → pale sage → cream → warm sandstone
terrain_cmap = LinearSegmentedColormap.from_list(
    "advanced_terrain",
    [(0.00, "#3E5641"),
     (0.20, "#6D8B74"),
     (0.40, "#A3B18A"),
     (0.55, "#D4C7A8"),
     (0.75, "#C7AE88"),
     (0.90, "#B58A66"),
     (1.00, "#F5EFE0")])
terrain_cmap.set_bad(alpha=0)

risk_norm = Normalize(vmin=55, vmax=100)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'font.size': 16,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 17, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'legend.fontsize': 13, 'legend.frameon': True,
    'legend.framealpha': 0.97, 'legend.edgecolor': '0.3',
    'legend.borderpad': 0.6, 'legend.labelspacing': 0.5,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# -----------------------------------------------------------------------------
# Figure — tighter, more balanced layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 8.5), dpi=500,facecolor='white')

# Panel axes — tighter horizontal spread
ax_a = fig.add_axes([0.045, 0.20, 0.415, 0.72], projection=ccrs.PlateCarree())
ax_b = fig.add_axes([0.540, 0.20, 0.415, 0.72], projection=ccrs.PlateCarree())

# -----------------------------------------------------------------------------
# Shared backdrop with drop-shadow on basin
# -----------------------------------------------------------------------------
def draw_backdrop(ax, risk_alpha, hex_size, risk_edge_lw):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)

    # Terrain
    ax.pcolormesh(lon2d, lat2d, np.ma.masked_invalid(dem_masked),
                  cmap=terrain_cmap, vmin=0, vmax=4000,
                  shading='gouraud',
                  transform=ccrs.PlateCarree(), zorder=1, rasterized=True)

    # Drop-shadow for basin area (a soft dark halo behind the polygon edge)
    basin.plot(ax=ax, facecolor='none', edgecolor='#000000',
               linewidth=6.0, alpha=0.10,
               transform=ccrs.PlateCarree(), zorder=2)

    # Risk mosaic hexes
    valid = np.isfinite(risk_masked)
    ax.scatter(lon2d[valid], lat2d[valid],
               c=risk_masked[valid], cmap=risk_cmap, norm=risk_norm,
               s=hex_size, marker='h',
               edgecolors='#4A0000', linewidths=risk_edge_lw,
               alpha=risk_alpha,
               transform=ccrs.PlateCarree(), zorder=3, rasterized=True)

    # Rivers
    if not shweli.empty:
        shweli.plot(ax=ax, color='#4A148C', linewidth=3.2,
                    transform=ccrs.PlateCarree(), zorder=5)
    if not irraw.empty:
        irraw.plot(ax=ax, color='#0D47A1', linewidth=2.6,
                   transform=ccrs.PlateCarree(), zorder=5)

    # Magenta border
    if not borders_clip.empty:
        borders_clip.plot(ax=ax, color='magenta', linewidth=2.4,
                          linestyle=(0, (5, 3)),
                          transform=ccrs.PlateCarree(), zorder=7)

    # Crisp basin edge
    basin.plot(ax=ax, facecolor='none', edgecolor='black',
               linewidth=2.2, transform=ccrs.PlateCarree(), zorder=6)

    # Grat lines
    gl = ax.gridlines(draw_labels=True, lw=0.4, color='#555',
                      alpha=0.45, linestyle=':')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'weight': 'bold'}
    gl.ylabel_style = {'size': 14, 'weight': 'bold'}

# =============================================================================
# Panel (a) — Risk mosaic
# =============================================================================
draw_backdrop(ax_a, risk_alpha=0.94, hex_size=95, risk_edge_lw=0.15)

# Stations
for _, row in stn_df.iterrows():
    ax_a.plot(row["Lon"], row["Lat"], marker='^',
              color='#d32f2f', markersize=11,
              markeredgecolor='white', markeredgewidth=1.2,
              transform=ccrs.PlateCarree(), zorder=10)

# Peak cell with gold halo
ax_a.scatter([peak_lon], [peak_lat], s=620, facecolor='none',
             edgecolor='#FFD700', linewidth=4.0,
             transform=ccrs.PlateCarree(), zorder=14)
ax_a.scatter([peak_lon], [peak_lat], s=380, facecolor='#d32f2f',
             edgecolor='white', linewidth=2.6,
             transform=ccrs.PlateCarree(), zorder=15)

# Arrow + callout
corners = [
    (EXT[0] + 0.5,  EXT[3] - 0.5),
    (EXT[1] - 2.5,  EXT[3] - 0.5),
    (EXT[0] + 0.5,  EXT[2] + 0.5),
    (EXT[1] - 2.5,  EXT[2] + 0.5),
]
if peak_lon < 97.5:
    tx, ty = EXT[1] - 2.5, EXT[3] - 0.5
else:
    tx, ty = EXT[0] + 0.5, EXT[3] - 0.5

ha_choice = 'left' if tx < peak_lon else 'right'
va_choice = 'top'  if ty > peak_lat else 'bottom'

ax_a.annotate(
    f"Peak flood risk cell\n({peak_lon:.1f}°E, {peak_lat:.1f}°N)\nRisk = {peak_val:.1f}",
    xy=(peak_lon, peak_lat), xytext=(tx, ty),
    fontsize=14, fontweight='bold',
    ha=ha_choice, va=va_choice,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
              edgecolor='#FFD700', linewidth=1.8, alpha=0.97),
    arrowprops=dict(arrowstyle='-|>', color='#FFD700', lw=2.0,
                    connectionstyle='arc3,rad=0.15'),
    transform=ccrs.PlateCarree(), zorder=100)

ax_a.set_title("(a) Terrain-weighted flood-risk sensitivity",
               loc='left', pad=12, fontsize=19)

legend_a = [
    Line2D([0],[0], color='black',   lw=2.0, label='Basin boundary'),
    Line2D([0],[0], color='magenta', lw=2.4, ls=(0, (5, 3)),
           label='China–Myanmar border'),
    Line2D([0],[0], color='#4A148C', lw=3.2, label='Shweli River'),
    Line2D([0],[0], color='#0D47A1', lw=2.6, label='Ayeyarwady River'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#d32f2f',
           markeredgecolor='white', markersize=12,
           linestyle='None', label='Stations (n = 14)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#d32f2f',
           markeredgecolor='#FFD700', markeredgewidth=2.0, markersize=15,
           linestyle='None', label='Peak risk cell'),
]
ax_a.legend(handles=legend_a, loc='lower right', fontsize=13,
            framealpha=0.97, edgecolor='0.3', borderpad=0.7,
            facecolor='#FFFDF7')

# =============================================================================
# Panel (b) — Monitoring network
# =============================================================================
draw_backdrop(ax_b, risk_alpha=0.32, hex_size=70, risk_edge_lw=0.10)

role_style = {
    "primary":   dict(marker='o', size=420, face='#B00020', edge='white'),
    "secondary": dict(marker='s', size=340, face='#FF8C00', edge='white'),
    "optional":  dict(marker='^', size=250, face='#6b7a8f', edge='white'),
}

for _, row in SITES.iterrows():
    st = role_style[row["role"]]
    ax_b.scatter([row["Lon"]], [row["Lat"]], s=st["size"],
                 marker=st["marker"], facecolor=st["face"],
                 edgecolor=st["edge"], linewidth=1.9,
                 transform=ccrs.PlateCarree(), zorder=12)

    dx, dy = LABEL_OFFSET[row["id"]]
    ax_b.text(row["Lon"] + dx, row["Lat"] + dy,
              f"{row['id']}  {row['name']}",
              fontsize=12.5,
              fontweight='bold' if row["role"] == "primary" else 'normal',
              transform=ccrs.PlateCarree(), zorder=13,
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white',
                        edgecolor=st["face"], linewidth=1.1, alpha=0.94))

ax_b.set_title("(b) Proposed water-monitoring network",
               loc='left', pad=12, fontsize=19)

legend_b = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#B00020',
           markeredgecolor='white', markeredgewidth=1.8, markersize=16,
           linestyle='None', label='Primary (Ruili, Nam Kham)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#FF8C00',
           markeredgecolor='white', markeredgewidth=1.8, markersize=15,
           linestyle='None', label='Secondary (Mangshi, Momeik)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='#6b7a8f',
           markeredgecolor='white', markeredgewidth=1.8, markersize=14,
           linestyle='None', label='Optional candidate'),
]
ax_b.legend(handles=legend_b, loc='lower right', fontsize=13,
            framealpha=0.97, edgecolor='0.3', borderpad=0.7,
            facecolor='#FFFDF7')

# =============================================================================
# Colorbars — tighter, sitting flush under maps
# =============================================================================
sm_risk = ScalarMappable(norm=risk_norm, cmap=risk_cmap)
sm_risk.set_array([])
cax_risk = fig.add_axes([0.045, 0.085, 0.415, 0.018])
cb_risk  = fig.colorbar(sm_risk, cax=cax_risk, orientation='horizontal',
                         extend='both')
cb_risk.set_label("Terrain-weighted risk index",
                   fontsize=15, fontweight='bold', labelpad=6)
cb_risk.ax.tick_params(labelsize=13, width=1.0)
cb_risk.outline.set_linewidth(1.0)

sm_elev = ScalarMappable(norm=Normalize(0, 4000), cmap=terrain_cmap)
sm_elev.set_array([])
cax_elev = fig.add_axes([0.540, 0.085, 0.415, 0.018])
cb_elev  = fig.colorbar(sm_elev, cax=cax_elev, orientation='horizontal')
cb_elev.set_label("Elevation (m)", fontsize=15, fontweight='bold', labelpad=6)
cb_elev.ax.tick_params(labelsize=13, width=1.0)
cb_elev.outline.set_linewidth(1.0)

# =============================================================================
# Header ribbon (subtle top accent)
# =============================================================================
fig.patches.append(FancyBboxPatch(
    (0.045, 0.955), 0.91, 0.005,
    boxstyle="round,pad=0.0", transform=fig.transFigure, alpha=0,
    facecolor='#B00020', edgecolor='none', zorder=1000))

# =============================================================================
# Save
# =============================================================================
out_png = FIG_DIR / "Fig13_RiskMap_Monitoring_v2.png"
out_pdf = FIG_DIR / "Fig13_RiskMap_Monitoring_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved PNG: {out_png}")
print(f"✅ Saved PDF: {out_pdf}")

# =============================================================================
# Console summary
# =============================================================================
print("\n=== Risk mosaic summary ===")
n_valid = int(np.isfinite(risk).sum())
print(f"  Cells ≥ 55%         : {n_valid}")
print(f"  Peak risk cell      : ({peak_lon:.2f}°E, {peak_lat:.2f}°N)  "
      f"risk = {peak_val:.1f}")

from math import radians, cos, sin, asin, sqrt
def haversine(lo1, la1, lo2, la2):
    R = 6371.0
    lo1, la1, lo2, la2 = map(radians, [lo1, la1, lo2, la2])
    a = sin((la2-la1)/2)**2 + cos(la1)*cos(la2)*sin((lo2-lo1)/2)**2
    return 2 * R * asin(sqrt(a))

dists   = [haversine(peak_lon, peak_lat, lo, la)
           for lo, la in zip(SITES["Lon"], SITES["Lat"])]
nearest = SITES.iloc[int(np.argmin(dists))]
print(f"  Nearest site        : {nearest['name']}  ({min(dists):.0f} km)")

print("\n=== Monitoring network ===")
for role in ["primary", "secondary", "optional"]:
    sub   = SITES[SITES["role"] == role]
    names = ", ".join(sub["name"].tolist())
    print(f"  {role.capitalize():<10s}: {len(sub)} — {names}")

plt.show()
```

    Loading risk mosaic...
    Loading shapefiles...
    Loading ETOPO...
    Masking data to basin boundary...
    Finding peak risk cell ALONG Shweli River...
      Peak risk cell (River-constrained): (97.08°E, 23.42°N) = 77.4
    
    ✅ Saved PNG: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig13_RiskMap_Monitoring_v2.png
    ✅ Saved PDF: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\Fig13_RiskMap_Monitoring_v2.pdf
    
    === Risk mosaic summary ===
      Cells ≥ 55%         : 5214
      Peak risk cell      : (97.08°E, 23.42°N)  risk = 77.4
      Nearest site        : Namtu  (49 km)
    
    === Monitoring network ===
      Primary   : 2 — Ruili, Nam Kham
      Secondary : 2 — Mangshi, Momeik
      Optional  : 6 — Longchuan, Lianghe, Namtu, Kyaukme, Hsipaw, Mongyai
    


    
![png](output_38_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S1 — DEM, basin delineation, and station QC
=========================================================================
Panels:
  (a) ETOPO1 regional DEM with basin outline
  (b) AW3D30 basin-zoom DEM
  (c) Per-station valid-day heatmap (14 stations × 45 years)
  (d) All-15 stations map — Hkamti excluded

Reads:
    prep_revised_v2/fig01/fig01_metadata.json
    prep_revised_v2/fig01/fig01_stations_all15.csv
    prep_revised_v2/fig01/fig01_stations_retained14.csv
    trend/Extracted_CSV/{station}.csv
    output_shapefiles/custom_basin.shp
    data/AW3D30_DEM_Shweli.tif
    G:\\ETOPO1_Bed_g_gmt4.nc

Writes:
    CMIP6_Evaluation/output_shapefiles/FigS1_Data_QC_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# ---- Paths ----
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
SHP_DIR = ROOT / "output_shapefiles"
EXTRACT = ROOT / "trend" / "Extracted_CSV"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META_JSON   = PREP / "fig01" / "fig01_metadata.json"
STN_14      = PREP / "fig01" / "fig01_stations_retained14.csv"
STN_ALL     = PREP / "fig01" / "fig01_stations_all15.csv"
BASIN_SHP   = SHP_DIR / "custom_basin.shp"
DEM_PATH    = ROOT / "data" / "AW3D30_DEM_Shweli.tif"
ETOPO_PATH  = Path(r"G:\ETOPO1_Bed_g_gmt4.nc")

meta       = json.load(open(META_JSON))
basin      = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
stn_all    = pd.read_csv(STN_ALL)
stn_14     = pd.read_csv(STN_14)
excluded   = meta["stations_excluded"]

minx, miny, maxx, maxy = basin.total_bounds
EXT_FULL   = [minx - 1.0, maxx + 1.0, miny - 1.0, maxy + 1.0]
EXT_ZOOM   = [minx - 0.2, maxx + 0.2, miny - 0.2, maxy + 0.2]

# ---- Per-station valid-day matrix (station × year) ----
print("Computing per-station valid-day matrix...")
years = list(range(1981, 2026))
mat   = np.full((len(stn_14), len(years)), np.nan)
for i, (_, r) in enumerate(stn_14.iterrows()):
    fp = EXTRACT / f"{r['Station']}.csv"
    if not fp.exists():
        continue
    try:
        df = pd.read_csv(fp)
        df["year"] = df["year"].astype(int)
        # Count non-null rainfall days per year
        cnt = (df.assign(has=df["CHIRPS_rainfall"].notna())
                 .groupby("year")["has"].sum())
        for j, yr in enumerate(years):
            mat[i, j] = cnt.get(yr, 0)
    except Exception as e:
        print(f"  [WARN] {r['Station']}: {e}")

mat_pct = mat / 365.0 * 100.0   # % of year valid (approx)

# ---- ETOPO1 ----
print("Loading ETOPO1...")
dd = xr.open_dataset(ETOPO_PATH)
if "x" in dd.coords: dd = dd.rename({"x":"lon"})
if "y" in dd.coords: dd = dd.rename({"y":"lat"})
dem_reg = dd["z"].squeeze().sel(lon=slice(EXT_FULL[0], EXT_FULL[1]),
                                  lat=slice(EXT_FULL[2], EXT_FULL[3]))
dd.close()

# ---- AW3D30 (downsampled) ----
print("Loading AW3D30...")
with rasterio.open(DEM_PATH) as src:
    aw3d = src.read(1, out_shape=(src.height // 5, src.width // 5)).astype("float32")
    b = src.bounds
    aw_lon = np.linspace(b.left, b.right, aw3d.shape[1])
    aw_lat = np.linspace(b.bottom, b.top, aw3d.shape[0])

ls = LightSource(azdeg=315, altdeg=45)

# ---- Style ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 19, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5,
    'legend.fontsize': 12, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# ---- Figure — 2x2 ----
fig = plt.figure(figsize=(18, 12),dpi=500)
gs = fig.add_gridspec(
    2, 2,
    hspace=0.30, wspace=0.14,
    left=0.055, right=0.965, top=0.945, bottom=0.06,
)

# ==========================================================
# (a) ETOPO1 regional DEM
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_a.set_extent(EXT_FULL, crs=ccrs.PlateCarree())
ax_a.add_feature(cfeature.OCEAN, facecolor='#d9ebf5', zorder=0)

im = ax_a.imshow(dem_reg.values, origin='upper',
                  extent=[dem_reg.lon.min(), dem_reg.lon.max(),
                          dem_reg.lat.min(), dem_reg.lat.max()],
                  cmap='terrain', alpha=0.88,
                  transform=ccrs.PlateCarree(), zorder=1)
hs = ls.hillshade(dem_reg.values, vert_exag=2, dx=1, dy=1)
ax_a.imshow(hs, origin='upper',
              extent=[dem_reg.lon.min(), dem_reg.lon.max(),
                      dem_reg.lat.min(), dem_reg.lat.max()],
              cmap='gray', alpha=0.30,
              transform=ccrs.PlateCarree(), zorder=2)

ax_a.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333', zorder=3)
ax_a.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor='#333',
                  alpha=0.6, linestyle='--', zorder=3)

basin.plot(ax=ax_a, facecolor='none', edgecolor='red', linewidth=2.6, zorder=5)

gl = ax_a.gridlines(draw_labels=True, lw=0.4, color='gray', alpha=0.5,
                     linestyle=':')
gl.top_labels = False; gl.right_labels = False
gl.xlabel_style = {'size': 12.5, 'weight': 'bold'}
gl.ylabel_style = {'size': 12.5, 'weight': 'bold'}

ax_a.set_title("(a) ETOPO1 regional DEM (1 arc-min)",
               loc='left', pad=10)
cax_a = fig.add_axes([0.055, 0.508, 0.42, 0.014])
cb_a = fig.colorbar(im, cax=cax_a, orientation='horizontal')
cb_a.set_label("Elevation (m)", fontsize=13, fontweight='bold', labelpad=6)
cb_a.ax.tick_params(labelsize=11)

# ==========================================================
# (b) AW3D30 basin zoom
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_b.set_extent(EXT_ZOOM, crs=ccrs.PlateCarree())

# Downsample for display
step = 2
aw_disp = aw3d[::step, ::step]
aw_lon_d = aw_lon[::step]
aw_lat_d = aw_lat[::step]

im_b = ax_b.imshow(aw_disp, origin='upper',
                    extent=[aw_lon_d.min(), aw_lon_d.max(),
                            aw_lat_d.min(), aw_lat_d.max()],
                    cmap='terrain', alpha=0.92,
                    transform=ccrs.PlateCarree(), zorder=1)
hs_b = ls.hillshade(aw_disp, vert_exag=2, dx=1, dy=1)
ax_b.imshow(hs_b, origin='upper',
              extent=[aw_lon_d.min(), aw_lon_d.max(),
                      aw_lat_d.min(), aw_lat_d.max()],
              cmap='gray', alpha=0.30,
              transform=ccrs.PlateCarree(), zorder=2)

basin.plot(ax=ax_b, facecolor='none', edgecolor='red',
            linewidth=2.6, zorder=5)

gl_b = ax_b.gridlines(draw_labels=True, lw=0.4, color='gray', alpha=0.5,
                       linestyle=':')
gl_b.top_labels = False; gl_b.right_labels = False
gl_b.xlabel_style = {'size': 12.5, 'weight': 'bold'}
gl_b.ylabel_style = {'size': 12.5, 'weight': 'bold'}

ax_b.set_title("(b) AW3D30 DEM (~30 m, downsampled ÷5)",
               loc='left', pad=10)
cax_b = fig.add_axes([0.545, 0.508, 0.42, 0.014])
cb_b = fig.colorbar(im_b, cax=cax_b, orientation='horizontal')
cb_b.set_label("Elevation (m)", fontsize=13, fontweight='bold', labelpad=6)
cb_b.ax.tick_params(labelsize=11)

# ==========================================================
# (c) Per-station valid-day heatmap
# ==========================================================
ax_c = fig.add_subplot(gs[1, 0])

imc = ax_c.imshow(mat_pct, aspect='auto', cmap='YlGnBu',
                   vmin=0, vmax=100,
                   extent=[years[0]-0.5, years[-1]+0.5,
                            len(stn_14)-0.5, -0.5])

ax_c.set_yticks(np.arange(len(stn_14)))
ax_c.set_yticklabels(stn_14["Station"].tolist(), fontsize=12.5)
ax_c.set_xlabel("Year")
ax_c.set_title("(c) Station valid-day coverage (% of year)",
               loc='left', pad=10)
ax_c.set_xticks(range(1981, 2026, 5))

# Red contour where coverage < 60%
X, Y = np.meshgrid(years, np.arange(len(stn_14)))
low = np.where(mat_pct < 60, 1, 0)
ax_c.contour(X, Y, low, levels=[0.5], colors='red', linewidths=1.5)

cax_c = fig.add_axes([0.055, 0.075, 0.42, 0.014])
cb_c = fig.colorbar(imc, cax=cax_c, orientation='horizontal',
                     ticks=[0, 25, 50, 75, 100])
cb_c.set_label("Valid days (%)", fontsize=13, fontweight='bold', labelpad=6)
cb_c.ax.tick_params(labelsize=11)

# ==========================================================
# (d) All-15 stations map with Hkamti excluded
# ==========================================================
ax_d = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax_d.set_extent(EXT_ZOOM, crs=ccrs.PlateCarree())
ax_d.add_feature(cfeature.OCEAN, facecolor='#d9ebf5', zorder=0)
ax_d.add_feature(cfeature.LAND, facecolor='#f0ece1', zorder=0)

basin.plot(ax=ax_d, facecolor='#e0f3f8', edgecolor='black',
            linewidth=2.2, alpha=0.6, zorder=1)

# Retained stations — red triangles
for _, r in stn_14.iterrows():
    ax_d.scatter(r["Lon"], r["Lat"], marker='^', s=280,
                  color='#d32f2f', edgecolor='black', linewidth=1.4,
                  transform=ccrs.PlateCarree(), zorder=10)
    name = r["Station"].split("(")[0].strip()
    ax_d.text(r["Lon"] + 0.08, r["Lat"] + 0.08, name,
              fontsize=10.5, fontweight='bold',
              transform=ccrs.PlateCarree(), zorder=11,
              bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                        edgecolor='#666', linewidth=0.5, alpha=0.92))

# Excluded station — grey X with red circle
for _, r in stn_all[stn_all["Station"].isin(excluded)].iterrows():
    ax_d.scatter(r["Lon"], r["Lat"], marker='X', s=520,
                  color='#9e9e9e', edgecolor='#c62828', linewidth=2.5,
                  transform=ccrs.PlateCarree(), zorder=12)
    ax_d.text(r["Lon"] + 0.12, r["Lat"] - 0.20, f"{r['Station']} (excluded)",
              fontsize=10.5, fontweight='bold', color='#c62828',
              transform=ccrs.PlateCarree(), zorder=13,
              bbox=dict(boxstyle='round,pad=0.22', facecolor='#fff0f0',
                        edgecolor='#c62828', linewidth=0.8, alpha=0.95))

gl_d = ax_d.gridlines(draw_labels=True, lw=0.4, color='gray', alpha=0.5,
                       linestyle=':')
gl_d.top_labels = False; gl_d.right_labels = False
gl_d.xlabel_style = {'size': 12.5, 'weight': 'bold'}
gl_d.ylabel_style = {'size': 12.5, 'weight': 'bold'}

ax_d.set_title(f"(d) Stations — {len(stn_14)} retained, {len(excluded)} excluded",
               loc='left', pad=10)

legend_d = [
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#d32f2f',
           markeredgecolor='black', markeredgewidth=1.4, markersize=14,
           linestyle='None', label=f'Retained (n = {len(stn_14)})'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#9e9e9e',
           markeredgecolor='#c62828', markeredgewidth=2.5, markersize=16,
           linestyle='None', label=f'Excluded (n = {len(excluded)})'),
]
ax_d.legend(handles=legend_d, loc='lower right', fontsize=13,
             framealpha=0.95, edgecolor='0.3')

# ---- Save ----
out_png = FIG_DIR / "FigS1_Data_QC_v2.png"
out_pdf = FIG_DIR / "FigS1_Data_QC_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_png}")
print(f"✅ Saved: {out_pdf}")
print(f"   Excluded stations: {excluded}")
plt.show()
```

    Computing per-station valid-day matrix...
    Loading ETOPO1...
    Loading AW3D30...
    
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS1_Data_QC_v2.png
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS1_Data_QC_v2.pdf
       Excluded stations: ['Hkamti']
    


    
![png](output_40_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S2 — Additional event examples by category
=========================================================================
Three representative events (NOT the Fig. 7 examples):
  (a) Flash event
  (b) Short event
  (c) Prolonged event (2020 case)

Reads:
    prep_revised_v2/fig07/fig07_trigger_daily.csv
    prep_revised_v2/events/event_catalog_full.csv
    prep_revised_v2/thresholds/final_operational_thresholds.json

Writes:
    CMIP6_Evaluation/output_shapefiles/FigS2_Event_Examples_v2.png / .pdf
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

trig = pd.read_csv(PREP/"fig07"/"fig07_trigger_daily.csv",
                    parse_dates=["date"], index_col="date").sort_index()
ev   = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                    parse_dates=["start_date","end_date"])
thr  = json.load(open(PREP/"thresholds"/"final_operational_thresholds.json"))

T_RX5  = float(thr["Rx5day_Stn"])
T_SM   = float(thr["SM"])
T_SWRI = float(thr["SWRI"])

COL = {"P":"#1f4e8c","Rx5":"#00A087","SM":"#7E6148","SWRI":"#E64B35",
       "trigger":"#E9C46A","Flash":"#E66101","Short":"#FDB863",
       "Prolonged":"#B2182B"}

# ---- Pick three events (avoiding Fig. 7's examples) ----
# Fig. 7 typically used 2010 Prolonged + 2023 Short/Flash.
flash_pool = ev[ev["category"] == "Flash"].copy()
short_pool = ev[ev["category"] == "Short"].copy()
prol_pool  = ev[(ev["category"] == "Prolonged") &
                 (ev["duration_days"] >= 8)].copy()

# Prefer recent events for the supplement
ex_flash = (flash_pool.sort_values("start_date", ascending=False).iloc[0]
             if len(flash_pool) else ev.iloc[0])
ex_short = (short_pool.sort_values("start_date", ascending=False).iloc[0]
             if len(short_pool) else ev.iloc[0])
# Prefer 2020 Prolonged (documented); fallback to longest
p2020 = prol_pool[prol_pool["start_date"].dt.year == 2020]
ex_prol = (p2020.iloc[0] if len(p2020) else
            prol_pool.sort_values("duration_days", ascending=False).iloc[0]
            if len(prol_pool) else ev.iloc[0])

# ---- Style ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 16, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5,
    'legend.fontsize': 12.5, 'legend.frameon': True,
    'legend.framealpha': 0.95, 'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

# ---- Draw helper ----
def draw_event(ax, ev_row, title):
    s, e = ev_row["start_date"], ev_row["end_date"]
    dur   = int(ev_row["duration_days"])
    cat   = ev_row["category"]

    w0 = s - pd.Timedelta(days=15)
    w1 = e + pd.Timedelta(days=10)
    w  = trig.loc[w0:w1]
    if len(w) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center',
                transform=ax.transAxes)
        return

    # Primary: rain bars + Rx5 + SWRI
    ax.bar(w.index, w["P_stn"], width=0.9, color=COL["P"],
            alpha=0.45, edgecolor='black', linewidth=0.3,
            label="Daily P (mm)", zorder=2)
    ax.plot(w.index, w["Rx5_stn"], color=COL["Rx5"], lw=2.8,
            marker='o', ms=5, markeredgecolor='black', markeredgewidth=0.6,
            label="Rx5 (mm)", zorder=4)
    ax.plot(w.index, w["SWRI"], color=COL["SWRI"], lw=2.8, ls='--',
            marker='s', ms=4.5, markeredgecolor='black',
            markeredgewidth=0.5, label="SWRI", zorder=4)

    ax.axhline(T_RX5,  color=COL["Rx5"],  ls=':', lw=2.0, zorder=3,
                label=f"$T_{{Rx5}}$ = {T_RX5:.0f} mm")
    ax.axhline(T_SWRI, color=COL["SWRI"], ls=':', lw=2.0, zorder=3,
                label=f"$T_{{SWRI}}$ = {T_SWRI:.2f}")

    # Event window
    ax.axvspan(s, e, color=COL[cat], alpha=0.16, zorder=1)
    trig_days = w[w["trigger"] == 1].index
    for td in trig_days:
        ax.axvline(td, color=COL["trigger"], lw=1.0, alpha=0.55, zorder=1)

    # Secondary: SM
    ax2 = ax.twinx()
    ax2.plot(w.index, w["SM"], color=COL["SM"], lw=2.6,
             marker='^', ms=5, markeredgecolor='black',
             markeredgewidth=0.5, label="SM (m³/m³)", zorder=5)
    ax2.axhline(T_SM, color=COL["SM"], ls=':', lw=2.0, zorder=3,
                 label=f"$T_{{SM}}$ = {T_SM:.3f}")
    ax2.set_ylabel("Soil moisture (m³/m³)", color=COL["SM"],
                    fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COL["SM"])
    ax2.set_ylim(T_SM - 0.10, 0.55)

    ax.set_ylabel("Rainfall / SWRI (mm)")
    y_max = max(w["Rx5_stn"].max(), w["SWRI"].max(), T_RX5) * 1.18
    ax.set_ylim(0, y_max)
    ax.set_xlim(w0, w1)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    ax.set_title(f"{title}  [{s.strftime('%d %b %Y')} → "
                 f"{e.strftime('%d %b %Y')}, {dur} d]",
                 loc='left', pad=10, fontsize=17)

    ax.grid(ls=':', alpha=0.35, zorder=0)

    # Legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ev_patch = Patch(facecolor=COL[cat], alpha=0.30, edgecolor=COL[cat],
                      linewidth=1.5, label=f"{cat} window ({dur} d)")
    trig_patch = Patch(facecolor=COL["trigger"], alpha=0.6,
                        label="Trigger day")
    ax.legend(h1 + h2 + [ev_patch, trig_patch],
              l1 + l2 + [f"{cat} window ({dur} d)", "Trigger day"],
              loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=4, fontsize=11, framealpha=0.95, edgecolor='0.3')

# ---- Figure ----
fig = plt.figure(figsize=(18, 12),dpi=500)
gs = fig.add_gridspec(
    3, 1,
    hspace=0.55,
    left=0.075, right=0.92, top=0.955, bottom=0.055,
)

ax_a = fig.add_subplot(gs[0])
draw_event(ax_a, ex_flash, "(a) Flash-event example")

ax_b = fig.add_subplot(gs[1])
draw_event(ax_b, ex_short, "(b) Short-event example")

ax_c = fig.add_subplot(gs[2])
draw_event(ax_c, ex_prol, "(c) Prolonged-event example")

# Save
out_png = FIG_DIR / "FigS2_Event_Examples_v2.png"
out_pdf = FIG_DIR / "FigS2_Event_Examples_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_png}")
print(f"✅ Saved: {out_pdf}")
print(f"   (a) Flash : {ex_flash['start_date'].date()} → "
      f"{ex_flash['end_date'].date()}  ({ex_flash['duration_days']} d)")
print(f"   (b) Short : {ex_short['start_date'].date()} → "
      f"{ex_short['end_date'].date()}  ({ex_short['duration_days']} d)")
print(f"   (c) Prolonged : {ex_prol['start_date'].date()} → "
      f"{ex_prol['end_date'].date()}  ({ex_prol['duration_days']} d)")
plt.show()
```

    
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS2_Event_Examples_v2.png
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS2_Event_Examples_v2.pdf
       (a) Flash : 1986-10-08 → 1986-10-12  (5 d)
       (b) Short : 2023-10-12 → 2023-10-14  (3 d)
       (c) Prolonged : 1990-06-23 → 1990-08-03  (42 d)
    


    
![png](output_42_1.png)
    



```python

```


```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S3 — Fast vs Prolonged synoptic TCWV-anomaly contrast
=========================================================================
Panels:
  (a) Fast-event (Flash + Short) TCWV anomaly
  (b) Prolonged-event TCWV anomaly
  (c) Prolonged − Fast difference field

Reads:
    prep_revised_v2/events/event_catalog_full.csv
    NCEP/Processed/MM_pr_wtr_daily.nc

Writes:
    CMIP6_Evaluation/output_shapefiles/FigS3_Category_Synoptic_v2.png / .pdf
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
NCEP    = ROOT / "trend" / "Raw" / "NCEP" / "Processed"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXT = [85.0, 108.0, 8.0, 30.0]
SHWELI_BOX = [95.5, 22.0, 101.5, 26.5]

ev = pd.read_csv(PREP/"events"/"event_catalog_full.csv",
                  parse_dates=["start_date","end_date"])
ev["group"] = ev["category"].map({"Flash":"Fast","Short":"Fast",
                                    "Prolonged":"Prolonged"})

def event_days(df):
    days = []
    for _, r in df.iterrows():
        days.extend(pd.date_range(r["start_date"], r["end_date"], freq="D"))
    return pd.DatetimeIndex(sorted(set(days)))

fast_days = event_days(ev[ev["group"] == "Fast"])
prol_days = event_days(ev[ev["group"] == "Prolonged"])
print(f"Fast days: {len(fast_days)} | Prolonged days: {len(prol_days)}")

# ---- Load TCWV daily field ----
print("Loading TCWV daily field...")
ds = xr.open_dataset(NCEP/"MM_pr_wtr_daily.nc")
for c in ["latitude","longitude"]:
    if c in ds.dims: ds = ds.rename({c: c[:3]})
ds = ds.sortby("lat").sortby("lon")
var = list(ds.data_vars)[0]
da = ds[var]
for d in list(da.dims):
    if d not in ("lat","lon","time") and da.sizes[d] == 1:
        da = da.isel({d:0})
da = da.sel(lat=slice(EXT[2], EXT[3]), lon=slice(EXT[0], EXT[1]))
valid_times = pd.to_datetime(da.time.values)

# JJAS climatology
jjas = da.where(da.time.dt.month.isin([6,7,8,9]), drop=True)
clim = jjas.mean(dim="time", skipna=True)

# Category composites
def composite(days):
    sel = [d for d in days if d in valid_times]
    if len(sel) < 5: return None
    return da.sel(time=pd.DatetimeIndex(sel), method="nearest").mean(
        dim="time", skipna=True)

fast_c = composite(fast_days)
prol_c = composite(prol_days)
fast_a = fast_c - clim
prol_a = prol_c - clim
diff   = prol_a - fast_a
ds.close()

print(f"Fast anomaly range: {float(fast_a.min()):+.1f} … "
      f"{float(fast_a.max()):+.1f} mm")
print(f"Prolonged anomaly range: {float(prol_a.min()):+.1f} … "
      f"{float(prol_a.max()):+.1f} mm")
print(f"Difference range: {float(diff.min()):+.1f} … "
      f"{float(diff.max()):+.1f} mm")

# ---- Style ----
anom_cmap = LinearSegmentedColormap.from_list(
    "anom",
    [(0.00,"#08306b"),(0.15,"#2171b5"),(0.35,"#9ecae1"),(0.50,"#f7f7f7"),
     (0.65,"#fdae6b"),(0.85,"#e6550d"),(1.00,"#a50f15")])

diff_cmap = plt.get_cmap("RdBu_r")

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 12, 'legend.framealpha': 0.95,
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

def draw_base(ax, title):
    ax.set_extent(EXT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='#e8f0f6', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#f5f2ea', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#555',
                    linestyle='--', alpha=0.6, zorder=3)
    lon0, lat0, lon1, lat1 = SHWELI_BOX
    ax.add_patch(Rectangle((lon0,lat0), lon1-lon0, lat1-lat0,
                            facecolor='none', edgecolor='red',
                            linewidth=2.6, zorder=8))
    ax.text((lon0+lon1)/2, lat1+0.55, "Shweli basin",
            fontsize=11.5, fontweight='bold', color='red',
            ha='center', va='bottom', transform=ccrs.PlateCarree(),
            zorder=9,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                      edgecolor='red', linewidth=0.9, alpha=0.92))
    gl = ax.gridlines(draw_labels=True, lw=0.4, color='gray',
                       alpha=0.5, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'weight': 'bold'}
    ax.set_title(title, loc='left', pad=10, fontsize=18)

# ---- Figure — 1x3 ----
fig = plt.figure(figsize=(20, 5.5),dpi=500)
gs = fig.add_gridspec(
    1, 3,
    wspace=0.20,
    left=0.045, right=0.975, top=0.92, bottom=0.13,
)

# (a) Fast anomaly
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
draw_base(ax_a, "(a) Fast-event TCWV anomaly (Flash + Short)")
lon2d, lat2d = np.meshgrid(fast_a.lon.values, fast_a.lat.values)
vmax = max(abs(float(fast_a.min())), abs(float(fast_a.max())))
lv = np.linspace(-vmax, vmax, 21)
cf_a = ax_a.contourf(lon2d, lat2d, fast_a.values, levels=lv,
                      cmap=anom_cmap, extend='both',
                      transform=ccrs.PlateCarree(), zorder=2)
box_anom = float(fast_a.sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                              lon=slice(SHWELI_BOX[0], SHWELI_BOX[2])
                              ).mean().values)
ax_a.text(0.97, 0.04, f"Basin-mean: {box_anom:+.2f} mm",
          transform=ax_a.transAxes, ha='right', va='bottom',
          fontsize=13, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='#333', alpha=0.95), zorder=100)

cax_a = fig.add_axes([0.045, 0.08, 0.29, 0.020])
cb_a = fig.colorbar(cf_a, cax=cax_a, orientation='horizontal',
                     extend='both')
cb_a.set_label("TCWV anomaly (mm)", fontsize=13, fontweight='bold',
                labelpad=6)
cb_a.ax.tick_params(labelsize=11)

# (b) Prolonged anomaly
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
draw_base(ax_b, "(b) Prolonged-event TCWV anomaly")
cf_b = ax_b.contourf(lon2d, lat2d, prol_a.values, levels=lv,
                      cmap=anom_cmap, extend='both',
                      transform=ccrs.PlateCarree(), zorder=2)
box_anom_b = float(prol_a.sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                                lon=slice(SHWELI_BOX[0], SHWELI_BOX[2])
                                ).mean().values)
ax_b.text(0.97, 0.04, f"Basin-mean: {box_anom_b:+.2f} mm",
          transform=ax_b.transAxes, ha='right', va='bottom',
          fontsize=13, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='#333', alpha=0.95), zorder=100)

cax_b = fig.add_axes([0.355, 0.08, 0.29, 0.020])
cb_b = fig.colorbar(cf_b, cax=cax_b, orientation='horizontal',
                     extend='both')
cb_b.set_label("TCWV anomaly (mm)", fontsize=13, fontweight='bold',
                labelpad=6)
cb_b.ax.tick_params(labelsize=11)

# (c) Difference field
ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
draw_base(ax_c, "(c) Prolonged − Fast difference")
vmax_d = max(abs(float(diff.min())), abs(float(diff.max())), 0.1)
lv_d = np.linspace(-vmax_d, vmax_d, 21)
cf_c = ax_c.contourf(lon2d, lat2d, diff.values, levels=lv_d,
                      cmap=diff_cmap, extend='both',
                      transform=ccrs.PlateCarree(), zorder=2)
# Zero contour in black
ax_c.contour(lon2d, lat2d, diff.values, levels=[0],
              colors='black', linewidths=2.0,
              transform=ccrs.PlateCarree(), zorder=5)
box_anom_c = float(diff.sel(lat=slice(SHWELI_BOX[1], SHWELI_BOX[3]),
                              lon=slice(SHWELI_BOX[0], SHWELI_BOX[2])
                              ).mean().values)
ax_c.text(0.97, 0.04,
          f"Basin-mean difference: {box_anom_c:+.2f} mm",
          transform=ax_c.transAxes, ha='right', va='bottom',
          fontsize=13, fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='#333', alpha=0.95), zorder=100)

cax_c = fig.add_axes([0.665, 0.08, 0.29, 0.020])
cb_c = fig.colorbar(cf_c, cax=cax_c, orientation='horizontal',
                     extend='both')
cb_c.set_label("ΔTCWV anomaly (mm)", fontsize=13, fontweight='bold',
                labelpad=6)
cb_c.ax.tick_params(labelsize=11)

# Shared legend
legend = [
    Line2D([0], [0], color='red', lw=2.6, label='Shweli focus box'),
    Line2D([0], [0], color='black', lw=2.0, label='Zero anomaly contour'),
]
#fig.legend(handles=legend, loc='upper center',
#           bbox_to_anchor=(0.5, 0.975), ncol=2, fontsize=12,
 #          framealpha=0.95, edgecolor='0.3')

# Save
out_png = FIG_DIR / "FigS3_Category_Synoptic_v2.png"
out_pdf = FIG_DIR / "FigS3_Category_Synoptic_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_png}")
print(f"✅ Saved: {out_pdf}")
plt.show()
```

    Fast days: 19 | Prolonged days: 116
    Loading TCWV daily field...
    Fast anomaly range: -21.8 … -0.6 mm
    Prolonged anomaly range: -1.5 … +2.7 mm
    Difference range: +0.2 … +23.5 mm
    
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS3_Category_Synoptic_v2.png
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS3_Category_Synoptic_v2.pdf
    


    
![png](output_45_1.png)
    



```python

```


```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S4 — RF hyperparameter and class-imbalance diagnostics
=========================================================================
Panels:
  (a) n_estimators sweep → AUC, POD, FAR
  (b) Decision threshold sweep → POD, FAR, F1
  (c) Feature importance stability across 10 random seeds
  (d) Class-weight effect on ROC

Reads:
    prep_revised_v2/fig12/fig12_daily_labelled.csv

Writes:
    CMIP6_Evaluation/output_shapefiles/FigS4_RF_Diagnostics_v2.png / .pdf
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit

ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"
FIG_DIR = ROOT / "CMIP6_Evaluation" / "output_shapefiles"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load daily data + features ----
print("Loading daily data...")
df = pd.read_csv(PREP/"fig12"/"fig12_daily_labelled.csv",
                  parse_dates=["date"], index_col="date")

# Build features (same as Fig. 12)
df["Rx5_lag1"] = df["Rx5_stn"].shift(1)
df["Rx5_lag3"] = df["Rx5_stn"].shift(3)
df["ante5"]    = df["P_stn"].rolling(5).sum()
df["SM_lag1"]  = df["SM"].shift(1)
df["SM_anom"]  = df["SM"] - df["SM"].rolling(30, min_periods=15).mean()

FEATS = ["Rx5_stn","Rx5_lag1","Rx5_lag3","ante5","SM","SM_lag1","SM_anom"]
data  = df.dropna(subset=FEATS).copy()
X     = data[FEATS].values
y     = data["flood_day"].values
print(f"  n = {len(y)}, positives = {int(y.sum())}, "
      f"rate = {100*y.mean():.3f}%")

# Fixed split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr, te = next(sss.split(X, y))

# ---- Style ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial','Helvetica','DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 12, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

fig = plt.figure(figsize=(18, 12),dpi=500)
gs = fig.add_gridspec(
    2, 2,
    hspace=0.36, wspace=0.24,
    left=0.07, right=0.965, top=0.945, bottom=0.065,
)

# ==========================================================
# (a) n_estimators sweep
# ==========================================================
ax_a = fig.add_subplot(gs[0, 0])

n_est_vals = [50, 100, 200, 300, 500, 800]
res_a = []
for n_est in n_est_vals:
    rf = RandomForestClassifier(n_estimators=n_est, class_weight="balanced",
                                 random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    prob = rf.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y[te], prob)
    a = auc(fpr, tpr)
    tp = int(((pred == 1) & (y[te] == 1)).sum())
    fn = int(((pred == 0) & (y[te] == 1)).sum())
    fp = int(((pred == 1) & (y[te] == 0)).sum())
    tn = int(((pred == 0) & (y[te] == 0)).sum())
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    res_a.append(dict(n=n_est, auc=a, pod=pod, far=far))

res_df = pd.DataFrame(res_a)
print("\nn_estimators sweep:")
print(res_df.round(3).to_string(index=False))

ax_a.plot(res_df["n"], res_df["auc"], 'o-', color='#c62828',
          lw=3.0, ms=10, markeredgecolor='black', markeredgewidth=1.0,
          label="AUC", zorder=4)
ax_a.plot(res_df["n"], res_df["pod"], 's-', color='#1b5e20',
          lw=2.4, ms=9, markeredgecolor='black', markeredgewidth=0.9,
          label="POD (hit rate)", zorder=3)
ax_a.plot(res_df["n"], res_df["far"], '^-', color='#ef6c00',
          lw=2.4, ms=9, markeredgecolor='black', markeredgewidth=0.9,
          label="FAR (false alarm)", zorder=3)

ax_a.axhline(0.5, color='gray', ls='--', lw=1.4, zorder=1)
ax_a.set_xlabel("n_estimators")
ax_a.set_ylabel("Score")
ax_a.set_ylim(-0.02, 1.05)
ax_a.set_title("(a) n_estimators sweep — AUC, POD, FAR", loc='left', pad=10)
ax_a.grid(ls=':', alpha=0.35)
ax_a.set_axisbelow(True)
ax_a.legend(loc='center right', fontsize=12, framealpha=0.95,
             edgecolor='0.3')

# ==========================================================
# (b) Decision threshold sweep
# ==========================================================
ax_b = fig.add_subplot(gs[0, 1])

# Fit one reference RF
rf_ref = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=42, n_jobs=-1).fit(X[tr], y[tr])
prob_te = rf_ref.predict_proba(X[te])[:, 1]

thresholds = np.linspace(0.05, 0.95, 40)
pod_b, far_b, f1_b = [], [], []
for th in thresholds:
    pred = (prob_te >= th).astype(int)
    tp = int(((pred == 1) & (y[te] == 1)).sum())
    fn = int(((pred == 0) & (y[te] == 1)).sum())
    fp = int(((pred == 1) & (y[te] == 0)).sum())
    tn = int(((pred == 0) & (y[te] == 0)).sum())
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * prec * pod / (prec + pod)) if (prec + pod) > 0 else 0
    pod_b.append(pod); far_b.append(far); f1_b.append(f1)

ax_b.plot(thresholds, pod_b, color='#1b5e20', lw=2.8,
           label="POD (hit rate)")
ax_b.plot(thresholds, far_b, color='#ef6c00', lw=2.8,
           label="FAR (false alarm)")
ax_b.plot(thresholds, f1_b,  color='#c62828', lw=2.8, ls='--',
           label="F1 score")
ax_b.axvline(0.50, color='black', ls=':', lw=1.6,
              label="Default threshold = 0.50")

ax_b.set_xlabel("Decision probability threshold")
ax_b.set_ylabel("Score")
ax_b.set_xlim(0.05, 0.95)
ax_b.set_ylim(-0.02, 1.05)
ax_b.set_title("(b) Decision-threshold sensitivity", loc='left', pad=10)
ax_b.grid(ls=':', alpha=0.35)
ax_b.set_axisbelow(True)
ax_b.legend(loc='upper right', fontsize=12, framealpha=0.95,
             edgecolor='0.3')

# ==========================================================
# (c) Feature importance stability across seeds
# ==========================================================
ax_c = fig.add_subplot(gs[1, 0])

seeds = list(range(10))
imp_mat = np.zeros((len(seeds), len(FEATS)))
for i, s in enumerate(seeds):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=s, n_jobs=-1).fit(X[tr], y[tr])
    imp_mat[i, :] = rf.feature_importances_

# Violin per feature
positions = np.arange(len(FEATS))
vp = ax_c.violinplot([imp_mat[:, i] for i in range(len(FEATS))],
                       positions=positions, widths=0.75,
                       showmeans=False, showmedians=True,
                       showextrema=False)
cmap_list = plt.get_cmap("tab10").colors
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(cmap_list[i % 10])
    body.set_edgecolor("black")
    body.set_linewidth(1.2)
    body.set_alpha(0.70)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(2.6)

# Individual seed dots
rng = np.random.RandomState(42)
for i in range(len(FEATS)):
    jitter = rng.normal(0, 0.045, len(seeds))
    ax_c.scatter(positions[i] + jitter, imp_mat[:, i],
                  s=42, color="white", edgecolor="black",
                  linewidth=0.8, alpha=0.85, zorder=5)

ax_c.set_xticks(positions)
ax_c.set_xticklabels(FEATS, fontsize=12.5, rotation=30, ha='right')
ax_c.set_ylabel("Gini importance")
ax_c.set_title("(c) Feature-importance stability (10 seeds, 300 trees)",
                loc='left', pad=10)
ax_c.grid(axis='y', ls=':', alpha=0.35)
ax_c.set_axisbelow(True)

# ==========================================================
# (d) Class-weight effect on ROC
# ==========================================================
ax_d = fig.add_subplot(gs[1, 1])

roc_specs = [
    ("balanced",   "#c62828", "-",  3.0),
    (None,         "#1f4e8c", "--", 3.0),
]
for cw, color, ls, lw in roc_specs:
    rf = RandomForestClassifier(n_estimators=500, class_weight=cw,
                                  random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    prob = rf.predict_proba(X[te])[:, 1]
    fpr, tpr, _ = roc_curve(y[te], prob)
    a = auc(fpr, tpr)
    label = "class_weight='balanced'" if cw == "balanced" else "class_weight=None"
    ax_d.plot(fpr, tpr, color=color, ls=ls, lw=lw,
               label=f"{label}   AUC = {a:.3f}")

ax_d.plot([0, 1], [0, 1], color='gray', ls=':', lw=1.6,
           label="No-skill (AUC = 0.5)")
ax_d.set_xlim(-0.01, 1.01)
ax_d.set_ylim(-0.01, 1.01)
ax_d.set_aspect('equal')
ax_d.set_xlabel("False positive rate")
ax_d.set_ylabel("True positive rate")
ax_d.set_title("(d) Class-weight effect on ROC", loc='left', pad=10)
ax_d.grid(ls=':', alpha=0.35)
ax_d.set_axisbelow(True)
ax_d.legend(loc='lower right', fontsize=12, framealpha=0.95,
             edgecolor='0.3')

# ---- Save ----
out_png = FIG_DIR / "FigS4_RF_Diagnostics_v2.png"
out_pdf = FIG_DIR / "FigS4_RF_Diagnostics_v2.pdf"
fig.savefig(out_png, dpi=400, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_png}")
print(f"✅ Saved: {out_pdf}")

# ---- Summary ----
print("\n=== RF diagnostic summary ===")
print(f"  Test set     : {len(y[te])} days "
      f"({int(y[te].sum())} positives)")
print(f"  n_estimators: AUC stable from 200 onwards")
print(f"  Best F1      : {max(f1_b):.3f} at threshold "
      f"{thresholds[int(np.argmax(f1_b))]:.2f}")
print(f"  Default (0.50) POD: {pod_b[list(thresholds).index(min(thresholds, key=lambda x: abs(x-0.5)))]:.3f}")
print(f"  Default (0.50) FAR: {far_b[list(thresholds).index(min(thresholds, key=lambda x: abs(x-0.5)))]:.3f}")
plt.show()
```

    Loading daily data...
      n = 16422, positives = 96, rate = 0.585%
    
    n_estimators sweep:
      n   auc  pod  far
     50 0.734  0.0  0.0
    100 0.775  0.0  0.0
    200 0.824  0.0  0.0
    300 0.823  0.0  0.0
    500 0.818  0.0  0.0
    800 0.814  0.0  0.0
    
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS4_RF_Diagnostics_v2.png
    ✅ Saved: G:\NUIST_YTU_Water\CMIP6_Evaluation\output_shapefiles\FigS4_RF_Diagnostics_v2.pdf
    
    === RF diagnostic summary ===
      Test set     : 4927 days (29 positives)
      n_estimators: AUC stable from 200 onwards
      Best F1      : 0.103 at threshold 0.14
      Default (0.50) POD: 0.000
      Default (0.50) FAR: 0.000
    


    
![png](output_47_1.png)
    



```python

```


```python
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT    = Path(r"G:\NUIST_YTU_Water")
PREP    = ROOT / "prep_revised_v2"

# Define output directories for Main and Supplementary
OUT_MAIN = ROOT / "CMIP6_Evaluation" / "output_tables" / "Main"
OUT_SUPP = ROOT / "CMIP6_Evaluation" / "output_tables" / "Supplementary"

OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_SUPP.mkdir(parents=True, exist_ok=True)

print("Loading generated data from prep_revised_v2...")

# Load the actual computed data
stn_14       = pd.read_csv(PREP/"fig01"/"fig01_stations_retained14.csv")
annual_thr   = pd.read_csv(PREP/"tables"/"table04_thresholds_annual_scale.csv")
daily_thr    = pd.read_csv(PREP/"tables"/"table04_thresholds_daily_scale.csv")
thresholds   = json.load(open(PREP/"thresholds"/"final_operational_thresholds.json"))

event_catalog = pd.read_csv(PREP/"events"/"event_catalog_full.csv", parse_dates=['start_date', 'end_date'])
trends        = pd.read_csv(PREP/"tables"/"table07_trends_annual.csv")
ks_tests      = pd.read_csv(PREP/"tables"/"table07_ks_early_late.csv")
climate_corr  = pd.read_csv(PREP/"tables"/"table08_climate_correlations.csv")
rf_perf       = pd.read_csv(PREP/"tables"/"table09_rf_daily_performance.csv").iloc[0]
rf_imp        = pd.read_csv(PREP/"fig11"/"fig11_rf_importance.csv")

# =============================================================================
# MAIN MANUSCRIPT TABLES (1 to 3)
# =============================================================================
print("\nGenerating Main Tables (1-3)...")

# --- Table 1: Station Metadata ---
stn_14['Status'] = 'Retained'
hkamti = pd.DataFrame([{'Station': 'Hkamti', 'Country': 'Myanmar', 'Lat': 25.85, 'Lon': 95.69, 'Status': 'Excluded (QC)'}])
t1 = pd.concat([stn_14, hkamti], ignore_index=True)
t1 = t1[['Station', 'Country', 'Lat', 'Lon', 'Status']]
t1.to_csv(OUT_MAIN / "Table1_Station_Metadata.csv", index=False)

# --- Table 2: Threshold Variable Evaluation ---
t2_annual = annual_thr[['variable', 'AUC', 'threshold_youden', 'POD_youden', 'youden_J', 'FAR_youden']].copy()
t2_annual.columns = ['Variable', 'AUC', 'Threshold', 'POD', 'Youden_J', 'FAR']
t2_annual['Scale'] = 'Annual'

t2_daily = daily_thr[['variable', 'AUC', 'threshold_youden', 'POD_youden', 'youden_J', 'FAR_youden']].copy()
t2_daily.columns = ['Variable', 'AUC', 'Threshold', 'POD', 'Youden_J', 'FAR']
t2_daily['Scale'] = 'Daily'

t2 = pd.concat([t2_annual, t2_daily], ignore_index=True)
t2 = t2[['Scale', 'Variable', 'AUC', 'Threshold', 'POD', 'Youden_J', 'FAR']]
t2.to_csv(OUT_MAIN / "Table2_Threshold_Evaluation.csv", index=False)

# --- Table 3: Final Operational Thresholds ---
t3 = pd.DataFrame([
    ["Rx1day", round(thresholds['Rx1day_Stn'], 2), "mm", "Annual", "Daily extreme rainfall trigger"],
    ["Rx5day", round(thresholds['Rx5day_Stn'], 2), "mm", "Annual", "Multi-day accumulation trigger"],
    ["SM", round(thresholds['SM'], 3), "m³/m³", "Daily", "Catchment saturation trigger"],
    ["SWRI", round(thresholds['SWRI'], 3), "Index", "Daily", "Combined rainfall-saturation trigger"]
], columns=["Variable", "Threshold", "Unit", "Calibration_Scale", "Physical_Meaning"])
t3.to_csv(OUT_MAIN / "Table3_Final_Operational_Thresholds.csv", index=False)


# =============================================================================
# SUPPLEMENTARY TABLES (S1 to S6)
# =============================================================================
print("\nGenerating Supplementary Tables (S1-S6)...")

# --- Table S1: Full Event Catalog ---
t_s1 = event_catalog[['event_id', 'start_date', 'end_date', 'duration_days', 'category', 'peak_SWRI', 'rain_sum_mm']].copy()
t_s1['start_date'] = t_s1['start_date'].dt.strftime('%Y-%m-%d')
t_s1['end_date'] = t_s1['end_date'].dt.strftime('%Y-%m-%d')
t_s1['Matched_Documented'] = event_catalog['matched'].apply(lambda x: 'Yes' if x == 1 else 'No')
t_s1.columns = ['Event_ID', 'Start_Date', 'End_Date', 'Duration_days', 'Category', 'Peak_SWRI', 'Rain_Sum_mm', 'Matched_Documented']
t_s1.to_csv(OUT_SUPP / "Table_S1_Full_Event_Catalog.csv", index=False)

# --- Table S2: Category Summary ---
cat_summary = event_catalog.groupby('category').agg(
    Count=('event_id', 'size'),
    Mean_Duration=('duration_days', 'mean'),
    Median_Rain_Sum=('rain_sum_mm', 'median'),
    Median_Peak_SWRI=('peak_SWRI', 'median')
).reset_index()

if 'Flash' not in cat_summary['category'].values:
    flash_row = pd.DataFrame([{'category': 'Flash', 'Count': 0, 'Mean_Duration': np.nan, 'Median_Rain_Sum': np.nan, 'Median_Peak_SWRI': np.nan}])
    cat_summary = pd.concat([cat_summary, flash_row], ignore_index=True)

total_row = pd.DataFrame([{
    'category': 'Total',
    'Count': cat_summary['Count'].sum(),
    'Mean_Duration': event_catalog['duration_days'].mean(),
    'Median_Rain_Sum': event_catalog['rain_sum_mm'].median(),
    'Median_Peak_SWRI': event_catalog['peak_SWRI'].median()
}])
t_s2 = pd.concat([cat_summary, total_row], ignore_index=True)
t_s2['Percentage'] = (t_s2['Count'] / event_catalog.shape[0] * 100).round(1)
t_s2 = t_s2[['category', 'Count', 'Percentage', 'Mean_Duration', 'Median_Rain_Sum', 'Median_Peak_SWRI']]
t_s2.columns = ['Category', 'Count', 'Percentage', 'Mean_Duration_days', 'Median_Rain_Sum_mm', 'Median_Peak_SWRI']
t_s2.to_csv(OUT_SUPP / "Table_S2_Category_Summary.csv", index=False)

# --- Table S3: Trend Analysis Results ---
t_s3 = trends[['variable', 'slope', 'p_mk']].copy()
t_s3['Significance'] = t_s3['p_mk'].apply(lambda p: 'Yes' if p < 0.05 else 'No')
t_s3.columns = ['Variable', 'Sen_Slope_per_decade', 'MK_P_Value', 'Significance']

ks_renamed = ks_tests[['variable', 'ks_stat', 'ks_p']].rename(columns={'variable': 'Variable', 'ks_stat': 'KS_Stat', 'ks_p': 'KS_P_Value'})
t_s3 = pd.merge(t_s3, ks_renamed, on='Variable', how='left')
t_s3 = t_s3[['Variable', 'Sen_Slope_per_decade', 'MK_P_Value', 'Significance', 'KS_Stat', 'KS_P_Value']]
t_s3.to_csv(OUT_SUPP / "Table_S3_Trend_Analysis.csv", index=False)

# --- Table S4: Climate Driver Correlation & RF Importance ---
t_s4 = climate_corr[['index', 'pearson_r', 'pearson_p', 'spearman_rho']].copy()
t_s4.columns = ['Index', 'Pearson_r', 'P_Value', 'Spearman_Rho']

rf_renamed = rf_imp.rename(columns={'feature': 'Index', 'importance': 'RF_Importance'})
t_s4 = pd.merge(t_s4, rf_renamed, on='Index', how='left')
t_s4.to_csv(OUT_SUPP / "Table_S4_Climate_Drivers.csv", index=False)

# --- Table S5: Daily Random Forest Performance ---
t_s5 = pd.DataFrame([
    ["AUC", round(rf_perf['AUC'], 3), "High discriminative skill"],
    ["POD", round(rf_perf['POD'], 3), "Failed to detect positive cases"],
    ["FAR", round(rf_perf['FAR'], 3), "No false alarms (all predicted no-flood)"],
    ["Brier_Score", round(rf_perf['brier'], 4), "Deceptively low due to majority class"],
    ["True_Positives", int(rf_perf['TP']), "-"],
    ["False_Negatives", int(rf_perf['FN']), "All flood days missed"],
    ["True_Negatives", int(rf_perf['TN']), "-"],
    ["False_Positives", int(rf_perf['FP']), "-"]
], columns=["Metric", "Value", "Notes"])
t_s5.to_csv(OUT_SUPP / "Table_S5_RF_Performance.csv", index=False)

# --- Table S6: Proposed Monitoring Network ---
t_s6 = pd.DataFrame([
    ["C2","Ruili","China","Primary",24.00,97.85,"Upstream capture of high-risk corridor"],
    ["M1","Nam Kham","Myanmar","Primary",23.90,97.75,"Downstream propagation monitoring"],
    ["C3","Mangshi","China","Secondary",24.43,98.57,"Auxiliary eastern tributary coverage"],
    ["M6","Momeik","Myanmar","Secondary",23.11,96.68,"Auxiliary southern lowland coverage"],
    ["C1","Longchuan","China","Optional",24.30,97.79,"Gap filling"],
    ["C4","Lianghe","China","Optional",24.80,98.30,"Gap filling"],
    ["M2","Namtu","Myanmar","Optional",23.10,97.40,"Closest to peak risk cell (49km)"],
    ["M3","Kyaukme","Myanmar","Optional",22.54,97.03,"Gap filling"],
    ["M4","Hsipaw","Myanmar","Optional",22.62,97.30,"Gap filling"],
    ["M5","Mongyai","Myanmar","Optional",22.35,98.03,"Gap filling"]
], columns=["Site_ID", "Site_Name", "Country", "Role", "Lat", "Lon", "Reason"])
t_s6.to_csv(OUT_SUPP / "Table_S6_Monitoring_Network.csv", index=False)

print(f"\n✅ SUCCESS: All tables exported!")
print(f"   Main Tables (1-3) -> {OUT_MAIN}")
print(f"   Supplementary (S1-S6) -> {OUT_SUPP}")
```

    Loading generated data from prep_revised_v2...
    
    Generating Main Tables (1-3)...
    
    Generating Supplementary Tables (S1-S6)...
    
    ✅ SUCCESS: All tables exported!
       Main Tables (1-3) -> G:\NUIST_YTU_Water\CMIP6_Evaluation\output_tables\Main
       Supplementary (S1-S6) -> G:\NUIST_YTU_Water\CMIP6_Evaluation\output_tables\Supplementary
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

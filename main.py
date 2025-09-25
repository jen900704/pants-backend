# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import random

app = FastAPI()

# ── CORS（讓 github.io 可以叫到 Render 後端） ───────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 讀取資料（啟動時） ─────────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "data/metadata.xlsx")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    盡量把欄位名標準化成前端會用到的名稱。
    """
    rename_map = {}
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*cands, default=None):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return default

    # 目標標準欄位
    id_col     = pick("PanTS ID", "case_id", "id", default=None)
    sex_col    = pick("sex", default=None)
    age_col    = pick("age", default=None)
    tumor_col  = pick("tumor", "tumor_flag", default=None)
    ct_col     = pick("ct phase", "ct_phase", "phase", default=None)
    mfr_col    = pick("manufacturer", "vendor", default=None)
    model_col  = pick("manufacturer model", "model", default=None)
    year_col   = pick("study year", "year", "study_year", default=None)

    if id_col:    rename_map[id_col]   = "PanTS ID"
    if sex_col:   rename_map[sex_col]  = "sex"
    if age_col:   rename_map[age_col]  = "age"
    if tumor_col: rename_map[tumor_col]= "tumor"
    if ct_col:    rename_map[ct_col]   = "ct phase"
    if mfr_col:   rename_map[mfr_col]  = "manufacturer"
    if model_col: rename_map[model_col]= "manufacturer model"
    if year_col:  rename_map[year_col] = "study year"

    out = df.rename(columns=rename_map).copy()

    # 補齊缺少的欄位
    for need in ["PanTS ID","sex","age","tumor","ct phase","manufacturer","manufacturer model","study year"]:
        if need not in out.columns:
            out[need] = np.nan

    return out

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 去空白、標準化文字
    def T(x):
        s = str(x).strip() if pd.notna(x) else ""
        return "" if s in {"nan","NaN","None","NULL","null","undefined"} else s

    out["PanTS ID"] = out["PanTS ID"].astype(str).map(T)

    # tumor: 允許 1/0/True/False/Yes/No
    def norm_tumor(v):
        s = T(v).lower()
        if s in {"1","true","yes","y"}: return "1"
        if s in {"0","false","no","n"}:  return "0"
        return ""
    out["tumor"] = out["tumor"].map(norm_tumor)

    # sex: M/F/空
    def norm_sex(v):
        s = T(v).upper()
        if s in {"M","F"}: return s
        return ""
    out["sex"] = out["sex"].map(norm_sex)

    # age -> 整數或空
    def norm_age(v):
        try:
            n = int(float(v))
            return n if 0 <= n <= 120 else np.nan
        except:
            return np.nan
    out["age"] = out["age"].map(norm_age)

    # ct phase / manufacturer / model / year
    out["ct phase"] = out["ct phase"].map(T)
    out["manufacturer"] = out["manufacturer"].map(T)
    out["manufacturer model"] = out["manufacturer model"].map(T)

    def norm_year(v):
        try:
            y = int(float(v))
            return y if 1900 <= y <= 2100 else np.nan
        except:
            return np.nan
    out["study year"] = out["study year"].map(norm_year)

    return out

# 載入資料
try:
    _raw = pd.read_excel(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"讀取 {DATA_PATH} 失敗：{e}")

_df_full = _clean_df(_normalize_columns(_raw)).reset_index(drop=True)

# 供 facets 計算的「全部資料」年份範圍
_year_min = int(np.nanmin(_df_full["study year"])) if _df_full["study year"].notna().any() else None
_year_max = int(np.nanmax(_df_full["study year"])) if _df_full["study year"].notna().any() else None


# ── 共用：apply filters ────────────────────────────────────────────────
def _apply_filters(
    df: pd.DataFrame,
    caseid: str = "",
    tumor: str = "",
    sex: str = "",
    tumor_is_null: int = 0,
    sex_is_null: int = 0,
    age_from: Optional[int] = None,
    age_to: Optional[int] = None,
    age_is_null: int = 0,
    ct_phase: str = "",
    ct_is_null: int = 0,
    manufacturer: str = "",
    manufacturer_is_null: int = 0,
    year_from: str = "",
    year_to: str = "",
    year_is_null: int = 0,
) -> pd.DataFrame:

    out = df

    # caseid: 支援 12 或 PanTS_00000012（後 8 碼比對）
    if caseid:
        q = caseid.strip()
        if q.isdigit():
            target = f"PanTS_{int(q):08d}"
            mask = (out["PanTS ID"].str.lower() == target.lower()) | \
                   (out["PanTS ID"].str.extract(r"(\d{1,8})", expand=False).fillna("").map(lambda s: s.zfill(8)) == f"{int(q):08d}")
        else:
            mask = (out["PanTS ID"].str.lower() == q.lower())
        out = out[mask]

    # tumor
    if tumor_is_null:
        out = out[(out["tumor"] == "")]
    else:
        if tumor in {"0","1"}:
            out = out[out["tumor"] == tumor]

    # sex
    if sex_is_null:
        out = out[(out["sex"] == "")]
    else:
        if sex in {"M","F"}:
            out = out[out["sex"] == sex]

    # age
    if age_is_null:
        out = out[out["age"].isna()]
    else:
        a = 0 if age_from is None else int(age_from)
        b = 100 if age_to   is None else int(age_to)
        out = out[(out["age"].fillna(-1) >= a) & (out["age"].fillna(-1) <= b)]

    # ct phase
    if ct_is_null:
        out = out[(out["ct phase"] == "")]
    else:
        if ct_phase:
            out = out[out["ct phase"].str.lower() == ct_phase.strip().lower()]

    # manufacturer（多選，用逗號分隔），或 is_null
    if manufacturer_is_null:
        out = out[(out["manufacturer"] == "")]
    else:
        if manufacturer.strip():
            wants = {x.strip().lower() for x in manufacturer.split(",") if x.strip()}
            out = out[out["manufacturer"].str.lower().isin(wants)]

    # study year
    if year_is_null:
        out = out[out["study year"].isna()]
    else:
        if year_from:
            try:
                yf = int(year_from)
                out = out[out["study year"].fillna(-1) >= yf]
            except:
                pass
        if year_to:
            try:
                yt = int(year_to)
                out = out[out["study year"].fillna(999999) <= yt]
            except:
                pass

    return out


# ── Helpers：序列化為前端需要的欄位 ───────────────────────────────────────
def _to_item(row: pd.Series) -> Dict[str, Any]:
    return {
        "PanTS ID": row.get("PanTS ID", ""),
        "sex": row.get("sex", ""),
        "age": None if pd.isna(row.get("age")) else int(row.get("age")),
        "tumor": row.get("tumor", ""),
        "ct phase": row.get("ct phase", ""),
        "manufacturer": row.get("manufacturer", ""),
        "manufacturer model": row.get("manufacturer model", ""),
        "study year": None if pd.isna(row.get("study year")) else int(row.get("study year")),
        # 保留幾個替代欄位名稱，前端也兼容
        "case_id": row.get("PanTS ID", ""),
        "ct_phase": row.get("ct phase", ""),
        "study year".replace(" ", "_"): None if pd.isna(row.get("study year")) else int(row.get("study year")),
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"ok": True, "service": "pants-backend", "rows": int(_df_full.shape[0])}

@app.get("/api/search")
def api_search(
    caseid: str = "",
    tumor: str = "",
    sex: str = "",
    tumor_is_null: int = 0,
    sex_is_null: int = 0,
    age_from: Optional[int] = Query(None),
    age_to: Optional[int] = Query(None),
    age_is_null: int = 0,
    ct_phase: str = "",
    ct_is_null: int = 0,
    manufacturer: str = "",
    manufacturer_is_null: int = 0,
    year_from: str = "",
    year_to: str = "",
    year_is_null: int = 0,
    page: int = 1,
    per_page: str = "20",
):
    filtered = _apply_filters(
        _df_full, caseid, tumor, sex, tumor_is_null, sex_is_null,
        age_from, age_to, age_is_null, ct_phase, ct_is_null,
        manufacturer, manufacturer_is_null, year_from, year_to, year_is_null
    )

    # 分頁
    if per_page == "all":
        page_size = int(1e9)
    else:
        try:
            page_size = max(1, int(per_page))
        except:
            page_size = 20

    total = int(filtered.shape[0])
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    end = start + page_size

    items = [_to_item(r) for _, r in filtered.iloc[start:end].iterrows()]

    return {
        "total": total,
        "page": page,
        "pages": pages,
        "items": items,
    }


@app.get("/api/facets")
def api_facets(
    fields: str,
    top_k: int = 10,
    guarantee: int = 0,
    # 同樣接受所有 filter 參數，讓 facets 可以依據目前條件顯示
    caseid: str = "",
    tumor: str = "",
    sex: str = "",
    tumor_is_null: int = 0,
    sex_is_null: int = 0,
    age_from: Optional[int] = Query(None),
    age_to: Optional[int] = Query(None),
    age_is_null: int = 0,
    ct_phase: str = "",
    ct_is_null: int = 0,
    manufacturer: str = "",
    manufacturer_is_null: int = 0,
    year_from: str = "",
    year_to: str = "",
    year_is_null: int = 0,
):
    base = _apply_filters(
        _df_full, caseid, tumor, sex, tumor_is_null, sex_is_null,
        age_from, age_to, age_is_null, ct_phase, ct_is_null,
        manufacturer, manufacturer_is_null, year_from, year_to, year_is_null
    )

    facets: Dict[str, List[Dict[str, Any]]] = {}
    req = [f.strip() for f in (fields or "").split(",") if f.strip()]

    for f in req:
        if f == "ct_phase":
            col = "ct phase"
        elif f == "manufacturer":
            col = "manufacturer"
        elif f == "year":
            # year_range
            pass
        else:
            col = f

        if f == "year":
            # 給前端 placeholder 用的年份區間
            facets["year"] = []
        else:
            if col in base.columns:
                vc = (
                    base[col]
                    .fillna("")
                    .replace({"nan": ""})
                    .value_counts(dropna=False)
                    .reset_index()
                )
                vc.columns = ["value", "count"]
                # 排除空字串的 facet 值
                vc = vc[vc["value"].astype(str).str.strip() != ""]
                top = vc.head(top_k)
                facets_key = f if f != "ct phase" else "ct_phase"
                facets[facets_key] = [
                    {"value": str(r["value"]), "count": int(r["count"])}
                    for _, r in top.iterrows()
                ]
            else:
                facets_key = f if f != "ct phase" else "ct_phase"
                facets[facets_key] = []

    # year_range：若目前篩選無資料，仍能回傳全域範圍（前端只拿來當 placeholder）
    if "year" in req:
        yr_col = base["study year"].dropna()
        if yr_col.empty or guarantee:
            y_min = _year_min
            y_max = _year_max
        else:
            y_min = int(yr_col.min())
            y_max = int(yr_col.max())
        facets["year_range"] = {"min": y_min, "max": y_max}

    return {"facets": facets, "year_range": facets.get("year_range", None)}


@app.get("/api/random")
def api_random(
    scope: str = "all",   # "filtered" or "all"
    n: int = 3,
    # 同樣接受 filter 參數（當 scope=filtered 時生效）
    caseid: str = "",
    tumor: str = "",
    sex: str = "",
    tumor_is_null: int = 0,
    sex_is_null: int = 0,
    age_from: Optional[int] = Query(None),
    age_to: Optional[int] = Query(None),
    age_is_null: int = 0,
    ct_phase: str = "",
    ct_is_null: int = 0,
    manufacturer: str = "",
    manufacturer_is_null: int = 0,
    year_from: str = "",
    year_to: str = "",
    year_is_null: int = 0,
):
    if scope == "filtered":
        base = _apply_filters(
            _df_full, caseid, tumor, sex, tumor_is_null, sex_is_null,
            age_from, age_to, age_is_null, ct_phase, ct_is_null,
            manufacturer, manufacturer_is_null, year_from, year_to, year_is_null
        )
    else:
        base = _df_full

    n = max(0, min(int(n), len(base)))
    if n == 0:
        items = []
    else:
        idxs = random.sample(range(len(base)), n)
        items = [_to_item(base.iloc[i]) for i in idxs]

    return {"items": items}


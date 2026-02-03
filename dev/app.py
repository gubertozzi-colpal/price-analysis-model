# app.py
# Streamlit app: Amazon Price x BSR Analytics (multi-page)
# ✅ Enterprise metadata improvement:
# - Downloadable metadata template
# - Flexible column mapping
# - Metadata validation + coverage diagnostics
# - Enrich ALL analytics with metadata fields and filters
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import os
import re
import glob
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# Constants / Defaults
# ----------------------------
DEFAULT_DATA_GLOB = "./dev/data/*-bsr-1y*.csv"
TZ = "America/Sao_Paulo"

DEFAULT_EVENTS = [
    {"name": "Prime Day 2025", "start": "2025-07-15", "end": "2025-07-16"},
    {"name": "Black Friday 2025", "start": "2025-11-28", "end": "2025-11-28"},
]

columns_map = {'ASIN': 'asin', 'Descrição': 'sku_name'}

# Recommended metadata schema (template)
TEMPLATE_COLS = [
    "asin",          # REQUIRED
    "sku_name",      # friendly label
    "brand",
    "subbrand",
    "segment",       # e.g. Premium / Core / Entry
    "pack_type",     # Single / Pack / Multipack / Kit
    "pack_qty",      # e.g. 2, 3
    "size_ml",       # or grams
    "size_g",
    "is_own",        # 1/0 or true/false (your SKU vs competitor)
    "ean",
    "notes",
]

# Common alternative column names you might receive
CANONICAL_MAP_CANDIDATES = {
    "asin": ["asin", "ASIN", "Asin", "asin_id", "asin code", "asin_code"],
    "sku_name": ["sku_name", "sku", "name", "product_name", "title", "desc", "descricao", "description"],
    "brand": ["brand", "marca"],
    "subbrand": ["subbrand", "sub_marca", "submarca", "sub-brand"],
    "segment": ["segment", "segmento", "tier", "faixa"],
    "pack_type": ["pack_type", "pack", "promo", "tipo_pack", "tipo", "bundle_type"],
    "pack_qty": ["pack_qty", "qtd_pack", "qty", "quantidade", "pack_quantity"],
    "size_ml": ["size_ml", "ml", "volume", "tamanho_ml"],
    "size_g": ["size_g", "g", "gramas", "peso", "tamanho_g"],
    "is_own": ["is_own", "own", "meu", "is_my", "is_mine", "my_sku", "owner"],
    "ean": ["ean", "EAN", "gtin", "GTIN", "barcode", "codigo_barras"],
    "notes": ["notes", "obs", "observacao", "observações", "comment", "comentario"],
}

# ----------------------------
# Style CSS
# ----------------------------

st.markdown("""
    <style>
        /* Ajusta o padding inferior do container principal */
        .main .block-container {
            padding-bottom: 5rem; 
        }
        
        /* Opcional: Se o rodapé estiver atrapalhando, você pode escondê-lo */
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper functions (core)
# ----------------------------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma cópia do DataFrame e remove espaços em branco extras 
    (trim) dos nomes de todas as colunas.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_asin_from_filename(path: str) -> str:
    """
    Extrai o ASIN (Amazon Standard Identification Number) a partir do nome do arquivo.
    Tenta primeiro um padrão específico ([ASIN]-bsr-1y) e, se não encontrar,
    busca por qualquer sequência de 10 caracteres alfanuméricos.
    """
    base = os.path.basename(path)
    # Tenta casamento com o sufixo específico de BSR
    m = re.match(r"([A-Z0-9]{10})-bsr-1y", base)
    if m:
        return m.group(1)
    # Fallback para qualquer ASIN de 10 caracteres no nome
    m2 = re.search(r"([A-Z0-9]{10})", base)
    return m2.group(1) if m2 else base


def parse_time_col(series: pd.Series, dayfirst=True) -> pd.Series:
    """
    Converte uma coluna para datetime. Tenta converter considerando fuso horário (UTC)
    e ajustando para o fuso local (variável TZ). Se falhar (ex: formatos mistos),
    tenta uma conversão simples sem fuso.
    """
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=True)
    try:
        dt = dt.dt.tz_convert(TZ).dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return dt


@st.cache_data(show_spinner=False)
def load_all(data_glob: str, dayfirst=True) -> pd.DataFrame:
    """
    Carrega múltiplos arquivos CSV baseados em um padrão de busca (glob).
    - Extrai o ASIN do nome do arquivo.
    - Padroniza nomes de colunas essenciais (Time, Sales Rank, etc).
    - Converte tipos de dados e remove linhas sem data válida.
    - Retorna um DataFrame consolidado e ordenado.
    """
    paths = sorted(glob.glob(data_glob))
    if not paths:
        return pd.DataFrame()

    all_dfs = []
    for p in paths:
        df = _clean_cols(pd.read_csv(p))
        asin = infer_asin_from_filename(p)
        df["asin"] = asin

        # Mapeamento para nomes internos padronizados
        col_map = {"Time": "time_raw", "Sales Rank": "bsr", "New Price": "price_new", "List Price": "price_list"}
        for src, dst in col_map.items():
            if src in df.columns:
                df = df.rename(columns={src: dst})
            elif dst not in df.columns:
                df[dst] = np.nan

        df["date"] = parse_time_col(df["time_raw"], dayfirst=dayfirst)
        for c in ["bsr", "price_new", "price_list"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["date"])
        all_dfs.append(df[["asin", "date", "bsr", "price_new", "price_list"]])

    raw = pd.concat(all_dfs, ignore_index=True)
    return raw.sort_values(["asin", "date"])


@st.cache_data(show_spinner=False)
def make_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma dados intra-diários em registros diários.
    - Pega o último valor registrado de preço e BSR no dia.
    - Define o 'price_effective' priorizando o preço novo sobre o preço de lista.
    - Cria colunas auxiliares de mês para agrupamentos temporais.
    """
    if raw.empty:
        return raw

    df = raw.copy()
    df["day"] = df["date"].dt.floor("D")
    daily = (
        df.sort_values(["asin", "date"])
        .groupby(["asin", "day"], as_index=False)
        .agg(
            price_new=("price_new", "last"),
            price_list=("price_list", "last"),
            price_mean=("price_new", "mean"),
            price_array=("price_new", lambda x: x.dropna().tolist()),
            bsr=("bsr", "last"),
            obs=("date", "count"), # Conta quantas observações originais existiam
        )
    )
    daily["price_effective"] = daily["price_new"].fillna(daily["price_list"])
    daily = daily.dropna(subset=["price_effective", "bsr"]).copy()
    daily["month"] = daily["day"].dt.to_period("M").astype(str)
    daily["month_dt"] = pd.to_datetime(daily["month"] + "-01")
    return daily


def add_base_and_promo(daily: pd.DataFrame, roll_days=30, q=0.8, promo_threshold=0.05) -> pd.DataFrame:
    """
    Identifica promoções comparando o preço atual com um 'preço base'.
    - O preço base é calculado usando um quantil móvel (rolling quantile) de 30 dias.
    - Se o desconto em relação à base for >= threshold (ex: 5%), é marcado como promo.
    """
    df = daily.sort_values(["asin", "day"]).copy()

    # Gu: Entender como calcula por produto o preço base.
    def _base(g):
        s = g["price_effective"]
        # 1. Janela Móvel (Rolling)
        base_roll = s.rolling(roll_days, min_periods=max(10, roll_days // 3)).quantile(q)
        # 2. Janela Expansiva (Expanding) - O "Fallback"
        base_expand = s.expanding(min_periods=10).quantile(q)
        # 3. Combinação
        g["price_base"] = base_roll.fillna(base_expand)
        return g

    df = df.groupby("asin", group_keys=False).apply(_base)
    df["discount_pct"] = (df["price_base"] - df["price_effective"]) / df["price_base"]
    df["is_promo"] = df["discount_pct"] >= promo_threshold
    df["rebate_value"] = df["price_base"] - df["price_effective"]
    df["price_promo"] = np.where(df["is_promo"], df["price_effective"], np.nan)
    return df


def method_corr_pivot(df: pd.DataFrame, value_col: str, method: str, id_prod: str) -> pd.DataFrame:
    """
    Calcula a correlação de <method> entre diferentes ASINs para uma métrica 
    específica (ex: preço), pivotando a tabela para ter ASINs como colunas.
    """

    columns_map = {'ASIN': 'asin', 'Descrição': 'sku_name'}
    pivot = df.pivot(index="day", columns=columns_map[id_prod], values=value_col)
    corr = pivot.corr(method=method.lower(), min_periods=30)
    corr.index.name = id_prod
    corr.columns.name = id_prod
    return corr


def scatter_corr(df: pd.DataFrame, value_col: str, id_prod: str) -> pd.DataFrame:
    """
    Retorna um DataFrame com as datas e os valores de dois produtos específicos,
    filtrando apenas os dias em que AMBOS possuem dados (intersecção).
    """    
    # 1. Pivotar: O índice vira 'day' e as colunas viram os produtos
    pivot = df.pivot(index="day", columns=columns_map[id_prod], values=value_col)

    # 2. Selecionar e Resetar Índice
    # Selecionamos as duas colunas e usamos reset_index para que 'day' volte a ser uma coluna
    df_scatter = pivot.reset_index()

    return df_scatter


def scatter_cross_corr(df: pd.DataFrame, prod1: str, prod2: str, id_prod: str) -> pd.DataFrame:
    """
    Retorna um DataFrame com 3 colunas:
    1. 'day'
    2. Coluna com nome do prod1 -> contendo PREÇO (price_effective)
    3. Coluna com nome do prod2 -> contendo BSR (bsr)
    
    Apenas dias onde ambos têm dados (join='inner').
    """
    col_id = columns_map[id_prod] # Define se filtramos por 'asin' ou 'sku_name'

    # 1. Extrair Série do Produto 1 (PREÇO)
    # Filtra linhas do prod1 -> Define dia como índice -> Pega só o preço -> Renomeia a série para o nome do produto
    s1 = (
        df[df[col_id] == prod1]
        .set_index("day")["price_effective"]
        .rename(prod1)
    )

    # 2. Extrair Série do Produto 2 (BSR)
    # Filtra linhas do prod2 -> Define dia como índice -> Pega só o BSR -> Renomeia
    s2 = (
        df[df[col_id] == prod2]
        .set_index("day")["bsr"]
        .rename(prod2)
    )

    # 3. Juntar as duas séries (Alinhamento temporal)
    # axis=1: Coloca uma do lado da outra
    # join="inner": Mantém apenas os dias que existem nas DUAS séries (intersecção)
    combined = pd.concat([s1, s2], axis=1, join="inner")

    # 4. Resetar índice para ter 'day' como coluna
    return combined.reset_index()


def price_vs_bsr_corr(df: pd.DataFrame, method:str, id_prod: str) -> pd.DataFrame:
    """
    Calcula a correlação entre Preço e BSR (Sales Rank) para cada produto.
    Ajuda a entender se a queda de preço melhora o ranking (correlação positiva).
    """
    out = []
    columns_map = {'ASIN': 'asin', 'Descrição': 'sku_name'}
    for asin, g in df.groupby(columns_map[id_prod]):
        n = g[["price_effective", "bsr"]].dropna().shape[0]
        # Exige ao menos 30 dias de dados para ser estatisticamente relevante
        r = g[["price_effective", "bsr"]].corr(method=method.lower()).iloc[0, 1] if n >= 30 else np.nan
        out.append({"asin": asin, "spearman_price_bsr": r, "n_obs": n})
    return pd.DataFrame(out).sort_values("spearman_price_bsr", ascending=False)


def cross_price_bsr_matrix(df: pd.DataFrame, method="spearman", id_prod='asin', min_periods=30) -> pd.DataFrame:
    """
    Gera uma matriz onde:
    - Linhas (Index) = ASIN cujo PREÇO mudou.
    - Colunas = ASIN cujo BSR (Rank) reagiu.
    
    Exemplo de leitura: Valor na linha A, coluna B diz a correlação entre
    o Preço de A e as Vendas (BSR) de B.
    """
    # 1. Pivotar Preços e BSRs separadamente
    # Index = Dia, Colunas = ASINs
    columns_map = {'ASIN': 'asin', 'Descrição': 'sku_name'}
    prices = df.pivot(index="day", columns=columns_map[id_prod], values="price_effective").sort_index(axis=1)
    ranks = df.pivot(index="day", columns=columns_map[id_prod], values="bsr").sort_index(axis=1)

    # Garante que temos as mesmas colunas em ambos
    common_asins = prices.columns.intersection(ranks.columns)
    prices = prices[common_asins]
    ranks = ranks[common_asins]

    # 2. Calcular correlações cruzadas
    # Vamos usar um loop eficiente com corrwith para comparar 
    # "Preço de Um" contra "BSR de Todos"
    matrix_data = {}
    
    for asin_driver in common_asins:
        # Série de preço do "Driver"
        p_series = prices[asin_driver]
        
        # Correlaciona esse preço contra TODOS os BSRs de uma vez
        # O resultado é uma Series com index = asin (responder)
        corrs = ranks.corrwith(p_series, method=method.lower(), drop=True)
        
        # Filtra quem não tem dados suficientes (min_periods não funciona direto no corrwith do jeito que queremos aqui as vezes,
        # mas o pandas lida com NaNs. Se quiser ser estrito, precisaria validar intersecção de índices).
        matrix_data[asin_driver] = corrs

    # Monta o DataFrame final (Transposta para ficar Linha=Preço, Coluna=BSR)
    cross_matrix = pd.DataFrame(matrix_data).T
    cross_matrix.index.name = "asin_price_driver"
    cross_matrix.columns.name = "asin_bsr_responder"
    
    return cross_matrix


def price_vs_bsr_corr_kmean(df: pd.DataFrame, method:str) -> pd.DataFrame:
    """
    Calcula a correlação entre Preço e BSR (Sales Rank) para cada produto.
    Ajuda a entender se a queda de preço melhora o ranking (correlação positiva).
    """
    out = []
    for asin, g in df.groupby('asin'):
        n = g[["price_effective", "bsr"]].dropna().shape[0]
        # Exige ao menos 30 dias de dados para ser estatisticamente relevante
        r = g[["price_effective", "bsr"]].corr(method=method.lower()).iloc[0, 1] if n >= 30 else np.nan
        out.append({"asin": asin, "spearman_price_bsr": r, "n_obs": n})
    return pd.DataFrame(out).sort_values("spearman_price_bsr", ascending=False)


def elasticity_proxy(df: pd.DataFrame, asin: str, bucket_round=2, min_n=6) -> pd.DataFrame:
    """
    Tenta estimar a elasticidade preço-demanda (usando BSR como proxy de vendas).
    Calcula a variação do log do BSR em relação à variação do preço.
    """
    g = df[df["asin"] == asin].copy()
    if g.empty:
        return pd.DataFrame()

    b = best_price_bucket(g, min_n=min_n, bucket_round=bucket_round)
    if b.empty:
        return b

    b = b.copy().sort_values("price_bucket")
    b["log_bsr_med"] = np.log(b["bsr_median"].clip(lower=1))
    b["d_price"] = b["price_bucket"].diff()
    b["d_log_bsr"] = b["log_bsr_med"].diff()
    b["elasticity_proxy"] = (b["d_log_bsr"] / b["d_price"]).replace([np.inf, -np.inf], np.nan)
    return b


def sku_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um resumo estatístico consolidado por SKU (ASIN).
    Inclui médias de preço, share de promoção e BSR mediano.
    """
    def _avg_discount_promo(x):
        # Média de desconto apenas quando o item estava em promoção
        xp = x[df.loc[x.index, "is_promo"]]
        return xp.mean() if len(xp) else np.nan

    return (
        df.groupby("asin")
        .agg(
            days=("day", "nunique"),
            avg_price=("price_effective", "mean"),
            med_price=("price_effective", "median"),
            avg_base=("price_base", "mean"),
            promo_share=("is_promo", "mean"),
            avg_discount_when_promo=("discount_pct", _avg_discount_promo),
            bsr_med=("bsr", "median"),
            bsr_mean=("bsr", "mean"),
        )
        .reset_index()
    )


def best_price_bucket(df: pd.DataFrame, min_n=6, bucket_round=2) -> pd.DataFrame:
    """
    Agrupa preços em 'baldes' (arredondados) para identificar em qual 
    faixa de preço o BSR tende a ser melhor (menor).
    """
    g = df.copy()
    g["price_bucket"] = g["price_effective"].round(bucket_round)
    agg = (
        g.groupby("price_bucket", as_index=False)
        .agg(
            n=("bsr", "size"),
            bsr_median=("bsr", "median"),
            bsr_mean=("bsr", "mean"),
            promo_share=("is_promo", "mean"),
            discount_median=("discount_pct", "median"),
        )
    )
    return agg[agg["n"] >= min_n].sort_values("bsr_median")


def build_best_prices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asin, g in df.groupby("asin"):
        best = best_price_bucket(g, min_n=6)
        if len(best):
            r = best.iloc[0].to_dict()
            r["asin"] = asin
            rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values("bsr_median") if not out.empty else out


def price_index(df: pd.DataFrame, leader_asin: str) -> pd.DataFrame:
    pivot = df.pivot(index="day", columns="asin", values="price_effective")
    if leader_asin not in pivot.columns:
        return pd.DataFrame()
    leader = pivot[leader_asin]
    idx = pivot.divide(leader, axis=0)
    idx = idx.reset_index().melt(id_vars="day", var_name="asin", value_name="price_index")
    return idx.dropna(subset=["price_index"])


def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    def _avg_disc_promo(x):
        xp = x[df.loc[x.index, "is_promo"]]
        return xp.mean() if len(xp) else np.nan

    return (
        df.groupby(["asin", "month", "month_dt"], as_index=False)
        .agg(
            price=("price_effective", "mean"),
            base=("price_base", "mean"),
            bsr_med=("bsr", "median"),
            bsr_mean=("bsr", "mean"),
            promo_share=("is_promo", "mean"),
            discount=("discount_pct", _avg_disc_promo),
        )
    )

def competitive_map(df: pd.DataFrame, k=4, random_state=42) -> pd.DataFrame:
    """
    Aplica KMeans para agrupar SKUs com comportamentos similares baseados em 
    preço, share de promoção e sensibilidade ao BSR.
    """
    summ = sku_summary(df)
    sens = price_vs_bsr_corr_kmean(df, ctl_corr)[["asin", "spearman_price_bsr"]]
    feat = summ.merge(sens, on="asin", how="left").copy()

    # Imputação de nulos pela mediana para o modelo
    for c in ["avg_discount_when_promo", "spearman_price_bsr"]:
        feat[c] = feat[c].fillna(feat[c].median())

    X = feat[["avg_price", "promo_share", "avg_discount_when_promo", "bsr_med", "spearman_price_bsr"]].values
    Xs = StandardScaler().fit_transform(X) # Padronização de escala
    k = max(2, min(k, len(feat)))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    feat["cluster"] = km.fit_predict(Xs)
    return feat

def event_summary(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, baseline_days=14, pre=7, post=7) -> pd.DataFrame:
    win_start, win_end = start - pd.Timedelta(days=pre), end + pd.Timedelta(days=post)
    base_start, base_end = start - pd.Timedelta(days=baseline_days), start - pd.Timedelta(days=1)

    dwin = df[(df["day"] >= win_start) & (df["day"] <= win_end)]
    dbase = df[(df["day"] >= base_start) & (df["day"] <= base_end)]

    rows = []
    for asin, g in dwin.groupby("asin"):
        b = dbase[dbase["asin"] == asin]
        rows.append(
            dict(
                asin=asin,
                window_days=g["day"].nunique(),
                price_avg_window=g["price_effective"].mean(),
                promo_share_window=g["is_promo"].mean(),
                discount_avg_window=g.loc[g["is_promo"], "discount_pct"].mean(),
                bsr_med_window=g["bsr"].median(),
                bsr_mean_window=g["bsr"].mean(),
                bsr_med_baseline=b["bsr"].median() if len(b) else np.nan,
                price_avg_baseline=b["price_effective"].mean() if len(b) else np.nan,
            )
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bsr_med_delta"] = out["bsr_med_window"] - out["bsr_med_baseline"]
    out["price_delta"] = out["price_avg_window"] - out["price_avg_baseline"]
    return out.sort_values("bsr_med_delta")


# ----------------------------
# METADATA: Enterprise Improvement
# ----------------------------
def make_metadata_template(asins: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"asin": asins})
    for c in TEMPLATE_COLS:
        if c not in df.columns:
            df[c] = ""
    # Put recommended columns in order
    return df[TEMPLATE_COLS]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    df.to_csv(bio, index=False, encoding="utf-8-sig")
    return bio.getvalue()


def normalize_bool(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "sim", "s"}
    false_set = {"0", "false", "f", "no", "n", "nao", "não"}
    return x.apply(lambda v: True if v in true_set else (False if v in false_set else np.nan))


def auto_suggest_mapping(meta_cols: List[str]) -> Dict[str, Optional[str]]:
    """Best-effort mapping from meta columns to canonical template fields."""
    lower = {c.lower(): c for c in meta_cols}
    mapping = {}
    for canon, candidates in CANONICAL_MAP_CANDIDATES.items():
        chosen = None
        for cand in candidates:
            if cand.lower() in lower:
                chosen = lower[cand.lower()]
                break
        mapping[canon] = chosen
    return mapping


def apply_column_mapping(meta: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename mapped columns to canonical names. Keep other columns as-is."""
    meta = meta.copy()
    rename = {}
    for canon, src in mapping.items():
        if src and src in meta.columns and canon != src:
            rename[src] = canon
    meta = meta.rename(columns=rename)
    return meta


def validate_metadata(meta: pd.DataFrame, asins_in_data: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Valida e limpa o arquivo de metadados enviado pelo usuário.
    - Garante a presença da coluna 'asin'.
    - Remove duplicatas.
    - Normaliza colunas booleanas.
    - Gera diagnósticos de cobertura (quais ASINs do dataset faltam no metadata).
    """
    diag = {"errors": [], "warnings": [], "coverage": {}}

    if meta is None or meta.empty:
        diag["warnings"].append("Nenhum metadata fornecido. O app vai rodar com nomes genéricos.")
        return pd.DataFrame(), diag

    meta = _clean_cols(meta)

    if "asin" not in [c.lower() for c in meta.columns]:
        diag["errors"].append("Coluna 'asin' não encontrada após mapeamento.")
        return pd.DataFrame(), diag

    # Ensure canonical asin column name
    asin_col = None
    for c in meta.columns:
        if c.lower() == "asin":
            asin_col = c
            break
    if asin_col != "asin":
        meta = meta.rename(columns={asin_col: "asin"})

    meta["asin"] = meta["asin"].astype(str).str.strip()

    # Duplicate ASINs
    dup = meta["asin"].duplicated(keep=False)
    if dup.any():
        dups = meta.loc[dup, "asin"].value_counts().head(10).to_dict()
        diag["warnings"].append(f"ASINs duplicados no metadata (mostrando até 10): {dups}")
        # Keep last by default
        meta = meta.drop_duplicates("asin", keep="last")

    # Normalize is_own
    if "is_own" in meta.columns:
        meta["is_own"] = normalize_bool(meta["is_own"])

    # Guarantee minimal friendly fields
    if "sku_name" not in meta.columns:
        meta["sku_name"] = meta["asin"]
        diag["warnings"].append("Coluna 'sku_name' ausente. Usei o ASIN como nome do SKU.")
    if "brand" not in meta.columns:
        meta["brand"] = "NA"
    if "segment" not in meta.columns:
        meta["segment"] = "NA"

    # Coverage checks
    asins_set = set(asins_in_data)
    meta_set = set(meta["asin"].tolist())
    missing = sorted(list(asins_set - meta_set))
    extra = sorted(list(meta_set - asins_set))

    diag["coverage"] = {
        "asins_in_data": len(asins_set),
        "asins_in_metadata": len(meta_set),
        "mapped": len(asins_set & meta_set),
        "missing_in_metadata": len(missing),
        "extra_not_in_data": len(extra),
        "missing_list": missing[:50],  # cap
        "extra_list": extra[:50],
    }

    if missing:
        diag["warnings"].append(f"Metadata não cobre {len(missing)} ASINs do dataset (mostrando até 50).")
    if extra:
        diag["warnings"].append(f"Metadata tem {len(extra)} ASINs que não estão no dataset (mostrando até 50).")

    return meta, diag


def apply_metadata(daily: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if meta is None or meta.empty:
        out = daily.copy()
        out["sku_name"] = out["asin"]
        out["brand"] = "NA"
        out["segment"] = "NA"
        out["is_own"] = np.nan
        return out

    out = daily.merge(meta, on="asin", how="left")
    out["sku_name"] = out.get("sku_name", out["asin"]).fillna(out["asin"])
    out["brand"] = out.get("brand", "NA").fillna("NA")
    out["segment"] = out.get("segment", "NA").fillna("NA")
    return out


def meta_filters_ui(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("🧩 Filtros (metadata)")

        brands = sorted(df["brand"].dropna().astype(str).unique().tolist())
        sel_brands = st.multiselect("Marca", options=brands, default=brands)

        segments = sorted(df["segment"].dropna().astype(str).unique().tolist())
        sel_segments = st.multiselect("Segmento", options=segments, default=segments)

        has_is_own = "is_own" in df.columns and df["is_own"].notna().any()
        mode = "Todos"
        if has_is_own:
            mode = st.selectbox("Tipo", ["Todos", "Só meus (is_own=1)", "Só concorrentes (is_own=0)"], index=0)

        q = st.text_input("Buscar SKU (nome contém)", value="").strip().lower()

    f = df.copy()
    f = f[f["brand"].astype(str).isin(sel_brands)]
    f = f[f["segment"].astype(str).isin(sel_segments)]

    if has_is_own and mode != "Todos":
        f = f[f["is_own"] == ("meus" in mode.lower())]

    if q:
        f = f[f["sku_name"].astype(str).str.lower().str.contains(q, na=False)]

    return f

def filter_period(df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """Filtra o DataFrame para incluir apenas datas entre min_date e max_date (inclusivo)."""
    return df[(df["day"] >= min_date) & (df["day"] <= max_date)]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Amazon Price x BSR Analytics", layout="wide")
st.title("📊 Amazon – Análise Profissional de Preço x BSR (multi-SKU)")

with st.sidebar:
    st.header("⚙️ Configurações")

    data_glob = st.text_input("DATA_GLOB (caminho/curinga dos CSVs)", value=DEFAULT_DATA_GLOB)
    dayfirst = st.toggle("Datas no formato dia/mês (dayfirst)", value=True)

    st.subheader("📅 Controles de Data")
    min_date = st.date_input("Data inicial mínima", value=pd.to_datetime("2025-01-01"))
    max_date = st.date_input("Data final máxima", value=pd.to_datetime("today"))

    st.subheader("📈 Controles de Correlação")
    ctl_corr = st.selectbox("Tipo de Correlação", ["Kendall", "Pearson", "Spearman"], index=2)
    ctl_prod = st.selectbox("Identificador", ["ASIN", "Descrição"], index=1)

    st.subheader("Separação Base vs Promo")
    roll_days = st.slider("Janela base (dias)", 14, 60, 30, 1)
    base_q = st.slider("Quantil para base (p80 recomendado)", 0.6, 0.95, 0.8, 0.05)
    promo_threshold = st.slider("Threshold promo (% abaixo do base)", 0.02, 0.25, 0.05, 0.01)

    st.subheader("Eventos")
    baseline_days = st.slider("Baseline antes do evento (dias)", 7, 28, 14, 1)
    pre = st.slider("Janela pré-evento (dias)", 0, 14, 7, 1)
    post = st.slider("Janela pós-evento (dias)", 0, 14, 7, 1)

    events = st.session_state.get("events", DEFAULT_EVENTS)
    st.caption("Edite/adicione eventos (YYYY-MM-DD).")
    if st.button("➕ Adicionar evento"):
        events = events + [{"name": "Novo Evento", "start": "2025-01-01", "end": "2025-01-01"}]
    new_events = []
    for i, ev in enumerate(events):
        st.markdown(f"**Evento {i+1}**")
        name = st.text_input(f"Nome {i+1}", value=ev["name"], key=f"ev_name_{i}")
        start = st.text_input(f"Início {i+1}", value=ev["start"], key=f"ev_start_{i}")
        end = st.text_input(f"Fim {i+1}", value=ev["end"], key=f"ev_end_{i}")
        new_events.append({"name": name, "start": start, "end": end})
        st.divider()
    st.session_state["events"] = new_events

    st.subheader("Mapa competitivo")
    k_clusters = st.slider("Número de clusters", 2, 8, 4, 1)

    st.subheader("Frequência")
    freq = st.selectbox("Granularidade principal", ["Diário", "Mensal"], index=1)

    st.subheader("📎 Metadata (enterprise)")
    st.caption("Upload de CSV + validação + template para baixar.")
    meta_file = st.file_uploader("Upload metadata CSV", type=["csv"])

# Load core data
raw = load_all(data_glob=data_glob, dayfirst=dayfirst)
if raw.empty:
    st.error("Não encontrei arquivos. Ajuste o DATA_GLOB (ex.: ./data/*-bsr-1y*.csv).")
    st.stop()

daily = add_base_and_promo(make_daily(raw), roll_days=roll_days, q=base_q, promo_threshold=promo_threshold)
asins_in_data = sorted(daily["asin"].unique().tolist())

# Metadata section (enterprise)
meta = pd.DataFrame()
diag = {"errors": [], "warnings": [], "coverage": {}}

with st.expander("📎 Metadata – Template, Mapeamento e Validação", expanded=True):
    st.markdown(
        """
Aqui você:
- Baixa um **template** pronto com todos os ASINs do dataset.
- Faz **upload** do metadata (pode vir com nomes de colunas diferentes).
- Ajusta um **mapeamento** (se necessário).
- Vê **validação e cobertura** (quantos SKUs estão mapeados).
        """
    )

    # Template download
    tpl = make_metadata_template(asins_in_data)
    st.download_button(
        "⬇️ Baixar template de metadata (com ASINs do dataset)",
        data=to_csv_bytes(tpl),
        file_name="metadata_template.csv",
        mime="text/csv",
    )

    if meta_file is not None:
        # Read metadata robustly
        file_bytes = meta_file.getvalue()
        # Try encodings
        loaded = None
        for enc in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                loaded = pd.read_csv(BytesIO(file_bytes), encoding=enc, sep=';')
                break
            except Exception:
                continue
        if loaded is None:
            st.error("Não consegui ler seu metadata CSV. Tente salvar como UTF-8 ou UTF-8-SIG.")
        else:
            loaded = _clean_cols(loaded)

            st.markdown("**1) Mapeamento de colunas (flexível)**")
            suggested = auto_suggest_mapping(list(loaded.columns))

            cols = ["(não mapear)"] + list(loaded.columns)
            mapping = {}
            c1, c2, c3 = st.columns(3)
            # show key fields first
            key_fields = ["asin", "sku_name", "brand", "segment", "is_own", "pack_type", "pack_qty", "size_ml", "size_g", "subbrand", "ean"]
            for i, canon in enumerate(key_fields):
                default = suggested.get(canon)
                idx = cols.index(default) if default in cols else 0
                target_col = (c1 if i % 3 == 0 else (c2 if i % 3 == 1 else c3)).selectbox(
                    f"{canon}  ←", options=cols, index=idx, key=f"map_{canon}"
                )
                mapping[canon] = None if target_col == "(não mapear)" else target_col

            meta = apply_column_mapping(loaded, mapping)
            meta, diag = validate_metadata(meta, asins_in_data)

            # Show diagnostics
            if diag["errors"]:
                st.error("Erros no metadata:\n- " + "\n- ".join(diag["errors"]))
            if diag["warnings"]:
                st.warning("Avisos:\n- " + "\n- ".join(diag["warnings"]))

            cov = diag.get("coverage", {})
            if cov:
                st.info(
                    f"Cobertura: {cov.get('mapped',0)}/{cov.get('asins_in_data',0)} ASINs mapeados. "
                    f"Faltando no metadata: {cov.get('missing_in_metadata',0)} | "
                    f"Extras fora do dataset: {cov.get('extra_not_in_data',0)}"
                )

                if cov.get("missing_list"):
                    st.caption("ASINs do dataset sem metadata (até 50): " + ", ".join(cov["missing_list"]))
                if cov.get("extra_list"):
                    st.caption("ASINs no metadata que não estão no dataset (até 50): " + ", ".join(cov["extra_list"]))

            st.markdown("**2) Preview do metadata (após mapeamento/limpeza)**")
            st.dataframe(meta.head(50), width='stretch', hide_index=True)
    else:
        st.info("Opcional: faça upload do metadata para habilitar filtros por marca/segmento e insights contextualizados.")

# Apply metadata to daily and filter
daily = filter_period(daily, pd.to_datetime(min_date), pd.to_datetime(max_date))
daily = apply_metadata(daily, meta)
daily_f = meta_filters_ui(daily)

# Build artifacts on filtered data
summ = sku_summary(daily_f)
sens = price_vs_bsr_corr(daily_f, ctl_corr, ctl_prod)
summ2 = summ.merge(sens[["asin", "spearman_price_bsr"]], on="asin", how="left")
best_prices = build_best_prices(daily_f)
monthly = monthly_agg(daily_f)
price_corr = method_corr_pivot(daily_f, "price_effective",ctl_corr, ctl_prod)
bsr_corr = method_corr_pivot(daily_f, "bsr",ctl_corr, ctl_prod)
cross_corr = cross_price_bsr_matrix(daily_f, ctl_corr, ctl_prod)
scatter_price = scatter_corr(daily_f, value_col="price_effective", id_prod=ctl_prod)
scatter_bsr = scatter_corr(daily_f, value_col="bsr", id_prod=ctl_prod)
asins = sorted(daily_f["asin"].unique().tolist())

# Tabs
pages = [
    "Visão Geral",
    "Evolução (Preço/BSR)",
    "Base vs Promo + Profundidade",
    "Correlação",
    "Índice de Preço (Price Index)",
    "Preço mágico + Elasticidade (proxy)",
    "Mapa Competitivo (clusters)",
    "Playbook de Eventos (Prime/Black/etc.)",
    "Recomendações (Tático & Estratégico)",
    "testes",
]
tabs = st.tabs(pages)

# Utility to attach metadata into summary-like tables
def enrich_with_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [c for c in ["sku_name", "brand", "subbrand", "segment", "pack_type", "pack_qty", "size_ml", "size_g", "is_own", "ean"] if c in daily.columns]
    meta_unique = daily[["asin"] + meta_cols].drop_duplicates("asin")
    return meta_unique.merge(df, on="asin", how="right")


# Tab 1
with tabs[0]:
    st.subheader("✅ Visão Geral (com metadata enriquecendo tudo)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs (filtrados)", daily_f["asin"].nunique())
    c2.metric("Período", f"{daily_f['day'].min().date()} → {daily_f['day'].max().date()}" if not daily_f.empty else "NA")
    c3.metric("% dias em promo (média)", f"{(daily_f.groupby('asin')['is_promo'].mean().mean()*100):.1f}%" if not daily_f.empty else "NA")
    c4.metric("BSR mediano (média)", f"{daily_f.groupby('asin')['bsr'].median().mean():.0f}" if not daily_f.empty else "NA")

    st.markdown(
        """
**Tático:** use filtros (marca/segmento/meus vs concorrentes) para entender “onde está a guerra” agora.  
**Estratégico:** estabilize preço base por segmento e defina regras de promo por cluster.
        """
    )

    st.dataframe(enrich_with_meta(summ2).sort_values("bsr_med"), width='stretch', hide_index=True)

# Tab 2
with tabs[1]:
    st.subheader("📈 Evolução – Preço e BSR")
    st.markdown(
        """
**Tático:** comparar preço/BSR por SKU e identificar mudanças abruptas.  
**Estratégico:** identificar regimes de preço (padrões mensais) para governança.
        """
    )

    if freq == "Mensal":
        fig = px.line(monthly.sort_values("month_dt"), x="month_dt", y="price", color="asin", markers=True,
                      title="Preço médio mensal (price_effective)")
        st.plotly_chart(fig, width='stretch')

        fig2 = px.line(monthly.sort_values("month_dt"), x="month_dt", y="bsr_med", color="asin", markers=True,
                       title="BSR mediano mensal (menor é melhor)")
        st.plotly_chart(fig2, width='stretch')
    else:
        pick = st.multiselect("Selecione ASINs", options=asins, default=asins[: min(6, len(asins))])
        d = daily_f[daily_f["asin"].isin(pick)].copy()

        fig = px.line(d, x="day", y="price_effective", color="asin", title="Preço efetivo diário")
        st.plotly_chart(fig, width='stretch')

        fig2 = px.line(d, x="day", y="bsr", color="asin", title="BSR diário (menor é melhor)")
        st.plotly_chart(fig2, width='stretch')

# Tab 3
with tabs[2]:
    st.subheader("🏷️ Base vs Promo – Rebaixa e Profundidade")
    st.markdown(
        """
**Tático:** calibrar profundidade mínima que melhora BSR.  
**Estratégico:** limitar frequência promocional por segmento (evitar destruição do base).
        """
    )

    a = st.selectbox("SKU para detalhar", options=asins, index=0)
    g = daily_f[daily_f["asin"] == a].copy()
    title_name = g["sku_name"].iloc[0] if ("sku_name" in g.columns and len(g)) else a

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["day"], y=g["price_effective"], mode="lines", name="Preço efetivo"))
    fig.add_trace(go.Scatter(x=g["day"], y=g["price_base"], mode="lines", name="Preço base"))
    fig.update_layout(title=f"Preço efetivo vs Base – {title_name}", xaxis_title="Dia", yaxis_title="Preço (R$)")
    st.plotly_chart(fig, width='stretch')

    fig2 = px.scatter(g, x="discount_pct", y="bsr", color="is_promo",
                      title=f"Profundidade (vs base) x BSR – {title_name}",
                      labels={"discount_pct": "Desconto vs base", "bsr": "BSR"})
    st.plotly_chart(fig2, width='stretch')

    promo_depth = daily_f[daily_f["is_promo"]].copy()
    fig3 = px.box(promo_depth, x="asin", y=promo_depth["discount_pct"] * 100,
                  title="Distribuição de profundidade promocional (% vs base)", labels={"y": "% desconto vs base"})
    st.plotly_chart(fig3, width='stretch')

# Tab 4
with tabs[3]:
    st.subheader("🔗 Correlação – Quem compete com quem")
    st.markdown(
        """
**Tático:** vigiar pares com correlação alta para antecipar reação.  
**Estratégico:** formar “grupos competitivos” por faixa/segmento.
        """
    )

    options_full = sorted(daily_f[columns_map[ctl_prod]].dropna().unique().tolist())
    
    col1, col2 = st.columns(2)

    with col1:
        # O primeiro controle mostra tudo
        prod_1 = st.selectbox(f"Produto A ({ctl_prod})", options=options_full)

    with col2:
        # 3. A Mágica: Cria uma nova lista EXCLUINDO o que foi selecionado no prod_1
        options_filtered = [p for p in options_full if p != prod_1]
    
        # O segundo controle usa a lista filtrada
        prod_2 = st.selectbox(f"Produto B ({ctl_prod})", options=options_filtered)

    fig1 = px.bar(sens.sort_values("spearman_price_bsr", ascending=False),
                  x="asin", y="spearman_price_bsr",
                  title=f"Sensibilidade: {ctl_corr}(Preço, BSR)",
                  labels={"spearman_price_bsr": "Sensibilidade",
                          "asin": "Produto"})
    
    fig1.update_traces(texttemplate="%{y:.4f}", 
                       textposition="inside")
    
    fig1.update_layout(uniformtext_minsize=12, 
                       uniformtext_mode='hide')
    st.plotly_chart(fig1, width='stretch')


    fig2 = px.imshow(cross_corr, text_auto=True, aspect="auto", title=f"Correlação {ctl_corr} Cruzada (diária)")
    fig2.update_layout(yaxis_title = "Preço", xaxis_title = "BSR")
    st.plotly_chart(fig2, width='stretch')

    cross_filtered = scatter_cross_corr(daily_f, prod_1, prod_2, ctl_prod)
    fig2_scatter = px.scatter(cross_filtered, x=prod_2, y=prod_1,
                             title=f"Dispersão de Preço {prod_1} vs BSR {prod_2}",
                             labels={prod_1: "Preço", prod_2: "BSR"},
                             hover_data={'day': '|%d/%m/%Y'})
    st.plotly_chart(fig2_scatter, width='stretch')
    
  
    fig3 = px.imshow(price_corr, text_auto=True, aspect="auto", title=f"Correlação {ctl_corr} de preço (diária)")
    st.plotly_chart(fig3, width='stretch')

    # Filtrar NaNs: dropna no subset dos dois produtos
    # Isso garante que só sobram dias onde prod_1 E prod_2 têm valores
    price_filtered = scatter_price.dropna(subset=[prod_1, prod_2])
    fig3_scatter = px.scatter(price_filtered, x=prod_1, y=prod_2,
                             title=f"Dispersão de Preço: {prod_1} vs {prod_2}",
                             labels={prod_1: f"{prod_1}", prod_2: f"{prod_2}"},
                             hover_data={'day': '|%d/%m/%Y'})
    st.plotly_chart(fig3_scatter, width='stretch')


    fig4 = px.imshow(bsr_corr, text_auto=True, aspect="auto", title=f"Correlação {ctl_corr} de BSR (diária)")
    st.plotly_chart(fig4, width='stretch')

    # Filtrar NaNs: dropna no subset dos dois produtos
    # Isso garante que só sobram dias onde prod_1 E prod_2 têm valores
    bsr_filtered = scatter_bsr.dropna(subset=[prod_1, prod_2])
    fig4_scatter = px.scatter(bsr_filtered, x=prod_1, y=prod_2,
                             title=f"Dispersão de BSR: {prod_1} vs {prod_2}",
                             labels={prod_1: f"{prod_1}", prod_2: f"{prod_2}"},
                             hover_data={'day': '|%d/%m/%Y'})
    st.plotly_chart(fig4_scatter, width='stretch')

# Tab 5
with tabs[4]:
    st.subheader("📌 Índice de Preço (Price Index)")
    st.markdown(
        """
**Tático:** gatilhos de ajuste via índice vs referência.  
**Estratégico:** arquitetura premium/core/entry com índices consistentes.
        """
    )

    # if metadata includes is_own, offer pool selection
    leader_pool = asins
    if "is_own" in daily_f.columns and daily_f["is_own"].notna().any():
        leader_mode = st.radio("Escolher referência entre:", ["Todos", "Só concorrentes (is_own=0)", "Só meus (is_own=1)"], horizontal=True)
        if "concorrentes" in leader_mode.lower():
            leader_pool = sorted(daily_f.loc[daily_f["is_own"] == False, "asin"].unique().tolist()) or asins
        elif "meus" in leader_mode.lower():
            leader_pool = sorted(daily_f.loc[daily_f["is_own"] == True, "asin"].unique().tolist()) or asins

    leader = st.selectbox("Escolha o SKU referência (líder)", options=leader_pool, index=0)
    idx = price_index(daily_f, leader_asin=leader)

    if idx.empty:
        st.warning("Não consegui montar índice (verifique se o líder tem preço para as datas).")
    else:
        if freq == "Mensal":
            idx_m = idx.copy()
            idx_m["month_dt"] = idx_m["day"].dt.to_period("M").dt.to_timestamp()
            idx_m = idx_m.groupby(["asin", "month_dt"], as_index=False)["price_index"].mean()
            fig = px.line(idx_m, x="month_dt", y="price_index", color="asin", title=f"Índice de preço mensal vs {leader}")
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="Referência = 1.0")
            st.plotly_chart(fig, width='stretch')
        else:
            fig = px.line(idx, x="day", y="price_index", color="asin", title=f"Índice de preço diário vs {leader}")
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="Referência = 1.0")
            st.plotly_chart(fig, width='stretch')

# Tab 6
with tabs[5]:
    st.subheader("✨ Preço mágico + Elasticidade (proxy)")
    st.markdown(
        """
**Tático:** definir “preço de ataque” por SKU/segmento.  
**Estratégico:** escada de promo e governança por cluster.
        """
    )

    if not best_prices.empty:
        st.dataframe(enrich_with_meta(best_prices), width='stretch', hide_index=True)
    else:
        st.info("Sem cálculo robusto de preço mágico (poucos buckets repetidos).")

    a = st.selectbox("Detalhar curva de um SKU", options=asins, index=0, key="elastic_asin")
    el = elasticity_proxy(daily_f, a, bucket_round=2, min_n=6)
    if el.empty:
        st.warning("Sem dados suficientes por bucket para este SKU.")
    else:
        title_name = daily_f.loc[daily_f["asin"] == a, "sku_name"].iloc[0] if "sku_name" in daily_f.columns else a
        fig = px.line(el, x="price_bucket", y="bsr_median", markers=True,
                      title=f"Curva preço → BSR mediano (buckets) – {title_name}")
        st.plotly_chart(fig, width='stretch')

        fig2 = px.bar(el, x="price_bucket", y="elasticity_proxy",
                      title=f"Elasticidade (proxy) – {title_name} (Δlog(BSR)/Δpreço)")
        st.plotly_chart(fig2, width='stretch')

# Tab 7
with tabs[6]:
    st.subheader("🗺️ Mapa Competitivo (clusters) – enriquecido com metadata")
    st.markdown(
        """
**Tático:** enxergar quem é agressivo em promo e onde você precisa reagir.  
**Estratégico:** política por cluster (premium/core/entry).
        """
    )

    comp = competitive_map(daily_f, k=k_clusters)
    comp = enrich_with_meta(comp)
    st.dataframe(comp.sort_values(["cluster", "avg_price"]), width='stretch', hide_index=True)

    hover_cols = [c for c in ["asin", "sku_name", "brand", "segment", "pack_type", "pack_qty", "size_ml", "size_g", "is_own", "bsr_med", "spearman_price_bsr"] if c in comp.columns]
    fig = px.scatter(
        comp,
        x="avg_price",
        y="promo_share",
        color="cluster",
        size="avg_discount_when_promo",
        hover_data=hover_cols,
        title="Mapa: Preço médio vs % dias em promo (tamanho = profundidade média em promo)",
        labels={"avg_price": "Preço médio", "promo_share": "% dias em promo"},
    )
    st.plotly_chart(fig, width='stretch')

# Tab 8
with tabs[7]:
    st.subheader("🎯 Playbook de Eventos – com leitura por metadata")
    st.markdown(
        """
**Tático:** replicar profundidade/duração que trouxe ΔBSR melhor.  
**Estratégico:** escolher SKUs “hero” por evento e segmentar investimento.
        """
    )

    for ev in st.session_state["events"]:
        try:
            s = pd.to_datetime(ev["start"])
            e = pd.to_datetime(ev["end"])
        except Exception:
            st.warning(f"Evento com data inválida: {ev}")
            continue

        st.markdown(f"### {ev['name']} ({ev['start']} → {ev['end']})")
        es = event_summary(daily_f, s, e, baseline_days=baseline_days, pre=pre, post=post)
        if es.empty:
            st.info("Sem dados nesse período.")
            continue

        es = enrich_with_meta(es)
        st.dataframe(es, width='stretch', hide_index=True)

        fig = px.bar(es.sort_values("bsr_med_delta"), x="asin", y="bsr_med_delta",
                     title="Δ BSR mediano (janela - baseline) — negativo = melhorou")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, width='stretch')

        color_col = "is_own" if ("is_own" in es.columns and es["is_own"].notna().any()) else None
        fig2 = px.scatter(
            es,
            x="price_delta",
            y="bsr_med_delta",
            size="promo_share_window",
            color=color_col,
            hover_data=[c for c in ["sku_name", "brand", "segment", "discount_avg_window", "bsr_med_baseline", "bsr_med_window"] if c in es.columns],
            title="Trade-off ΔPreço vs ΔBSR (tamanho = % dias em promo na janela)",
        )
        fig2.add_hline(y=0, line_dash="dash")
        fig2.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig2, width='stretch')

# Tab 9
with tabs[8]:
    st.subheader("🧠 Recomendações (Tático & Estratégico) – contextualizadas")
    st.markdown(
        """
As recomendações aqui respeitam o **filtro atual** (marca/segmento/meus vs concorrentes).  
Isso é perfeito para reuniões de categoria: você troca o filtro e o plano muda na hora.
        """
    )

    gov = enrich_with_meta(summ2).copy()
    gov["promo_share_pct"] = gov["promo_share"] * 100
    gov["avg_discount_promo_pct"] = gov["avg_discount_when_promo"] * 100

    st.markdown("### 1) Onde promo tende a funcionar melhor (sensibilidade preço→BSR)")
    sens_rank = sens.dropna(subset=["spearman_price_bsr"]).sort_values("spearman_price_bsr", ascending=False)
    if len(sens_rank):
        top = sens_rank.head(5)["asin"].tolist()
        top_named = enrich_with_meta(pd.DataFrame({"asin": top})).merge(sens_rank, on="asin", how="left")
        st.dataframe(top_named, width='stretch', hide_index=True)
    else:
        st.info("Sem dados suficientes para ranquear sensibilidade no recorte filtrado.")

    st.markdown("### 2) Governança de promo (frequência e profundidade)")
    show_cols = [c for c in ["asin","sku_name","brand","segment","is_own","avg_price","promo_share_pct","avg_discount_promo_pct","bsr_med","spearman_price_bsr"] if c in gov.columns]
    st.dataframe(gov[show_cols].sort_values("promo_share_pct", ascending=False), width='stretch', hide_index=True)

    st.markdown("### 3) Sugestões de ‘preço de ataque’ (preço mágico)")
    if best_prices.empty:
        st.info("Sem cálculo robusto de preço mágico no recorte filtrado (poucos buckets repetidos).")
    else:
        st.dataframe(enrich_with_meta(best_prices), width='stretch', hide_index=True)

    st.markdown("### 4) Checklist tático (semana)")
    st.markdown(
        """
- **Alerta de guerra de preço:** observe pares com correlação alta (aba Correlação).
- **Guardrails por segmento:** limite % dias em promo e defina 2–3 degraus de profundidade.
- **Preço de ataque:** use ‘preço mágico’ + ‘eventos’ como referência mínima.
- **Evitar over-promo:** se promo_share sobe e BSR não melhora, pare e reavalie (provável driver fora de preço).
        """
    )

# Tab 10
with tabs[9]:
    st.subheader("🧪 Testes")

    st.markdown("### Preview dados brutos")
    st.dataframe(raw.head(10), width='stretch', hide_index=True)  

    st.dataframe(sens.head(10), width='stretch', hide_index=True)

st.caption("App de análise Preço x BSR com metadata enterprise (template + mapeamento + validação + cobertura).")

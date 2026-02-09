import io
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import statsmodels.formula.api as smf


# =========================
# Config
# =========================
st.set_page_config(page_title="Amazon Pricing Analytics (2 produtos) - Plotly", layout="wide")


# =========================
# Parsing / Prep
# =========================
EXPECTED_COLS = {
    "time": ["time", "Time", "timestamp", "data", "date"],
    "rank": ["sales rank", "Sales Rank", "rank", "bsr", "BSR"],
    "new_price": ["new price", "New Price", "price", "preço", "preco", "current price"],
    "list_price": ["list price", "List Price", "base price", "preço base", "preco base", "msrp"],
}

DOW_MAP = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def coerce_money_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    if s.str.contains(",").any():
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce", format="%d/%m/%Y, %H:%M:%S")
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt


def prep_product(df_raw: pd.DataFrame, label: str) -> pd.DataFrame:
    df = normalize_cols(df_raw)

    c_time = find_col(df, EXPECTED_COLS["time"])
    c_rank = find_col(df, EXPECTED_COLS["rank"])
    c_new = find_col(df, EXPECTED_COLS["new_price"])
    c_list = find_col(df, EXPECTED_COLS["list_price"])

    missing = [name for name, col in
               [("Time", c_time), ("Sales Rank", c_rank), ("New Price", c_new), ("List Price", c_list)] if col is None]
    if missing:
        raise ValueError(
            f"Colunas não encontradas: {missing}. Esperado algo como Time, Sales Rank, New Price, List Price."
        )

    out = pd.DataFrame({
        "Time": df[c_time],
        "Sales Rank": pd.to_numeric(df[c_rank], errors="coerce"),
        "New Price": coerce_money_series(df[c_new]),
        "List Price": coerce_money_series(df[c_list]),
    })

    out["datetime"] = parse_datetime_series(out["Time"])
    out = out.dropna(subset=["datetime", "Sales Rank", "New Price", "List Price"]).copy()

    out["date"] = out["datetime"].dt.date
    out["product"] = label
    out["discount_pct"] = (out["List Price"] - out["New Price"]) / out["List Price"]
    out["log_rank"] = np.log(out["Sales Rank"].clip(lower=1))
    out["neg_log_rank"] = -out["log_rank"]

    return out


def daily_agg(d: pd.DataFrame) -> pd.DataFrame:
    g = (d.groupby("date")
         .agg(
             obs=("Sales Rank", "size"),
             rank_mean=("Sales Rank", "mean"),
             rank_median=("Sales Rank", "median"),
             rank_min=("Sales Rank", "min"),
             rank_max=("Sales Rank", "max"),
             price_mean=("New Price", "mean"),
             price_median=("New Price", "median"),
             list_mean=("List Price", "mean"),
             discount_mean=("discount_pct", "mean"),
         )
         .reset_index())
    g["date"] = pd.to_datetime(g["date"])
    return g


# =========================
# Analytics
# =========================

def pick_magic_price(price_perf_A: pd.DataFrame) -> pd.Series | None:
    """
    Preço 'mágico' = balanceia performance (rank baixo) com evidência (days alto).
    Score menor é melhor.
    """
    if price_perf_A is None or price_perf_A.empty:
        return None

    pp = price_perf_A.copy()
    # normaliza para score: rank baixo é bom; days alto é bom
    rank_range = (pp["rank_mean"].max() - pp["rank_mean"].min())
    days_range = (pp["days"].max() - pp["days"].min())
    
    # Evita divisão por zero
    denom_rank = rank_range if rank_range > 0 else 1.0
    denom_days = days_range if days_range > 0 else 1.0

    pp["rank_norm"] = (pp["rank_mean"] - pp["rank_mean"].min()) / denom_rank
    pp["days_norm"] = (pp["days"] - pp["days"].min()) / denom_days

    # peso: performance > evidência, mas considera evidência
    pp["score"] = 0.65 * pp["rank_norm"] + 0.35 * (1 - pp["days_norm"])
    best = pp.sort_values("score").head(1)
    if best.empty:
        return None
    return best.iloc[0]


def classify_index_bins(index_perf: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica faixas de index em Verde/Amarelo/Vermelho com base no rank médio do A.
    Verde = melhores (menor rank), Vermelho = piores.
    """
    x = index_perf.copy()
    x = x.dropna(subset=["A_rank_mean"]).copy()
    if x.empty:
        return x

    # usa quantis para classificar
    q1 = x["A_rank_mean"].quantile(0.33)
    q2 = x["A_rank_mean"].quantile(0.66)

    def cat(v):
        if v <= q1:
            return "Verde (bom) ✅"
        elif v <= q2:
            return "Amarelo (ok) ⚠️"
        return "Vermelho (ruim) ❌"

    x["status"] = x["A_rank_mean"].apply(cat)
    return x.sort_values(["status", "A_rank_mean"])


def build_action_plan(
        label_A: str,
        label_B: str,
        coefA: pd.DataFrame,
        kinkA: dict | None,
        magic_row: pd.Series | None,
        index_classified: pd.DataFrame,
) -> dict:
    """
    Gera recomendações acionáveis em linguagem de negócio (marketing).
    """
    # elasticidades proxy
    own = float(coefA.loc["log_price_A", "coef"]) if "log_price_A" in coefA.index else np.nan
    cross = float(coefA.loc["log_price_B", "coef"]) if "log_price_B" in coefA.index else np.nan

    # preço mágico
    if magic_row is not None:
        magic_price = float(magic_row["price"])
        magic_days = int(magic_row["days"])
        magic_rank = float(magic_row["rank_mean"])
        magic_top20 = float(magic_row.get("top20_share", np.nan))
    else:
        magic_price, magic_days, magic_rank, magic_top20 = np.nan, 0, np.nan, np.nan

    # guardrail de breakpoint
    if kinkA is not None:
        breakpoint_price = float(kinkA["kink_price"])
        slope_below = float(kinkA["slope_below"])
        slope_above = float(kinkA["slope_above"])
    else:
        breakpoint_price, slope_below, slope_above = np.nan, np.nan, np.nan

    # faixa verde de index
    green = index_classified[index_classified["status"].str.contains("Verde")] if not index_classified.empty else pd.DataFrame()
    yellow = index_classified[index_classified["status"].str.contains("Amarelo")] if not index_classified.empty else pd.DataFrame()
    red = index_classified[index_classified["status"].str.contains("Vermelho")] if not index_classified.empty else pd.DataFrame()

    green_bins = green["index_bin"].astype(str).head(3).tolist() if not green.empty else []
    red_bins = red["index_bin"].astype(str).head(3).tolist() if not red.empty else []

    # narrativa simples
    if np.isfinite(own):
        if own < -1.5:
            own_msg = f"📉 **{label_A} é muito sensível** a preço (subir preço tende a piorar o rank rapidamente)."
        elif own < -0.5:
            own_msg = f"📉 **{label_A} é moderadamente sensível** a preço."
        else:
            own_msg = f"📉 **{label_A} parece pouco sensível** a preço (no histórico)."
    else:
        own_msg = "Não foi possível estimar a sensibilidade de preço (dados insuficientes)."

    if np.isfinite(cross):
        if abs(cross) < 0.2:
            cross_msg = f"⚖️ O preço do **{label_B} tem pouco efeito direto** no desempenho do {label_A} (no histórico)."
        else:
            cross_msg = f"⚖️ O preço do **{label_B} tem efeito relevante** no desempenho do {label_A} (vale monitorar)."
    else:
        cross_msg = "Não foi possível estimar o efeito do concorrente."

    # playbook (ações)
    actions = []
    actions.append(
        f"**Preço Mágico (para ganhar rank/volume):** ~ **R$ {magic_price:.2f}** (evidência: {magic_days} dias; rank médio ~ {magic_rank:.1f})." if np.isfinite(
            magic_price) else
        "**Preço Mágico:** não identificado (pouca recorrência de preços).")

    if np.isfinite(breakpoint_price):
        actions.append(
            f"**Guardrail de preço (Breakpoint):** evite ficar acima de **R$ {breakpoint_price:.2f}** por muito tempo se o objetivo for performance. (Acima do ponto a sensibilidade muda.)")
    else:
        actions.append(
            "**Guardrail de preço (Breakpoint):** não encontrado com segurança — use os preços mágicos e o index como guia.")

    if green_bins:
        actions.append(f"**Faixas de Price Index recomendadas (✅):** {', '.join(green_bins)}")
    if not yellow.empty:
        actions.append(f"**Faixas aceitáveis (⚠️):** {', '.join(yellow['index_bin'].astype(str).head(3).tolist())}")
    if red_bins:
        actions.append(f"**Faixas a evitar (❌):** {', '.join(red_bins)}")

    # regras “Se X então Y”
    rules = [
        f"Se o **rank piorar** por 3–5 dias seguidos, teste reduzir {label_A} em **-3% a -7%** por 48–96h e acompanhe o rank.",
        f"Se o {label_B} entrar em promoção forte (queda grande de preço), mantenha seu **Price Index** dentro das faixas ✅/⚠️ para não perder visibilidade.",
        f"Se você precisar **subir preço** (margem), suba em passos pequenos (ex.: +2%/+3%) e espere 3–7 dias para ver o efeito no rank.",
        f"Use ‘**pulsos promocionais**’ (2–4 dias) no preço mágico para recuperar rank antes de datas importantes."
    ]

    weekly = [
        "Toda **segunda-feira**: revisar rank e preço da semana anterior (A e B) + Price Index médio.",
        "Toda **quarta-feira**: checar se A entrou nas faixas ❌ do index e se o rank melhorou/piorou.",
        "Toda **sexta-feira**: decidir pulso de promo (fim de semana) se rank estiver fraco."
    ]

    return {
        "own_msg": own_msg,
        "cross_msg": cross_msg,
        "own": own,
        "cross": cross,
        "magic_price": magic_price,
        "breakpoint_price": breakpoint_price,
        "actions": actions,
        "rules": rules,
        "weekly": weekly
    }


def corr_pack(df_merge: pd.DataFrame) -> pd.DataFrame:
    corrs = {}
    for method in ["pearson", "spearman"]:
        corrs[method] = {
            "Preço A vs Preço B": df_merge["price_mean_A"].corr(df_merge["price_mean_B"], method=method),
            "Rank A vs Rank B": df_merge["rank_mean_A"].corr(df_merge["rank_mean_B"], method=method),
            "Rank A vs Preço A": df_merge["rank_mean_A"].corr(df_merge["price_mean_A"], method=method),
            "Rank A vs Preço B": df_merge["rank_mean_A"].corr(df_merge["price_mean_B"], method=method),
            "Rank A vs Price Index (A/B)": df_merge["rank_mean_A"].corr(df_merge["price_index"], method=method),
        }
    return pd.DataFrame(corrs).round(3)


def fit_elasticity(df_merge: pd.DataFrame):
    d = df_merge.copy()
    d["t"] = (d["date"] - d["date"].min()).dt.days
    d["dow"] = d["date"].dt.dayofweek.astype("category")

    d["neg_log_rank_A"] = -np.log(d["rank_mean_A"].clip(lower=1))
    d["neg_log_rank_B"] = -np.log(d["rank_mean_B"].clip(lower=1))
    d["log_price_A"] = np.log(d["price_mean_A"].clip(lower=0.01))
    d["log_price_B"] = np.log(d["price_mean_B"].clip(lower=0.01))

    # Elasticidade proxy (log-log) com controle de tendência e dia da semana
    mA = smf.ols("neg_log_rank_A ~ log_price_A + log_price_B + t + C(dow)", data=d).fit(cov_type="HC3")
    mB = smf.ols("neg_log_rank_B ~ log_price_B + log_price_A + t + C(dow)", data=d).fit(cov_type="HC3")

    def coef_table(m):
        return pd.DataFrame({"coef": m.params, "se": m.bse, "pvalue": m.pvalues})

    keyA = coef_table(mA).loc[["log_price_A", "log_price_B", "t"]].round(4)
    keyB = coef_table(mB).loc[["log_price_B", "log_price_A", "t"]].round(4)

    return keyA, keyB, mA, mB, d


def price_level_perf(daily_A: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    x = daily_A.copy()
    x["price"] = x["price_mean"].round(2)
    perf = (x.groupby("price")
            .agg(
                days=("date", "count"),
                rank_mean=("rank_mean", "mean"),
                rank_median=("rank_median", "median"),
                top10_share=("rank_mean", lambda s: (s <= 10).mean()),
                top20_share=("rank_mean", lambda s: (s <= 20).mean()),
                top50_share=("rank_mean", lambda s: (s <= 50).mean()),
            )
            .reset_index())
    perf = perf[perf["days"] >= min_days].copy()
    perf = perf.sort_values(["rank_mean", "days"], ascending=[True, False]).reset_index(drop=True)
    return perf


def bin_index_perf(df_merge: pd.DataFrame, bins=None) -> pd.DataFrame:
    if bins is None:
        bins = [0, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.20, 2.00]

    d = df_merge.copy()
    d["index_bin"] = pd.cut(d["price_index"], bins=bins, include_lowest=True)

    out = (d.groupby("index_bin", observed=True)
           .agg(
               days=("date", "count"),
               A_rank_mean=("rank_mean_A", "mean"),
               A_top10=("rank_mean_A", lambda s: (s <= 10).mean()),
               A_top20=("rank_mean_A", lambda s: (s <= 20).mean()),
               A_top50=("rank_mean_A", lambda s: (s <= 50).mean()),
               B_rank_mean=("rank_mean_B", "mean"),
           )
           .reset_index())
    
    # --- CORREÇÃO: Converter Interval para String ---
    out["index_bin"] = out["index_bin"].astype(str)
    
    return out


def discount_bins(daily_df: pd.DataFrame, who: str) -> pd.DataFrame:
    x = daily_df.copy()
    # Cria os intervalos
    x["disc_bin"] = pd.cut(x["discount_mean"], bins=[-1, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 1], include_lowest=True)
    
    out = (x.groupby("disc_bin", observed=True) # observed=True melhora performance com categoricas
           .agg(
               days=("date", "count"),
               price_mean=("price_mean", "mean"),
               rank_mean=("rank_mean", "mean"),
               top10=("rank_mean", lambda s: (s <= 10).mean()),
               top20=("rank_mean", lambda s: (s <= 20).mean()),
               top50=("rank_mean", lambda s: (s <= 50).mean()),
           )
           .reset_index())
    
    # --- CORREÇÃO: Converter Interval para String ---
    out["disc_bin"] = out["disc_bin"].astype(str)
    
    out["product"] = who
    return out


def fit_price_kink(daily_df: pd.DataFrame, min_seg_points: int = 30, n_grid: int = 40):
    d = daily_df.dropna(subset=["price_mean", "rank_mean"]).copy()
    d = d[d["price_mean"] > 0].copy()

    d["neg_log_rank"] = -np.log(d["rank_mean"].clip(lower=1))
    d["log_price"] = np.log(d["price_mean"])

    prices = d["price_mean"].values
    if len(prices) == 0:
        return None
    p_low, p_high = np.percentile(prices, [10, 90])
    if p_high <= p_low:
        return None

    grid = np.linspace(p_low, p_high, n_grid)
    best = None
    best_sse = np.inf
    best_kink = None
    rows = []

    for kink in grid:
        left = (d["price_mean"] <= kink).sum()
        right = (d["price_mean"] > kink).sum()
        if left < min_seg_points or right < min_seg_points:
            continue

        log_k = np.log(kink)
        d["hinge"] = np.maximum(0.0, d["log_price"] - log_k)

        try:
            m = smf.ols("neg_log_rank ~ log_price + hinge", data=d).fit()
            sse = float(np.sum(m.resid ** 2))

            rows.append({
                "kink_price": float(kink),
                "left_points": int(left),
                "right_points": int(right),
                "sse": sse,
                "coef_log_price": float(m.params.get("log_price", np.nan)),
                "coef_hinge": float(m.params.get("hinge", np.nan)),
            })

            if sse < best_sse:
                best_sse = sse
                best = m
                best_kink = kink
        except:
            continue

    if not rows:
        return None

    grid_tbl = pd.DataFrame(rows).sort_values("sse").reset_index(drop=True)
    b1 = float(best.params.get("log_price", np.nan))
    b2 = b1 + float(best.params.get("hinge", 0.0))

    return {
        "kink_price": float(best_kink),
        "slope_below": b1,
        "slope_above": b2,
        "grid": grid_tbl,
        "range": (float(p_low), float(p_high)),
    }


def predict_rank_from_model(model, base_design_df: pd.DataFrame, priceA: float, priceB: float, date_ref: pd.Timestamp,
                            dow_int: int):
    dmin = base_design_df["date"].min()
    t = int((pd.to_datetime(date_ref) - dmin).days)

    row = pd.DataFrame({
        "log_price_A": [np.log(max(priceA, 0.01))],
        "log_price_B": [np.log(max(priceB, 0.01))],
        "t": [t],
        "dow": pd.Categorical([dow_int], categories=sorted(base_design_df["dow"].cat.categories)),
    })

    yhat = float(model.predict(row)[0])
    rank_hat = float(np.exp(-yhat))
    return yhat, rank_hat


def make_excel_report(
        corr: pd.DataFrame,
        series: pd.DataFrame,
        price_perf: pd.DataFrame,
        index_perf: pd.DataFrame,
        discA: pd.DataFrame,
        discB: pd.DataFrame,
        coefA: pd.DataFrame,
        coefB: pd.DataFrame,
        kinkA: dict | None,
        kinkB: dict | None,
) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        corr.to_excel(writer, sheet_name="correlacoes")
        series.to_excel(writer, sheet_name="serie_diaria", index=False)
        price_perf.to_excel(writer, sheet_name="A_preco_levels", index=False)
        index_perf.to_excel(writer, sheet_name="A_price_index_bins", index=False)
        discA.to_excel(writer, sheet_name="A_desconto_bins", index=False)
        discB.to_excel(writer, sheet_name="B_desconto_bins", index=False)
        coefA.to_excel(writer, sheet_name="reg_A_keycoeffs")
        coefB.to_excel(writer, sheet_name="reg_B_keycoeffs")
        if kinkA:
            kinkA["grid"].to_excel(writer, sheet_name="kink_A_grid", index=False)
        if kinkB:
            kinkB["grid"].to_excel(writer, sheet_name="kink_B_grid", index=False)
    return bio.getvalue()


# =========================
# UI
# =========================
st.title("Amazon Pricing Analytics (2 produtos) — Interativo (Plotly)")

with st.sidebar:
    st.header("1) Upload")
    file_A = st.file_uploader("Produto A (seu produto) - CSV", type=["csv"])
    file_B = st.file_uploader("Produto B (concorrente) - CSV", type=["csv"])

    st.divider()
    st.header("2) Nomes")
    label_A = st.text_input("Nome do Produto A", value="Produto A")
    label_B = st.text_input("Nome do Produto B", value="Produto B")

    st.divider()
    st.header("3) Parâmetros")
    min_days_magic = st.slider("Mín. de dias para considerar preço (Preços Mágicos)", 3, 30, 5)
    min_seg_points = st.slider("Mín. de pontos por lado (Breakpoint)", 10, 80, 30)
    kink_grid = st.slider("Resolução da busca (Breakpoint)", 20, 120, 40)

    st.caption("Colunas esperadas (ou similares): Time, Sales Rank, New Price, List Price")

if not file_A or not file_B:
    st.info("Envie os dois CSVs no menu lateral para liberar as análises.")
    st.stop()

dfA_raw = pd.read_csv(file_A)
dfB_raw = pd.read_csv(file_B)

try:
    A = prep_product(dfA_raw, label_A)
    B = prep_product(dfB_raw, label_B)
except Exception as e:
    st.error(f"Erro ao preparar os dados: {e}")
    st.stop()

daily_A = daily_agg(A)
daily_B = daily_agg(B)

merge = pd.merge(daily_A, daily_B, on="date", how="inner", suffixes=("_A", "_B"))
if len(merge) == 0:
    st.error("Não há datas em comum entre os dois arquivos para análise comparativa.")
    st.stop()

merge["price_index"] = merge["price_mean_A"] / merge["price_mean_B"]

corr_tbl = corr_pack(merge)
coefA, coefB, modelA, modelB, design_base = fit_elasticity(merge)
price_perf_A = price_level_perf(daily_A, min_days=min_days_magic)
index_perf = bin_index_perf(merge)
discA = discount_bins(daily_A, label_A)
discB = discount_bins(daily_B, label_B)

kinkA = fit_price_kink(daily_A, min_seg_points=min_seg_points, n_grid=kink_grid)
kinkB = fit_price_kink(daily_B, min_seg_points=min_seg_points, n_grid=kink_grid)

series_tbl = merge[["date", "price_mean_A", "rank_mean_A", "price_mean_B", "rank_mean_B", "price_index"]].copy()

excel_bytes = make_excel_report(
    corr=corr_tbl,
    series=series_tbl,
    price_perf=price_perf_A,
    index_perf=index_perf,
    discA=discA,
    discB=discB,
    coefA=coefA,
    coefB=coefB,
    kinkA=kinkA,
    kinkB=kinkB,
)

# Definição das Abas (Adicionada aba 8)
tabs = st.tabs([
    "Visão Geral",
    "Correlação",
    "Elasticidade (proxy)",
    "Preços Mágicos",
    "Price Index",
    "Descontos",
    "Breakpoints",
    "Simulador",
    "Insights / Plano de Ação",
    "Export",
])

# -------------------------
# Tab 0: Visão Geral
# -------------------------
with tabs[0]:
    st.subheader("Como usar (prático)")
    with st.expander("O que olhar aqui e como decidir", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** entender o “filme” do ano: preço e rank andando ao longo do tempo.

        **KPIs usados:**
        - **Preço (média diária)**: quanto foi cobrado em média no dia.
        - **Rank/BSR (média diária)**: posição no ranking de vendas. **Quanto menor, melhor**.
        - **Price Index (A/B)**: `preço A ÷ preço B`.
        - Ex.: 0,70 significa A está 30% mais barato que B.

        **Decisão prática:**
        - Se você vê preço subindo e **rank piorando**, você está “pagando em performance”.
        - Se o concorrente cai preço e você não acompanha, é comum seu rank piorar (depende da categoria, por isso analisamos nas outras abas).
        """)

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=series_tbl["date"], y=series_tbl["price_mean_A"], mode="lines", name=label_A))
        fig_price.add_trace(go.Scatter(x=series_tbl["date"], y=series_tbl["price_mean_B"], mode="lines", name=label_B))
        fig_price.update_layout(title="Preço ao longo do tempo (média diária)", xaxis_title="Data",
                                yaxis_title="Preço (R$)", hovermode="x unified")
        st.plotly_chart(fig_price, use_container_width=True)

        fig_rank = go.Figure()
        fig_rank.add_trace(go.Scatter(x=series_tbl["date"], y=series_tbl["rank_mean_A"], mode="lines", name=label_A))
        fig_rank.add_trace(go.Scatter(x=series_tbl["date"], y=series_tbl["rank_mean_B"], mode="lines", name=label_B))
        fig_rank.update_layout(title="Rank (BSR) ao longo do tempo (média diária) — menor é melhor", xaxis_title="Data",
                               yaxis_title="Rank", hovermode="x unified")
        st.plotly_chart(fig_rank, use_container_width=True)

    with c2:
        st.subheader("Resumo rápido")
        st.metric("Dias em comum (para comparar A vs B)", f"{len(series_tbl)}")
        st.metric(f"Preço médio {label_A}", f"R$ {daily_A['price_mean'].mean():.2f}")
        st.metric(f"Preço médio {label_B}", f"R$ {daily_B['price_mean'].mean():.2f}")
        st.metric(f"Rank médio {label_A}", f"{daily_A['rank_mean'].mean():.1f}")
        st.metric(f"Rank médio {label_B}", f"{daily_B['rank_mean'].mean():.1f}")
        st.metric("Price Index médio (A/B)", f"{series_tbl['price_index'].mean():.3f}")

# -------------------------
# Tab 1: Correlação
# -------------------------
with tabs[1]:
    st.subheader("Como usar (prático)")
    with st.expander("O que esta aba responde e como decidir", expanded=True):
        st.markdown("""
        **Objetivo desta aba:** descobrir “o que anda junto”.

        **Como ler correlação (bem simples):**
        - Vai de **-1 a +1**.
        - Perto de **0** = não tem relação clara.
        - Perto de **+1** = andam juntos (quando um sobe, o outro tende a subir).
        - Perto de **-1** = andam ao contrário.

        **Decisão prática:**
        - Se **Rank A vs Price Index** tem correlação alta, significa que estar muito caro vs concorrente “machuca”.
        - Se **Preço A vs Preço B** tem correlação baixa, significa que o concorrente não está “sempre” se movendo igual você → pode ter janela de oportunidade.
        """)

    st.dataframe(corr_tbl, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_sc1 = px.scatter(daily_A, x="price_mean", y="rank_mean",
                             title=f"{label_A}: Preço vs Rank (menor é melhor)",
                             labels={"price_mean": "Preço (R$)", "rank_mean": "Rank (BSR)"})
        st.plotly_chart(fig_sc1, use_container_width=True)

    with c2:
        fig_sc2 = px.scatter(series_tbl, x="price_index", y="rank_mean_A",
                             title=f"{label_A}: Price Index (A/B) vs Rank",
                             labels={"price_index": "Price Index (A/B)", "rank_mean_A": "Rank (BSR)"})
        st.plotly_chart(fig_sc2, use_container_width=True)

# -------------------------
# Tab 2: Elasticidade (proxy)
# -------------------------
with tabs[2]:
    st.subheader("Como usar (prático)")
    with st.expander("Elasticidade (explicado para Marketing)", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** medir o quanto **o rank é sensível ao preço**.

        ### O que é “elasticidade” aqui?
        A gente usa um modelo que tenta responder:
        > “Quando o preço muda, o rank tende a mudar quanto?”

        Como não temos vendas, usamos o **Rank/BSR como proxy**.

        ### KPIs e cálculo
        - Transformamos o rank para um indicador de performance: `neg_log_rank = -log(rank)`
        (maior = melhor performance)
        - Transformamos preço em log: `log(preço)`
        - O modelo estima:
        `neg_log_rank_A ~ log(preço A) + log(preço B) + tendência + dia da semana`

        ### Como decidir
        - **Coeficiente de log(preço A)** (própria):
        - se for **negativo**, subir seu preço tende a **piorar** performance
        - quanto mais “negativo”, mais “dolorido” aumentar preço
        - **Coeficiente de log(preço B)** (cruzada):
        - mostra se o preço do concorrente impacta seu desempenho

        **Dica prática:** use isso para definir o quanto você pode subir preço antes de perder performance.
        """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"### {label_A} — impacto próprio e do concorrente")
        st.dataframe(coefA, use_container_width=True)
        st.caption(f"R² do modelo: {modelA.rsquared:.3f}")

    with c2:
        st.markdown(f"### {label_B} — impacto próprio e do seu produto")
        st.dataframe(coefB, use_container_width=True)
        st.caption(f"R² do modelo: {modelB.rsquared:.3f}")

    st.info("Interpretação: isso é uma bússola tática. Para ‘lucro ótimo’ precisaríamos de custo/margem e vendas.")

# -------------------------
# Tab 3: Preços Mágicos
# -------------------------
with tabs[3]:
    st.subheader("Como usar (prático)")
    with st.expander("Preços mágicos = preços que melhor performaram no seu histórico", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** achar preços que historicamente deram melhor resultado (rank menor).

        ### KPIs e cálculo
        Agrupamos por nível de preço (arredondado em centavos) e calculamos:
        - **days**: quantos dias aquele preço apareceu (evidência)
        - **rank_mean**: rank médio quando o preço foi esse
        - **top10_share / top20_share / top50_share**: % de dias que ficou em Top 10 / Top 20 / Top 50

        ### Como decidir
        - Para ações de **crescimento de volume/rank**: escolha preços com **rank_mean baixo** + **days alto**.
        - Evite tomar decisão com preço que apareceu só 1–2 dias (pouca evidência).
        """)

    st.dataframe(price_perf_A, use_container_width=True)

    top_prices = price_perf_A.head(15).copy()
    fig_bar = px.bar(
        top_prices.sort_values("rank_mean"),
        x="price",
        y="rank_mean",
        hover_data=["days", "top10_share", "top20_share", "top50_share"],
        title=f"{label_A}: Top preços por rank médio (melhor = menor)",
        labels={"price": "Preço (R$)", "rank_mean": "Rank médio (BSR)"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------
# Tab 4: Price Index
# -------------------------
with tabs[4]:
    st.subheader("Como usar (prático)")
    with st.expander("Price Index = sua posição de preço vs concorrente", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** entender qual faixa de preço relativo (A vs B) tende a ser melhor.

        ### KPI e cálculo
        - **Price Index (A/B)** = `preço A ÷ preço B`
        - 0,70 = A está 30% mais barato que B
        - 1,00 = preços iguais
        - 1,10 = A está 10% mais caro

        Agrupamos em faixas e medimos:
        - **A_rank_mean** e shares de Top10/Top20/Top50

        ### Como decidir
        - A melhor faixa para **performance** costuma ser onde **A_rank_mean é menor**.
        - Isso vira “guardrail”: manter o index dentro de uma faixa na maior parte do tempo.
        """)

    st.dataframe(index_perf, use_container_width=True)

    fig_idx = px.scatter(series_tbl, x="price_index", y="rank_mean_A",
                         title=f"{label_A}: Price Index (A/B) vs Rank",
                         labels={"price_index": "Price Index (A/B)", "rank_mean_A": "Rank (BSR)"})
    
    # --- CORREÇÃO AQUI ---
    # Adicionado key="tab4_price_index_scatter" para evitar conflito de ID
    st.plotly_chart(fig_idx, use_container_width=True, key="tab4_price_index_scatter")

# -------------------------
# Tab 5: Descontos
# -------------------------
with tabs[5]:
    st.subheader("Como usar (prático)")
    with st.expander("Desconto: quando ‘ancorar’ preço pode ajudar conversão", expanded=True):
        st.markdown("""
        **Objetivo desta aba:** entender se maior desconto (vs preço de lista) está associado a melhor performance.

        ### KPI e cálculo
        - **discount_pct** = `(List Price - New Price) / List Price`
        - 0,30 = 30% off

        Agrupamos em faixas e medimos rank e shares.

        ### Como decidir
        - Se faixas com desconto maior têm rank melhor, usar estratégia de:
        - manter list price como âncora
        - fazer promoções com desconto percebido
        """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"### {label_A}")
        st.dataframe(discA, use_container_width=True)
        fig_discA = px.bar(discA, x="disc_bin", y="rank_mean", title=f"{label_A}: Rank médio por faixa de desconto",
                           labels={"disc_bin": "Faixa de desconto", "rank_mean": "Rank médio (BSR)"})
        st.plotly_chart(fig_discA, use_container_width=True)

    with c2:
        st.markdown(f"### {label_B}")
        st.dataframe(discB, use_container_width=True)
        fig_discB = px.bar(discB, x="disc_bin", y="rank_mean", title=f"{label_B}: Rank médio por faixa de desconto",
                           labels={"disc_bin": "Faixa de desconto", "rank_mean": "Rank médio (BSR)"})
        st.plotly_chart(fig_discB, use_container_width=True)

# -------------------------
# Tab 6: Breakpoints
# -------------------------
with tabs[6]:
    st.subheader("Como usar (prático)")
    with st.expander("Breakpoint: o preço onde ‘passa a doer muito’ subir", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** encontrar um ponto em que a relação **preço → performance** muda.

        ### Como funciona (sem matemática pesada)
        O algoritmo testa vários candidatos de preço e escolhe o que melhor explica a mudança de “inclinação” da curva.

        ### Como decidir
        - Use o breakpoint como **linha de alerta**:
        - acima dele, o risco de perder performance pode ser maior (depende do slope acima)
        - Excelente para definir:
        - preço “dia a dia” (abaixo do ponto)
        - preço “promo” (bem abaixo)
        """)

    c1, c2 = st.columns(2)

    def kink_card(kink, name):
        if kink is None:
            st.warning(
                "Não foi possível achar breakpoint com os parâmetros atuais. Tente reduzir ‘mín. de pontos por lado’.")
            return
        st.metric("Breakpoint (preço)", f"R$ {kink['kink_price']:.2f}")
        st.metric("Sensibilidade abaixo (slope)", f"{kink['slope_below']:.3f}")
        st.metric("Sensibilidade acima (slope)", f"{kink['slope_above']:.3f}")
        st.caption("Quanto mais negativo o slope, mais subir preço tende a piorar performance (proxy).")
        st.dataframe(kink["grid"].head(15), use_container_width=True)

    with c1:
        st.markdown(f"### {label_A}")
        kink_card(kinkA, label_A)

    with c2:
        st.markdown(f"### {label_B}")
        kink_card(kinkB, label_B)

    # Visual do breakpoint em A
    fig_bp = px.scatter(daily_A, x="price_mean", y="rank_mean",
                        title=f"{label_A}: Preço vs Rank com breakpoint",
                        labels={"price_mean": "Preço (R$)", "rank_mean": "Rank (BSR)"})
    if kinkA is not None:
        fig_bp.add_vline(x=kinkA["kink_price"], line_dash="dash",
                         annotation_text=f"Breakpoint ~ R$ {kinkA['kink_price']:.2f}")
    st.plotly_chart(fig_bp, use_container_width=True)

# -------------------------
# Tab 7: Simulador
# -------------------------
with tabs[7]:
    st.subheader("Como usar (prático)")
    with st.expander("Simulador: testar cenários antes de mexer no preço", expanded=True):
        st.markdown(f"""
        **Objetivo desta aba:** responder rápido:
        > “Se eu colocar {label_A} em R$ X e {label_B} em R$ Y, o rank esperado do {label_A} melhora ou piora?”

        ### Como funciona
        O simulador usa a regressão da aba Elasticidade para prever o rank (proxy) do Produto A.

        ### Como decidir
        - Use o cenário pontual para validar uma hipótese (ex.: “se eu baixar 5%, recupero rank?”)
        - Use o mapa (grid) para encontrar regiões boas de preço relativo (A x B)
        """)

    baseline_A = float(daily_A["price_mean"].median())
    baseline_B = float(daily_B["price_mean"].median())
    last_date = pd.to_datetime(series_tbl["date"].max())

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        priceA_sim = st.number_input("Preço A (R$)", min_value=0.01, value=float(baseline_A), step=0.10)
        priceB_sim = st.number_input("Preço B (R$)", min_value=0.01, value=float(baseline_B), step=0.10)

    with col2:
        date_ref = st.date_input("Data de referência (tendência)", value=last_date.date())
        dow_choice = st.selectbox("Dia da semana", options=list(DOW_MAP.keys()),
                                  format_func=lambda k: DOW_MAP[k], index=int(last_date.dayofweek))

    with col3:
        st.caption("Baseline para comparação")
        st.write(f"- Baseline A (mediana): **R$ {baseline_A:.2f}**")
        st.write(f"- Baseline B (mediana): **R$ {baseline_B:.2f}**")

    _, rank_hat = predict_rank_from_model(
        model=modelA,
        base_design_df=design_base,
        priceA=priceA_sim,
        priceB=priceB_sim,
        date_ref=pd.to_datetime(date_ref),
        dow_int=int(dow_choice),
    )
    _, rank_base = predict_rank_from_model(
        model=modelA,
        base_design_df=design_base,
        priceA=baseline_A,
        priceB=baseline_B,
        date_ref=pd.to_datetime(date_ref),
        dow_int=int(dow_choice),
    )

    st.markdown("### Resultado do cenário (Produto A)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rank previsto (A)", f"{rank_hat:.1f}")
    with c2:
        st.metric("Rank baseline (A)", f"{rank_base:.1f}")
    with c3:
        delta = (rank_hat / rank_base - 1.0) * 100.0
        st.metric("Variação vs baseline", f"{delta:+.1f}%")
        st.caption("Rank menor é melhor. Se a variação for negativa, tende a melhorar.")

    st.divider()

    st.markdown("### Mapa de cenários (grid A x B) — Rank previsto do A")
    colg1, colg2, colg3 = st.columns(3)

    with colg1:
        a_min = st.number_input("A min (R$)", min_value=0.01, value=float(np.percentile(daily_A["price_mean"], 10)),
                                step=0.10)
        a_max = st.number_input("A max (R$)", min_value=0.01, value=float(np.percentile(daily_A["price_mean"], 90)),
                                step=0.10)
    with colg2:
        b_min = st.number_input("B min (R$)", min_value=0.01, value=float(np.percentile(daily_B["price_mean"], 10)),
                                step=0.10)
        b_max = st.number_input("B max (R$)", min_value=0.01, value=float(np.percentile(daily_B["price_mean"], 90)),
                                step=0.10)
    with colg3:
        grid_n = st.slider("Resolução do grid", 10, 40, 20)

    if a_max <= a_min or b_max <= b_min:
        st.warning("Ajuste os ranges: max precisa ser maior que min.")
    else:
        A_vals = np.linspace(a_min, a_max, grid_n)
        B_vals = np.linspace(b_min, b_max, grid_n)

        Z = np.zeros((grid_n, grid_n))
        for i, pa in enumerate(A_vals):
            for j, pb in enumerate(B_vals):
                _, r = predict_rank_from_model(
                    model=modelA,
                    base_design_df=design_base,
                    priceA=float(pa),
                    priceB=float(pb),
                    date_ref=pd.to_datetime(date_ref),
                    dow_int=int(dow_choice),
                )
                Z[j, i] = r  # y=B, x=A

        heat = go.Figure(
            data=go.Heatmap(
                z=Z,
                x=A_vals,
                y=B_vals,
                colorbar=dict(title="Rank previsto (A)"),
                hovertemplate="Preço A: R$ %{x:.2f}<br>Preço B: R$ %{y:.2f}<br>Rank A: %{z:.1f}<extra></extra>",
            )
        )
        heat.update_layout(
            title=f"Heatmap — Rank previsto do {label_A} (menor = melhor)",
            xaxis_title="Preço A (R$)",
            yaxis_title="Preço B (R$)",
        )
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Dica: busque regiões com rank previsto menor (melhor).")

# -------------------------
# Tab 8: Insights / Plano de Ação
# -------------------------
with tabs[8]:
    st.subheader("Insights e Plano de Ação (linguagem prática para Marketing)")

    with st.expander("Como usar esta aba (em 2 minutos)", expanded=True):
        st.markdown(f"""
        **Objetivo:** transformar os gráficos em decisões do dia a dia.

        ### KPIs (simples)
        - **Rank/BSR:** posição no ranking. **Quanto menor, melhor** (mais visibilidade/vendas prováveis).
        - **Preço médio diário:** preço praticado no dia.
        - **Price Index (A/B):** `preço do {label_A} ÷ preço do {label_B}`.
          - 0,70 = você está 30% mais barato; 1,00 = empate; 1,10 = 10% mais caro.
        - **Preços mágicos:** níveis de preço que historicamente tiveram **rank melhor** com evidência (dias).
        - **Breakpoint:** preço em que “subir mais” tende a mudar a sensibilidade (pode começar a doer mais).

        ### Como decidir
        1) Escolha uma **faixa de index (✅)** para ficar na maior parte do tempo.
        2) Use o **preço mágico** para pulsos promocionais (ganhar rank).
        3) Use o **breakpoint** como guardrail de “não ficar caro demais”.
        """)

    # prepara insights
    magic_row = pick_magic_price(price_perf_A)
    index_classified = classify_index_bins(index_perf)
    plan = build_action_plan(label_A, label_B, coefA, kinkA, magic_row, index_classified)

    # cards executivos
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Elasticidade própria (proxy)", f"{plan['own']:.3f}" if np.isfinite(plan["own"]) else "n/a")
        st.caption("Quanto mais negativo, mais subir preço tende a piorar o rank.")
    with c2:
        st.metric("Efeito do concorrente (proxy)", f"{plan['cross']:.3f}" if np.isfinite(plan["cross"]) else "n/a")
        st.caption("Se for relevante, acompanhar concorrente importa mais.")
    with c3:
        st.metric("Preço ‘mágico’ sugerido",
                  f"R$ {plan['magic_price']:.2f}" if np.isfinite(plan["magic_price"]) else "n/a")
        st.caption("Usar em pulsos para recuperar/girar rank.")

    st.markdown("### Diagnóstico rápido")
    st.write(plan["own_msg"])
    st.write(plan["cross_msg"])

    st.divider()

    # Faixas de index (verde/amarelo/vermelho)
    st.markdown("### Faixas de Price Index (A/B) — para decisão rápida")
    st.dataframe(index_classified, use_container_width=True)

    # gráfico interativo de performance por faixa
    idx_plot = index_classified.copy()
    idx_plot["A_rank_mean"] = idx_plot["A_rank_mean"].astype(float)

    fig_idxbar = px.bar(
        idx_plot.sort_values("A_rank_mean"),
        x="index_bin",
        y="A_rank_mean",
        color="status",
        title=f"Performance por faixa de Price Index — {label_A} (menor rank = melhor)",
        labels={"index_bin": "Faixa de Price Index (A/B)", "A_rank_mean": "Rank médio do A"}
    )
    st.plotly_chart(fig_idxbar, use_container_width=True)

    st.divider()

    # Preço mágico (e evidência)
    st.markdown("### Preço Mágico (com evidência)")
    if magic_row is None:
        st.warning("Não foi possível identificar um preço mágico com segurança (pouca repetição de níveis de preço).")
    else:
        st.markdown(f"""
        - **Preço sugerido:** **R$ {float(magic_row['price']):.2f}**
        - **Evidência:** apareceu em **{int(magic_row['days'])}** dias no histórico (boa confiança)
        - **Resultado:** rank médio ~ **{float(magic_row['rank_mean']):.1f}**
        - **Top20 share:** **{float(magic_row.get('top20_share', np.nan)) * 100:.1f}%** (se disponível)
        """)

    # gráfico: top 15 preços por rank
    top_prices = price_perf_A.sort_values("rank_mean").head(15).copy()
    fig_magic = px.scatter(
        top_prices,
        x="price",
        y="rank_mean",
        size="days",
        hover_data=["days", "top10_share", "top20_share", "top50_share"],
        title=f"Top 15 preços por performance — tamanho do ponto = evidência (dias)",
        labels={"price": "Preço (R$)", "rank_mean": "Rank médio (menor é melhor)"}
    )
    st.plotly_chart(fig_magic, use_container_width=True)

    st.divider()

    # Guardrail breakpoint
    st.markdown("### Guardrail de preço (Breakpoint)")
    if kinkA is None:
        st.info(
            "Breakpoint não encontrado com segurança. Use Price Index + Preços Mágicos como guardrails principais.")
    else:
        st.markdown(f"""
        - **Breakpoint estimado:** **R$ {kinkA['kink_price']:.2f}**
        - **Sensibilidade abaixo:** {kinkA['slope_below']:.3f}
        - **Sensibilidade acima:** {kinkA['slope_above']:.3f}

        **Como usar:** se o objetivo é **rank/volume**, tente manter o preço “dia a dia” abaixo do breakpoint.
        """)

    st.divider()

    # Plano de ação
    st.markdown("### Plano de Ação (pronto para executar)")
    st.markdown("**Ações recomendadas:**")
    for a in plan["actions"]:
        st.markdown(f"- {a}")

    st.markdown("**Regras ‘Se X então Y’:**")
    for r in plan["rules"]:
        st.markdown(f"- {r}")

    st.markdown("**Rotina semanal sugerida:**")
    for w in plan["weekly"]:
        st.markdown(f"- {w}")

    st.divider()

    # Checklist simples
    st.markdown("### Checklist de campanha (Marketing)")
    st.markdown("""
    - [ ] O Price Index ficou em faixa ✅ na maior parte da semana?
    - [ ] O rank melhorou nos dias de preço mágico/pulso?
    - [ ] O concorrente entrou em promoção forte? (se sim, reagimos com index)
    - [ ] Tivemos 2–4 dias de pulso (quando necessário) para recuperar rank?
    - [ ] Estamos evitando ficar acima do breakpoint no dia a dia?
    """)

# -------------------------
# Tab 9: Export
# -------------------------
with tabs[9]:
    st.subheader("Export Excel")
    with st.expander("Para que serve e como usar", expanded=True):
        st.markdown("""
        Use o Excel para:
        - compartilhar com time/comercial/finanças
        - guardar evidência do histórico
        - anexar num plano de ação (promo calendar)

        Inclui:
        - correlações
        - série diária
        - preços mágicos
        - bins de price index
        - bins de desconto
        - coeficientes do modelo
        - grid do breakpoint (quando aplicável)
        """)

    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name="amazon_pricing_2produtos_relatorio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
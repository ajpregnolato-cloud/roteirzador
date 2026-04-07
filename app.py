import os
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# =========================
# Configurações gerais
# =========================
APP_TITLE = "Agrupador de Clientes SP - 11 Regiões | Versão 5"
DEFAULT_N_REGIONS = 11
RANDOM_SEED = 42

PDF_DPI = 300
PDF_FIGSIZE = (16, 9)

matplotlib.rcParams["savefig.dpi"] = PDF_DPI
matplotlib.rcParams["figure.dpi"] = 140
matplotlib.rcParams["pdf.fonttype"] = 42  # melhor compatibilidade de fontes no PDF
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.antialiased"] = True
matplotlib.rcParams["lines.antialiased"] = True

REGION_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
]

SP_ENVELOPE = {
    "lat_min": -25.5,
    "lat_max": -19.0,
    "lon_min": -53.5,
    "lon_max": -44.0,
}


# =========================
# Base de dados / cluster
# =========================

def normalizar_coluna(nome: str) -> str:
    return str(nome).strip().lower().replace(" ", "_")


class KMeansNumpy:
    def __init__(self, n_clusters: int = 11, max_iter: int = 200, tol: float = 1e-5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Quantidade de pontos ({n_samples}) menor que o número de regiões ({self.n_clusters})."
            )

        centroids = []
        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.n_clusters):
            dists = np.min(np.sum((X[:, None, :] - np.array(centroids)[None, :, :]) ** 2, axis=2), axis=1)
            prob = dists / dists.sum()
            next_idx = rng.choice(n_samples, p=prob)
            centroids.append(X[next_idx])

        return np.array(centroids, dtype=float)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            distances = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)

            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
            centroids = new_centroids
            if shift <= self.tol:
                break

        distances = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        inertia = float(np.sum((X - centroids[labels]) ** 2))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        return self


def identificar_colunas(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    cols = list(df.columns)
    cols_norm = {normalizar_coluna(c): c for c in cols}

    lat_candidates = ["latitude", "lat"]
    lon_candidates = ["longitude", "long", "lng", "lon"]
    city_candidates = ["cidade", "municipio", "município"]
    uf_candidates = ["uf", "estado"]

    lat_col = next((cols_norm[c] for c in lat_candidates if c in cols_norm), None)
    lon_col = next((cols_norm[c] for c in lon_candidates if c in cols_norm), None)
    city_col = next((cols_norm[c] for c in city_candidates if c in cols_norm), None)
    uf_col = next((cols_norm[c] for c in uf_candidates if c in cols_norm), None)

    if lat_col is None and len(cols) >= 10:
        lat_col = cols[9]
    if lon_col is None and len(cols) >= 11:
        lon_col = cols[10]

    if lat_col is None or lon_col is None:
        raise ValueError("Não foi possível identificar as colunas de Latitude e Longitude.")

    return lat_col, lon_col, city_col, uf_col


def preparar_dados(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str, str]:
    lat_col, lon_col, city_col, uf_col = identificar_colunas(df)
    base = df.copy()

    if uf_col:
        base = base[base[uf_col].astype(str).str.upper().str.strip() == "SP"].copy()

    base[lat_col] = pd.to_numeric(base[lat_col], errors="coerce")
    base[lon_col] = pd.to_numeric(base[lon_col], errors="coerce")
    base = base.dropna(subset=[lat_col, lon_col]).copy()

    if base.empty:
        raise ValueError("Após o filtro e validação de coordenadas, não restaram registros válidos.")

    if city_col is None:
        base["Cidade"] = "NÃO INFORMADA"
        city_col = "Cidade"
    else:
        base[city_col] = base[city_col].fillna("NÃO INFORMADA").astype(str).str.strip()

    return base, lat_col, lon_col, city_col, uf_col


def ordenar_clusters_geograficamente(centroids: np.ndarray) -> Dict[int, int]:
    items = [(idx, lat, lon) for idx, (lat, lon) in enumerate(centroids)]
    items_sorted = sorted(items, key=lambda x: (-x[1], x[2]))
    return {old_idx: new_idx + 1 for new_idx, (old_idx, _, _) in enumerate(items_sorted)}


def aplicar_regioes(df: pd.DataFrame, lat_col: str, lon_col: str, n_regions: int = 11) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df[[lat_col, lon_col]].to_numpy(dtype=float)
    model = KMeansNumpy(n_clusters=n_regions, random_state=RANDOM_SEED)
    model.fit(X)

    mapping = ordenar_clusters_geograficamente(model.centroids_)
    df = df.copy()
    df["Regiao_ID"] = [mapping[label] for label in model.labels_]
    df["Regiao"] = df["Regiao_ID"].apply(lambda x: f"Região {x:02d}")
    df["Cor_Regiao"] = df["Regiao_ID"].apply(lambda x: REGION_COLORS[(x - 1) % len(REGION_COLORS)])

    centroid_rows = []
    for old_idx, (lat, lon) in enumerate(model.centroids_):
        rid = mapping[old_idx]
        centroid_rows.append(
            {
                "Regiao_ID": rid,
                "Regiao": f"Região {rid:02d}",
                "Centroide_Latitude": float(lat),
                "Centroide_Longitude": float(lon),
                "Cor_Regiao": REGION_COLORS[(rid - 1) % len(REGION_COLORS)],
            }
        )

    centroides = pd.DataFrame(centroid_rows).sort_values("Regiao_ID").reset_index(drop=True)
    return df, centroides


def resumir_regioes(df: pd.DataFrame, city_col: str, lat_col: str, lon_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cities_rows = []

    for rid in sorted(df["Regiao_ID"].unique()):
        bloco = df[df["Regiao_ID"] == rid].copy()
        cidades = sorted({c.strip() for c in bloco[city_col].dropna().astype(str).tolist() if c.strip()})
        rows.append(
            {
                "Regiao_ID": rid,
                "Regiao": f"Região {rid:02d}",
                "Qtde_Clientes": int(len(bloco)),
                "Qtde_Cidades": int(len(cidades)),
                "Latitude_Media": float(bloco[lat_col].mean()),
                "Longitude_Media": float(bloco[lon_col].mean()),
                "Cidades": ", ".join(cidades),
                "Cor_Regiao": REGION_COLORS[(rid - 1) % len(REGION_COLORS)],
            }
        )
        for cidade in cidades:
            qtd = int((bloco[city_col].astype(str).str.strip() == cidade).sum())
            cities_rows.append(
                {
                    "Regiao_ID": rid,
                    "Regiao": f"Região {rid:02d}",
                    "Cidade": cidade,
                    "Qtde_Clientes_na_Cidade": qtd,
                }
            )

    resumo = pd.DataFrame(rows).sort_values("Regiao_ID").reset_index(drop=True)
    cidades_regiao = pd.DataFrame(cities_rows).sort_values(["Regiao_ID", "Cidade"]).reset_index(drop=True)
    return resumo, cidades_regiao


# =========================
# Regras geográficas / layout
# =========================

def detectar_coordenadas_fora_sp(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    mask = (
        (df[lat_col] < SP_ENVELOPE["lat_min"])
        | (df[lat_col] > SP_ENVELOPE["lat_max"])
        | (df[lon_col] < SP_ENVELOPE["lon_min"])
        | (df[lon_col] > SP_ENVELOPE["lon_max"])
    )
    return df.loc[mask].copy()


def ajustar_limites_mapa(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    pad_ratio: float = 0.08,
    min_pad_lat: float = 0.15,
    min_pad_lon: float = 0.15,
) -> Tuple[float, float, float, float]:
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()

    lat_pad = max((lat_max - lat_min) * pad_ratio, min_pad_lat)
    lon_pad = max((lon_max - lon_min) * pad_ratio, min_pad_lon)
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def chunks(seq: List, size: int) -> List[List]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def formatar_int(v) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(v)


def salvar_pagina(pdf: PdfPages, fig):
    """Padroniza exportação de página PDF com melhor nitidez."""
    fig.set_size_inches(*PDF_FIGSIZE)
    pdf.savefig(fig, dpi=PDF_DPI, facecolor="white")
    plt.close(fig)


def desenhar_cabecalho(fig, titulo: str, subtitulo: str = ""):
    fig.patches.extend(
        [Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, facecolor="#1f2937", edgecolor="none", zorder=-1)]
    )
    fig.text(0.03, 0.965, titulo, color="white", fontsize=19, fontweight="bold", va="center")
    if subtitulo:
        fig.text(0.03, 0.935, subtitulo, color="#e5e7eb", fontsize=9, va="center")


def estilizar_tabela(table, header_color="#dbeafe", alt_color="#f8fafc", font_size=9, row_height=1.30):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_height)

    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#6b7280")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="#111827")
        elif r % 2 == 0:
            cell.set_facecolor(alt_color)


def adicionar_bloco_resumo(ax, linhas: List[Tuple[str, str]], cor_borda: str):
    ax.axis("off")
    y = 0.95
    ax.add_patch(
        Rectangle((0.02, 0.05), 0.96, 0.88, facecolor="#f8fafc", edgecolor=cor_borda, linewidth=2, transform=ax.transAxes)
    )
    for label, valor in linhas:
        ax.text(0.07, y, label, fontsize=10, color="#374151", weight="bold", va="top", transform=ax.transAxes)
        ax.text(0.55, y, valor, fontsize=10, color="#111827", va="top", transform=ax.transAxes)
        y -= 0.12


def plotar_mapa_regiao(ax, df_all: pd.DataFrame, bloco: pd.DataFrame, centro: pd.Series, lat_col: str, lon_col: str, cor: str, titulo: str):
    ax.scatter(df_all[lon_col], df_all[lat_col], s=8, alpha=0.08, c="#9ca3af", linewidths=0, rasterized=True)
    ax.scatter(bloco[lon_col], bloco[lat_col], s=36, alpha=0.95, c=cor, edgecolors="#111827", linewidths=0.35)
    ax.scatter(
        centro["Centroide_Longitude"],
        centro["Centroide_Latitude"],
        s=280,
        marker="X",
        c=cor,
        edgecolors="#111827",
        linewidths=1.0,
        zorder=5,
    )
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.20)
    xmin, xmax, ymin, ymax = ajustar_limites_mapa(bloco, lat_col, lon_col)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def pagina_capa(pdf: PdfPages, df: pd.DataFrame, resumo: pd.DataFrame, lat_col: str, lon_col: str, anomalias: pd.DataFrame):
    fig = plt.figure(figsize=PDF_FIGSIZE, facecolor="white")
    desenhar_cabecalho(
        fig,
        "Agrupamento de Clientes em 11 Regiões - Estado de São Paulo",
        "Relatório executivo com layout revisado para leitura, impressão e distribuição mais limpa.",
    )

    ax_left = fig.add_axes([0.04, 0.12, 0.54, 0.74])
    ax_right = fig.add_axes([0.62, 0.12, 0.34, 0.74])
    ax_right.axis("off")

    for rid in sorted(df["Regiao_ID"].unique()):
        bloco = df[df["Regiao_ID"] == rid]
        cor = REGION_COLORS[(rid - 1) % len(REGION_COLORS)]
        ax_left.scatter(
            bloco[lon_col],
            bloco[lat_col],
            s=24,
            c=cor,
            alpha=0.85,
            edgecolors="#111827",
            linewidths=0.25,
            label=f"Região {rid:02d} ({len(bloco)})",
        )
        centro = bloco[[lat_col, lon_col]].mean()
        ax_left.text(float(centro[lon_col]), float(centro[lat_col]), f"R{rid:02d}", fontsize=8, fontweight="bold", color="#111827")

    ax_left.set_title("Mapa Geral - Zoom Operacional do Estado de São Paulo", fontsize=14, fontweight="bold", pad=10)
    ax_left.set_xlabel("Longitude")
    ax_left.set_ylabel("Latitude")
    ax_left.set_xlim(SP_ENVELOPE["lon_min"], SP_ENVELOPE["lon_max"])
    ax_left.set_ylim(SP_ENVELOPE["lat_min"], SP_ENVELOPE["lat_max"])
    ax_left.grid(True, linestyle="--", alpha=0.18)
    ax_left.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)

    total_clientes = formatar_int(len(df))
    total_cidades = formatar_int(resumo["Qtde_Cidades"].sum())
    total_regioes = formatar_int(resumo["Regiao_ID"].nunique())

    texto = (
        f"Total de clientes válidos: {total_clientes}\n"
        f"Total de regiões: {total_regioes}\n"
        f"Total de cidades distintas nas regiões: {total_cidades}\n\n"
        f"Critério: clusterização geográfica por latitude/longitude.\n"
        f"Coordenadas fora do envelope esperado de SP: {formatar_int(len(anomalias))}.\n"
        "PDF otimizado para nitidez em tela e impressão (300 DPI)."
    )
    ax_right.text(
        0.02,
        0.98,
        texto,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    resumo_show = resumo[["Regiao", "Qtde_Clientes", "Qtde_Cidades"]].copy()
    resumo_show["Qtde_Clientes"] = resumo_show["Qtde_Clientes"].map(formatar_int)
    resumo_show["Qtde_Cidades"] = resumo_show["Qtde_Cidades"].map(formatar_int)
    table_ax = fig.add_axes([0.62, 0.20, 0.30, 0.36])
    table_ax.axis("off")
    table = table_ax.table(cellText=resumo_show.values, colLabels=resumo_show.columns, loc="center", cellLoc="center")
    estilizar_tabela(table, font_size=9, row_height=1.22)

    salvar_pagina(pdf, fig)


def pagina_resumo(pdf: PdfPages, resumo: pd.DataFrame):
    fig = plt.figure(figsize=PDF_FIGSIZE, facecolor="white")
    desenhar_cabecalho(fig, "Resumo Consolidado das Regiões", "Tabela consolidada com quantidade de clientes, cidades e centroides médios.")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.82])
    ax.axis("off")

    resumo_show = resumo[["Regiao", "Qtde_Clientes", "Qtde_Cidades", "Latitude_Media", "Longitude_Media"]].copy()
    resumo_show["Qtde_Clientes"] = resumo_show["Qtde_Clientes"].map(formatar_int)
    resumo_show["Qtde_Cidades"] = resumo_show["Qtde_Cidades"].map(formatar_int)
    resumo_show["Latitude_Media"] = resumo_show["Latitude_Media"].map(lambda x: f"{x:.5f}")
    resumo_show["Longitude_Media"] = resumo_show["Longitude_Media"].map(lambda x: f"{x:.5f}")

    table = ax.table(cellText=resumo_show.values, colLabels=resumo_show.columns, loc="center", cellLoc="center")
    estilizar_tabela(table, font_size=10, row_height=1.55)

    salvar_pagina(pdf, fig)


def pagina_anomalias(pdf: PdfPages, anomalias: pd.DataFrame, lat_col: str, lon_col: str, city_col: str):
    if anomalias.empty:
        return

    fig = plt.figure(figsize=PDF_FIGSIZE, facecolor="white")
    desenhar_cabecalho(
        fig,
        "Diagnóstico de Coordenadas Anômalas",
        "Registros fora do envelope geográfico esperado de SP e que podem distorcer os mapas.",
    )

    ax_map = fig.add_axes([0.05, 0.12, 0.42, 0.74])
    ax_tbl = fig.add_axes([0.52, 0.12, 0.43, 0.74])
    ax_tbl.axis("off")

    ax_map.scatter(anomalias[lon_col], anomalias[lat_col], s=60, c="#dc2626", alpha=0.85, edgecolors="#111827", linewidths=0.4)
    for _, row in anomalias.iterrows():
        ax_map.text(row[lon_col], row[lat_col], str(row[city_col])[:12], fontsize=8)
    ax_map.set_title("Registros fora do envelope esperado", fontsize=13, fontweight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(True, linestyle="--", alpha=0.20)

    tbl = anomalias[[city_col, lat_col, lon_col, "Regiao"]].copy()
    tbl.columns = ["Cidade", "Latitude", "Longitude", "Região"]
    tbl["Latitude"] = tbl["Latitude"].map(lambda x: f"{x:.6f}")
    tbl["Longitude"] = tbl["Longitude"].map(lambda x: f"{x:.6f}")
    table = ax_tbl.table(cellText=tbl.values, colLabels=tbl.columns, loc="center", cellLoc="center")
    estilizar_tabela(table, header_color="#fee2e2", alt_color="#fff7f7", font_size=9, row_height=1.45)

    salvar_pagina(pdf, fig)


def pagina_mapa_regiao(
    pdf: PdfPages,
    df: pd.DataFrame,
    bloco: pd.DataFrame,
    centro: pd.Series,
    rid: int,
    cor: str,
    lat_col: str,
    lon_col: str,
    cidades_df: pd.DataFrame,
):
    fig = plt.figure(figsize=PDF_FIGSIZE, facecolor="white")
    desenhar_cabecalho(
        fig,
        f"Detalhamento da Região {rid:02d}",
        "Página cartográfica dedicada. A tabela de cidades foi separada para maximizar legibilidade.",
    )

    ax_map = fig.add_axes([0.05, 0.12, 0.60, 0.74])
    ax_info = fig.add_axes([0.70, 0.18, 0.25, 0.60])

    plotar_mapa_regiao(ax_map, df, bloco, centro, lat_col, lon_col, cor, f"Mapa da Região {rid:02d}")

    linhas = [
        ("Região", f"{rid:02d}"),
        ("Clientes", formatar_int(len(bloco))),
        ("Cidades", formatar_int(cidades_df["Cidade"].nunique())),
        ("Latitude média", f"{bloco[lat_col].mean():.5f}"),
        ("Longitude média", f"{bloco[lon_col].mean():.5f}"),
        ("Cor", cor),
    ]
    adicionar_bloco_resumo(ax_info, linhas, cor)

    salvar_pagina(pdf, fig)


def pagina_tabela_cidades(pdf: PdfPages, rid: int, cor: str, page_idx: int, total_pages: int, cidades_chunk: pd.DataFrame):
    fig = plt.figure(figsize=PDF_FIGSIZE, facecolor="white")
    desenhar_cabecalho(fig, f"Região {rid:02d} - Cidades que compõem a região", f"Página {page_idx}/{total_pages} da listagem de cidades.")

    ax = fig.add_axes([0.06, 0.10, 0.88, 0.78])
    ax.axis("off")

    ax.text(0.0, 1.02, f"Distribuição municipal da Região {rid:02d}", fontsize=13, fontweight="bold", color="#111827", transform=ax.transAxes)
    ax.text(0.0, 0.98, "Tabela dedicada para evitar compressão do mapa e sobreposição visual.", fontsize=9, color="#4b5563", transform=ax.transAxes)

    shown = cidades_chunk[["Cidade", "Qtde_Clientes_na_Cidade"]].copy()
    shown["Qtde_Clientes_na_Cidade"] = shown["Qtde_Clientes_na_Cidade"].map(formatar_int)

    table = ax.table(cellText=shown.values, colLabels=["Cidade", "Qtde. Clientes"], loc="center", cellLoc="left", bbox=[0, 0.02, 1, 0.90])
    estilizar_tabela(table, header_color="#eef2ff", alt_color="#fafafa", font_size=10, row_height=1.45)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(cor)
            cell.set_text_props(color="white", weight="bold")
        if c == 1 and r > 0:
            cell._loc = "center"

    salvar_pagina(pdf, fig)


def gerar_pdf(
    pdf_path: str,
    df: pd.DataFrame,
    resumo: pd.DataFrame,
    cidades_regiao: pd.DataFrame,
    centroides: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    city_col: str,
):
    anomalias = detectar_coordenadas_fora_sp(df, lat_col, lon_col)

    metadata = {
        "Title": "Agrupamento Geográfico de Clientes SP",
        "Author": "Agrupador SP",
        "Subject": "Relatório geográfico de regiões",
    }

    with PdfPages(pdf_path, metadata=metadata) as pdf:
        pagina_capa(pdf, df, resumo, lat_col, lon_col, anomalias)
        pagina_resumo(pdf, resumo)
        pagina_anomalias(pdf, anomalias, lat_col, lon_col, city_col)

        max_rows_por_pagina = 28

        for rid in sorted(df["Regiao_ID"].unique()):
            bloco = df[df["Regiao_ID"] == rid].copy()
            cor = REGION_COLORS[(rid - 1) % len(REGION_COLORS)]
            centro = centroides[centroides["Regiao_ID"] == rid].iloc[0]
            cidades_df = cidades_regiao[cidades_regiao["Regiao_ID"] == rid].copy().reset_index(drop=True)

            pagina_mapa_regiao(pdf, df, bloco, centro, rid, cor, lat_col, lon_col, cidades_df)

            if cidades_df.empty:
                cidades_df = pd.DataFrame([{"Cidade": "NÃO INFORMADA", "Qtde_Clientes_na_Cidade": 0}])

            partes = chunks(cidades_df.to_dict("records"), max_rows_por_pagina)
            total_paginas_tabela = len(partes)

            for idx, parte in enumerate(partes, start=1):
                pagina_tabela_cidades(pdf, rid, cor, idx, total_paginas_tabela, pd.DataFrame(parte))


# =========================
# Exportação Excel
# =========================

def exportar_excel(xlsx_path: str, df: pd.DataFrame, resumo: pd.DataFrame, cidades_regiao: pd.DataFrame, centroides: pd.DataFrame):
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.sort_values(["Regiao_ID"]).to_excel(writer, index=False, sheet_name="clientes_agrupados")
        resumo.to_excel(writer, index=False, sheet_name="resumo_regioes")
        cidades_regiao.to_excel(writer, index=False, sheet_name="cidades_por_regiao")
        centroides.to_excel(writer, index=False, sheet_name="centroides")
        writer.book["clientes_agrupados"].freeze_panes = "A2"


# =========================
# GUI
# =========================
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("920x580")
        self.root.minsize(860, 540)

        self.excel_path = tk.StringVar()
        self.n_regions = tk.IntVar(value=DEFAULT_N_REGIONS)
        self.status = tk.StringVar(value="Selecione a planilha para processar.")

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Agrupamento Geográfico de Clientes - Estado de São Paulo", font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 12))

        ttk.Label(
            main,
            text=(
                "Versão 5 com PDF em alta nitidez: renderização em 300 DPI, fontes compatíveis com impressão "
                "e páginas estruturadas para leitura em tela e papel."
            ),
            wraplength=860,
            justify="left",
        ).pack(anchor="w", pady=(0, 16))

        arquivo_frame = ttk.LabelFrame(main, text="Planilha de entrada", padding=12)
        arquivo_frame.pack(fill="x", pady=(0, 12))
        ttk.Entry(arquivo_frame, textvariable=self.excel_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(arquivo_frame, text="Selecionar Excel", command=self.selecionar_arquivo).pack(side="left")

        cfg = ttk.LabelFrame(main, text="Configurações", padding=12)
        cfg.pack(fill="x", pady=(0, 12))
        ttk.Label(cfg, text="Quantidade de regiões:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(cfg, from_=2, to=50, textvariable=self.n_regions, width=10).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(cfg, text="(recomendado: 11 para SP)").grid(row=0, column=2, sticky="w", padx=(8, 0))

        info = ttk.LabelFrame(main, text="Saídas geradas", padding=12)
        info.pack(fill="both", expand=True, pady=(0, 12))
        ttk.Label(
            info,
            justify="left",
            text=(
                "1. Excel com clientes agrupados\n"
                "2. Excel com resumo por região\n"
                "3. Excel com cidades por região\n"
                "4. PDF com alta nitidez para projeção e impressão\n"
                "5. Página cartográfica separada da tabela em cada região"
            ),
        ).pack(anchor="w")

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(6, 8))
        ttk.Button(actions, text="Processar e Exportar", command=self.processar).pack(side="left")
        ttk.Button(actions, text="Fechar", command=self.root.destroy).pack(side="right")

        status_frame = ttk.LabelFrame(main, text="Status", padding=12)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status, wraplength=860, justify="left").pack(anchor="w")

    def selecionar_arquivo(self):
        path = filedialog.askopenfilename(title="Selecione a planilha Excel", filetypes=[("Arquivos Excel", "*.xlsx *.xls")])
        if path:
            self.excel_path.set(path)
            self.status.set(f"Planilha selecionada: {path}")

    def processar(self):
        path = self.excel_path.get().strip()
        if not path:
            messagebox.showwarning("Atenção", "Selecione uma planilha Excel primeiro.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Erro", "O arquivo selecionado não foi encontrado.")
            return

        try:
            self.status.set("Lendo planilha...")
            self.root.update_idletasks()
            df = pd.read_excel(path)
            base, lat_col, lon_col, city_col, _ = preparar_dados(df)

            n_regions = int(self.n_regions.get())
            if len(base) < n_regions:
                raise ValueError(f"A planilha possui {len(base)} registros válidos, insuficientes para {n_regions} regiões.")

            self.status.set("Calculando regiões geográficas...")
            self.root.update_idletasks()
            agrupado, centroides = aplicar_regioes(base, lat_col, lon_col, n_regions=n_regions)

            self.status.set("Montando resumo e distribuição municipal...")
            self.root.update_idletasks()
            resumo, cidades_regiao = resumir_regioes(agrupado, city_col, lat_col, lon_col)

            base_name = os.path.splitext(path)[0]
            xlsx_out = f"{base_name}_agrupado_{n_regions}_regioes_sp_v5.xlsx"
            pdf_out = f"{base_name}_agrupado_{n_regions}_regioes_sp_v5.pdf"

            self.status.set("Exportando Excel...")
            self.root.update_idletasks()
            exportar_excel(xlsx_out, agrupado, resumo, cidades_regiao, centroides)

            self.status.set("Gerando PDF em alta qualidade...")
            self.root.update_idletasks()
            gerar_pdf(pdf_out, agrupado, resumo, cidades_regiao, centroides, lat_col, lon_col, city_col)

            self.status.set(f"Processamento concluído com sucesso.\nExcel: {xlsx_out}\nPDF: {pdf_out}")
            messagebox.showinfo("Concluído", f"Arquivos gerados com sucesso:\n\nExcel:\n{xlsx_out}\n\nPDF:\n{pdf_out}")

        except Exception as e:
            erro = f"Erro ao processar:\n{str(e)}\n\nDetalhes técnicos:\n{traceback.format_exc()}"
            self.status.set(f"Falha no processamento: {str(e)}")
            messagebox.showerror("Erro", erro)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

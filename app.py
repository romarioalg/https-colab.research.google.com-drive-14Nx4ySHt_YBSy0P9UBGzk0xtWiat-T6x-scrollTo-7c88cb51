import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pulp as pl
import io
import os

# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Alocação Ótima de Produção",
    page_icon="🏭",
    layout="wide",
)

# ── CSS customizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"]          { background: #16181f; border-right: 1px solid #2a2d3a; }
  h1 { color: #e8e8e8; font-family: 'Courier New', monospace; letter-spacing: -1px; }
  h2, h3 { color: #c8c8c8; }
  .metric-card {
    background: #1a1d26;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { color: #00d4aa; font-size: 28px; font-weight: 700; font-family: monospace; }
  .status-ok   { color: #00d4aa; }
  .status-warn { color: #f0a500; }
  div[data-testid="stDownloadButton"] button {
    background: #00d4aa22;
    border: 1px solid #00d4aa66;
    color: #00d4aa;
    font-family: monospace;
    font-size: 13px;
  }
  div[data-testid="stDownloadButton"] button:hover {
    background: #00d4aa44;
    border-color: #00d4aa;
  }
</style>
""", unsafe_allow_html=True)

# ── Constantes ──────────────────────────────────────────────────────────────
MILP_TIMEOUT        = 246
FORBIDDEN_THRESHOLD = 998

# ── Helpers ─────────────────────────────────────────────────────────────────
def make_unique_column_names(names):
    unique, counts = [], {}
    for name in names:
        if name in counts:
            counts[name] += 1
            unique.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            unique.append(name)
    return unique


def carregar_arquivo(conteudo_bytes):
    raw = pd.read_excel(io.BytesIO(conteudo_bytes), header=None)

    COL_COD, COL_DESC, COL_QTD = 0, 1, 2
    COL_LMIN, COL_MAC_I, COL_MAC_F = 5, 6, 35

    nomes_maquinas = [str(m) for m in raw.iloc[0, COL_MAC_I:COL_MAC_F + 1].tolist()]
    linha_delay    = raw.iloc[1, COL_MAC_I:COL_MAC_F + 1].tolist()
    delays = {m: float(v) if pd.notna(v) else 0.0 for m, v in zip(nomes_maquinas, linha_delay)}

    df_prod = raw.iloc[2:].copy()
    df_prod.columns = make_unique_column_names(raw.iloc[0].tolist())

    col_names = list(df_prod.columns)
    df_prod.rename(columns={
        col_names[COL_COD] : "Produto",
        col_names[COL_DESC]: "Descricao",
        col_names[COL_QTD] : "_Quantidade",
        col_names[COL_LMIN]: "_LoteMinimo",
    }, inplace=True)

    df_prod["_Quantidade"] = pd.to_numeric(df_prod["_Quantidade"], errors="coerce").fillna(0)
    df_prod["_LoteMinimo"] = pd.to_numeric(df_prod["_LoteMinimo"], errors="coerce").fillna(0)
    df_prod = df_prod[(df_prod["Produto"].notna()) & (df_prod["_Quantidade"] > 0)]

    df_banco = df_prod.melt(
        id_vars=["Produto", "Descricao"],
        value_vars=nomes_maquinas,
        var_name="Maquina", value_name="Tempo"
    )
    df_banco["Tempo"] = pd.to_numeric(df_banco["Tempo"], errors="coerce")
    df_banco = df_banco.dropna(subset=["Tempo"])
    df_banco = df_banco[(df_banco["Tempo"] > 0) & (df_banco["Tempo"] < FORBIDDEN_THRESHOLD)]

    COL_DIV   = 3
    col_div   = col_names[COL_DIV]
    df_prod["_Divisao"] = pd.to_numeric(df_prod.get(col_div, 1), errors="coerce").fillna(1).replace(0, 1)

    valid = df_banco["Produto"].unique()
    dv    = df_prod[df_prod["Produto"].isin(valid)]

    lote        = dict(zip(dv["Produto"], dv["_Quantidade"].astype(int)))
    max_maquinas = dict(zip(dv["Produto"], dv["_Divisao"].astype(int)))
    lote_minimo  = dict(zip(dv["Produto"], dv["_LoteMinimo"].astype(int)))

    return df_banco, lote, max_maquinas, lote_minimo, delays, nomes_maquinas


def validar_entradas(df_banco, lote, lote_minimo):
    pb, pl_ = set(df_banco["Produto"].unique()), set(lote.keys())
    if pb != pl_:
        raise Exception(f"Divergência de produtos!\nNo lote e não no banco: {pl_ - pb}\nNo banco e não no lote: {pb - pl_}")
    for p in lote:
        if lote_minimo.get(p, 0) > lote[p]:
            raise Exception(f"Lote mínimo ({lote_minimo[p]}) > quantidade ({lote[p]}) para '{p}'.")


def expandir(assign, maquinas, produtos, descricao):
    base = pd.MultiIndex.from_product([maquinas, produtos], names=["Maquina", "Produto"]).to_frame(index=False)
    df   = base.merge(assign, on=["Maquina", "Produto"], how="left")
    df["Unidades"] = df["Unidades"].fillna(0)
    df["Descricao"] = df["Produto"].map(descricao)
    return df


def solver(df, lote, max_maquinas, lote_minimo, delays, progress_cb=None):
    maquinas  = df["Maquina"].unique()
    produtos  = list(lote.keys())
    tempo     = {(r["Maquina"], r["Produto"]): r["Tempo"] for _, r in df.iterrows()}
    descricao = dict(zip(df["Produto"], df["Descricao"]))

    # ── Solver A: Makespan ───────────────────────────────────────────────────
    if progress_cb: progress_cb(0.05, "Configurando solver Makespan…")
    mA = pl.LpProblem("A", pl.LpMinimize)
    xA = {(m, p): pl.LpVariable(f"x_{m}_{p}", 0, None, pl.LpInteger)
          for m in maquinas for p in produtos if (m, p) in tempo}
    yA = {(m, p): pl.LpVariable(f"y_{m}_{p}", cat="Binary")
          for m in maquinas for p in produtos if (m, p) in tempo}
    z  = {m: pl.LpVariable(f"z_{m}", cat="Binary") for m in maquinas}
    T  = pl.LpVariable("T", 0)
    CA = {m: pl.LpVariable(f"CA_{m}", 0) for m in maquinas}

    mA += T + 0.01 * pl.lpSum(CA[m] for m in maquinas) - 0.1 * pl.lpSum(yA[(m, p)] for (m, p) in yA)

    for p in produtos:
        mA += pl.lpSum(xA[(m, p)] for m in maquinas if (m, p) in xA) == lote[p]
    for (m, p) in xA:
        mA += xA[(m, p)] <= lote[p]        * yA[(m, p)]
        mA += xA[(m, p)] >= lote_minimo[p] * yA[(m, p)]
    for p in produtos:
        mA += pl.lpSum(yA[(m, p)] for m in maquinas if (m, p) in yA) <= max_maquinas[p]
    for m in maquinas:
        mA += pl.lpSum(yA[(m, p)] for p in produtos if (m, p) in yA) <= 1000 * z[m]
    for m in maquinas:
        mA += CA[m] == delays[m] + pl.lpSum(tempo[(m, p)] * xA[(m, p)] for p in produtos if (m, p) in xA)
    M_BIG = sum(max((tempo.get((m, p), 0) * lote[p]) for p in produtos) for m in maquinas)
    for m in maquinas:
        mA += CA[m] <= T + M_BIG * (1 - z[m])

    if progress_cb: progress_cb(0.15, "Rodando solver Makespan… (pode levar até 4 min)")
    mA.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=MILP_TIMEOUT))
    if pl.LpStatus[mA.status] != "Optimal":
        raise Exception(f"Solver Makespan não encontrou solução ótima: {pl.LpStatus[mA.status]}")

    carga = {m: delays[m] for m in maquinas}
    seqA, linhasA = [], []
    for (m, p), v in xA.items():
        val = v.value()
        if val and val > 0:
            ini, fim = carga[m], carga[m] + tempo[(m, p)] * val
            carga[m] = fim
            seqA.append([m, p, ini, fim, int(round(val))])
            linhasA.append([m, p, int(round(val))])

    assignA = pd.DataFrame(linhasA, columns=["Maquina", "Produto", "Unidades"])
    seqA    = pd.DataFrame(seqA,    columns=["Maquina", "Produto", "Inicio", "Fim", "Unidades"])
    makespan_val = T.value()

    # ── Solver B: Balanceado ────────────────────────────────────────────────
    if progress_cb: progress_cb(0.55, "Configurando solver Balanceado…")
    mB = pl.LpProblem("B", pl.LpMinimize)
    xB = {(m, p): pl.LpVariable(f"xB_{m}_{p}", 0, None, pl.LpInteger)
          for m in maquinas for p in produtos if (m, p) in tempo}
    yB = {(m, p): pl.LpVariable(f"yB_{m}_{p}", cat="Binary")
          for m in maquinas for p in produtos if (m, p) in tempo}
    C   = {m: pl.LpVariable(f"C_{m}", 0) for m in maquinas}
    media   = pl.lpSum(C[m] for m in maquinas) / len(maquinas)
    dev_pos = {m: pl.LpVariable(f"dpos_{m}", 0) for m in maquinas}
    dev_neg = {m: pl.LpVariable(f"dneg_{m}", 0) for m in maquinas}

    mB += pl.lpSum(dev_pos[m] + dev_neg[m] for m in maquinas) + 0.1 * pl.lpSum(C[m] for m in maquinas)

    for p in produtos:
        mB += pl.lpSum(xB[(m, p)] for m in maquinas if (m, p) in xB) == lote[p]
    for (m, p) in xB:
        mB += xB[(m, p)] <= lote[p]        * yB[(m, p)]
        mB += xB[(m, p)] >= lote_minimo[p] * yB[(m, p)]
    for p in produtos:
        mB += pl.lpSum(yB[(m, p)] for m in maquinas if (m, p) in yB) <= max_maquinas[p]
    for m in maquinas:
        mB += C[m] == delays[m] + pl.lpSum(tempo[(m, p)] * xB[(m, p)] for p in produtos if (m, p) in xB)
    for m in maquinas:
        mB += C[m] - media == dev_pos[m] - dev_neg[m]

    if progress_cb: progress_cb(0.65, "Rodando solver Balanceado… (pode levar até 4 min)")
    mB.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=MILP_TIMEOUT))
    if pl.LpStatus[mB.status] != "Optimal":
        raise Exception(f"Solver Balanceado não encontrou solução ótima: {pl.LpStatus[mB.status]}")

    carga = {m: delays[m] for m in maquinas}
    seqB, linhasB = [], []
    for (m, p), v in xB.items():
        val = v.value()
        if val and val > 0:
            ini, fim = carga[m], carga[m] + tempo[(m, p)] * val
            carga[m] = fim
            seqB.append([m, p, ini, fim, int(round(val))])
            linhasB.append([m, p, int(round(val))])

    assignB = pd.DataFrame(linhasB, columns=["Maquina", "Produto", "Unidades"])
    seqB    = pd.DataFrame(seqB,    columns=["Maquina", "Produto", "Inicio", "Fim", "Unidades"])
    cargas_b = {m: C[m].value() or 0 for m in maquinas}

    return assignA, seqA, assignB, seqB, descricao, makespan_val, cargas_b


def gerar_gantt(seq, maquinas, delays, lote, titulo, dark=True):
    bg  = "#0f1117" if dark else "white"
    fg  = "#e8e8e8" if dark else "#111"
    grid_c = "#2a2d3a" if dark else "#ddd"

    produtos_unicos = list(lote.keys())
    palette = plt.cm.tab20.colors
    cores   = {p: palette[i % 20] for i, p in enumerate(produtos_unicos)}

    fig, ax = plt.subplots(figsize=(18, max(6, len(maquinas) * 0.55)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    yticks, ylabels = [], []
    for i, m in enumerate(maquinas):
        y = i
        yticks.append(y)
        ylabels.append(m)

        d = delays.get(m, 0)
        if d > 0:
            ax.barh(y, d, color="#444" if dark else "#bbb", height=0.6, left=0)

        bloco = seq[seq["Maquina"] == m]
        if bloco.empty:
            ax.barh(y, 0, height=0.6)
            continue

        for _, r in bloco.iterrows():
            dur = r["Fim"] - r["Inicio"]
            ax.barh(y, dur, left=r["Inicio"], height=0.6,
                    color=cores[r["Produto"]], edgecolor=bg, linewidth=0.5)
            if dur > 0.3:
                ax.text((r["Inicio"] + r["Fim"]) / 2, y,
                        f"{r['Produto']}\n({int(r['Unidades'])})",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "black", fontweight="bold")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8, color=fg)
    ax.set_xlabel("Tempo", color=fg, fontsize=9)
    ax.set_title(titulo, color=fg, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=fg)
    ax.spines[:].set_color(grid_c)
    ax.xaxis.grid(True, color=grid_c, linewidth=0.4, linestyle="--")
    ax.set_axisbelow(True)

    legend = [mpatches.Patch(color=cores[p], label=p) for p in produtos_unicos]
    ax.legend(handles=legend, loc="lower right", fontsize=7,
              facecolor=bg, edgecolor=grid_c, labelcolor=fg)

    fig.tight_layout()
    return fig


def excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Detalhe", index=False)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# INTERFACE
# ════════════════════════════════════════════════════════════════════════════

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Alocação Ótima")
    st.markdown("---")
    st.markdown("""
**Como usar:**
1. Faça o upload do arquivo Excel
2. Clique em **Rodar Otimização**
3. Veja os Gantt e baixe os resultados

**Formato esperado:**
- Linha 1: cabeçalhos + nomes de máquinas (G→AJ)
- Linha 2: delays por máquina
- Linha 3+: produtos com quantidades e tempos
""")
    st.markdown("---")
    st.markdown("<small style='color:#555'>V8.6 · MILP via CBC</small>", unsafe_allow_html=True)

# ── Cabeçalho ───────────────────────────────────────────────────────────────
st.markdown("# 🏭 Sistema de Alocação Ótima de Produção")
st.markdown("Faça upload do arquivo Excel de planejamento para calcular as alocações **Makespan** e **Balanceada**.")
st.markdown("---")

# ── Upload ──────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂 Selecione o arquivo Excel (.xlsx ou .xlsm)",
    type=["xlsx", "xlsm"],
    label_visibility="collapsed"
)

if uploaded:
    st.success(f"✔ Arquivo carregado: **{uploaded.name}**")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        rodar = st.button("⚡ Rodar Otimização", use_container_width=True, type="primary")

    if rodar:
        progresso  = st.progress(0)
        status_txt = st.empty()

        def atualizar(pct, msg):
            progresso.progress(pct)
            status_txt.markdown(f"<span style='color:#00d4aa'>⏳ {msg}</span>", unsafe_allow_html=True)

        try:
            atualizar(0.02, "Lendo arquivo…")
            conteudo = uploaded.read()
            df_banco, lote, max_maq, lote_min, delays, maquinas = carregar_arquivo(conteudo)

            atualizar(0.04, "Validando dados…")
            validar_entradas(df_banco, lote, lote_min)

            assignA, seqA, assignB, seqB, desc, makespan_val, cargas_b = solver(
                df_banco, lote, max_maq, lote_min, delays, progress_cb=atualizar
            )

            atualizar(0.92, "Expandindo resultados…")
            assignA = expandir(assignA, maquinas, lote.keys(), desc)
            assignB = expandir(assignB, maquinas, lote.keys(), desc)

            atualizar(0.96, "Gerando visualizações…")

            progresso.progress(1.0)
            status_txt.markdown("<span class='status-ok'>✔ Otimização concluída!</span>", unsafe_allow_html=True)

            st.markdown("---")

            # ── Métricas ─────────────────────────────────────────────────────
            n_maq_ativas = len(seqA["Maquina"].unique())
            n_prod       = len(lote)
            carga_max_b  = max(cargas_b.values()) if cargas_b else 0
            carga_min_b  = min(v for v in cargas_b.values() if v > 0) if cargas_b else 0

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, "Produtos", n_prod),
                (c2, "Máquinas Ativas", n_maq_ativas),
                (c3, f"Makespan", f"{makespan_val:.1f}"),
                (c4, "Carga Máx. (Bal.)", f"{carga_max_b:.1f}"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Gantt ─────────────────────────────────────────────────────────
            tab1, tab2 = st.tabs(["📊 Makespan", "⚖️ Balanceado"])

            with tab1:
                st.markdown("### Gráfico de Gantt — Makespan mínimo")
                fig_a = gerar_gantt(seqA, maquinas, delays, lote, "Makespan")
                st.pyplot(fig_a, use_container_width=True)

                bytes_a = excel_bytes(assignA)
                st.download_button(
                    "⬇ Baixar Excel — Makespan",
                    data=bytes_a,
                    file_name="resultado_makespan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with tab2:
                st.markdown("### Gráfico de Gantt — Carga Balanceada")
                fig_b = gerar_gantt(seqB, maquinas, delays, lote, "Balanceado")
                st.pyplot(fig_b, use_container_width=True)

                bytes_b = excel_bytes(assignB)
                st.download_button(
                    "⬇ Baixar Excel — Balanceado",
                    data=bytes_b,
                    file_name="resultado_balanceado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            progresso.empty()
            status_txt.empty()
            st.error(f"❌ Erro: {e}")

else:
    st.info("👆 Faça o upload do arquivo Excel para começar.")

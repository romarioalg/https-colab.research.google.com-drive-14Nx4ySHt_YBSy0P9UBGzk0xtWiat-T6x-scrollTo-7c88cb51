import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pulp as pl
import io
import time
import threading
import random

st.set_page_config(page_title="Alocação Ótima de Produção", page_icon="🏭", layout="wide")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"]          { background: #16181f; border-right: 1px solid #2a2d3a; }
  h1,h2,h3 { color: #e8e8e8; }
  h1 { font-family: 'Courier New', monospace; letter-spacing: -1px; }
  .metric-card {
    background: #1a1d26; border: 1px solid #2a2d3a;
    border-radius: 8px; padding: 16px 20px; text-align: center;
  }
  .metric-label { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { color: #00d4aa; font-size: 28px; font-weight: 700; font-family: monospace; }
  .status-ok { color: #00d4aa; }
  .phase-badge {
    display: inline-block; background: #00d4aa22; border: 1px solid #00d4aa55;
    border-radius: 4px; padding: 2px 10px; color: #00d4aa;
    font-family: monospace; font-size: 13px; margin-bottom: 8px;
  }
  div[data-testid="stDownloadButton"] button {
    background: #00d4aa22; border: 1px solid #00d4aa66;
    color: #00d4aa; font-family: monospace; font-size: 13px;
  }
  div[data-testid="stDownloadButton"] button:hover {
    background: #00d4aa44; border-color: #00d4aa;
  }
</style>
""", unsafe_allow_html=True)

MILP_TIMEOUT        = 246
FORBIDDEN_THRESHOLD = 998

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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
    nomes_maquinas = [str(m) for m in raw.iloc[0, COL_MAC_I:COL_MAC_F+1].tolist()]
    linha_delay = raw.iloc[1, COL_MAC_I:COL_MAC_F+1].tolist()
    delays = {m: float(v) if pd.notna(v) else 0.0 for m, v in zip(nomes_maquinas, linha_delay)}
    df_prod = raw.iloc[2:].copy()
    df_prod.columns = make_unique_column_names(raw.iloc[0].tolist())
    col_names = list(df_prod.columns)
    df_prod.rename(columns={
        col_names[COL_COD]: "Produto", col_names[COL_DESC]: "Descricao",
        col_names[COL_QTD]: "_Quantidade", col_names[COL_LMIN]: "_LoteMinimo",
    }, inplace=True)
    df_prod["_Quantidade"] = pd.to_numeric(df_prod["_Quantidade"], errors="coerce").fillna(0)
    df_prod["_LoteMinimo"] = pd.to_numeric(df_prod["_LoteMinimo"], errors="coerce").fillna(0)
    df_prod = df_prod[(df_prod["Produto"].notna()) & (df_prod["_Quantidade"] > 0)]
    df_banco = df_prod.melt(id_vars=["Produto","Descricao"], value_vars=nomes_maquinas,
                             var_name="Maquina", value_name="Tempo")
    df_banco["Tempo"] = pd.to_numeric(df_banco["Tempo"], errors="coerce")
    df_banco = df_banco.dropna(subset=["Tempo"])
    df_banco = df_banco[(df_banco["Tempo"] > 0) & (df_banco["Tempo"] < FORBIDDEN_THRESHOLD)]
    col_div = col_names[3]
    df_prod["_Divisao"] = pd.to_numeric(df_prod.get(col_div, 1), errors="coerce").fillna(1).replace(0, 1)
    valid = df_banco["Produto"].unique()
    dv = df_prod[df_prod["Produto"].isin(valid)]
    lote        = dict(zip(dv["Produto"], dv["_Quantidade"].astype(int)))
    max_maquinas = dict(zip(dv["Produto"], dv["_Divisao"].astype(int)))
    lote_minimo  = dict(zip(dv["Produto"], dv["_LoteMinimo"].astype(int)))
    return df_banco, lote, max_maquinas, lote_minimo, delays, nomes_maquinas

def validar_entradas(df_banco, lote, lote_minimo):
    pb, pl_ = set(df_banco["Produto"].unique()), set(lote.keys())
    if pb != pl_:
        raise Exception(f"Divergência de produtos!\nNo lote e não no banco: {pl_-pb}\nNo banco e não no lote: {pb-pl_}")
    for p in lote:
        if lote_minimo.get(p, 0) > lote[p]:
            raise Exception(f"Lote mínimo ({lote_minimo[p]}) > quantidade ({lote[p]}) para '{p}'.")

def expandir(assign, maquinas, produtos, descricao):
    base = pd.MultiIndex.from_product([maquinas, produtos], names=["Maquina","Produto"]).to_frame(index=False)
    df   = base.merge(assign, on=["Maquina","Produto"], how="left")
    df["Unidades"] = df["Unidades"].fillna(0)
    df["Descricao"] = df["Produto"].map(descricao)
    return df

def excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Detalhe", index=False)
    return buf.getvalue()

# ─────────────────────────────────────────────
# GANTT RENDERER
# ─────────────────────────────────────────────
def gerar_gantt(seq, maquinas, delays, lote, titulo, highlight_last=False):
    bg, fg, grid_c = "#0f1117", "#e8e8e8", "#2a2d3a"
    palette = plt.cm.tab20.colors
    cores   = {p: palette[i % 20] for i, p in enumerate(lote.keys())}

    fig, ax = plt.subplots(figsize=(18, max(5, len(maquinas) * 0.55)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    last_row = seq.iloc[-1] if not seq.empty else None

    for i, m in enumerate(maquinas):
        d = delays.get(m, 0)
        if d > 0:
            ax.barh(i, d, color="#333", height=0.6, left=0)

        bloco = seq[seq["Maquina"] == m]
        for _, r in bloco.iterrows():
            dur = r["Fim"] - r["Inicio"]
            is_last = highlight_last and last_row is not None and r["Produto"] == last_row["Produto"] and r["Maquina"] == last_row["Maquina"]
            edge_color = "#ffffff" if is_last else bg
            edge_lw    = 2.0 if is_last else 0.5
            alpha      = 1.0 if is_last else 0.85
            ax.barh(i, dur, left=r["Inicio"], height=0.6,
                    color=cores[r["Produto"]], edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha)
            if dur > 0.3:
                ax.text((r["Inicio"]+r["Fim"])/2, i,
                        f"{r['Produto']}\n({int(r['Unidades'])})",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")

    ax.set_yticks(range(len(maquinas)))
    ax.set_yticklabels(maquinas, fontsize=8, color=fg)
    ax.set_xlabel("Tempo", color=fg, fontsize=9)
    ax.set_title(titulo, color=fg, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=fg)
    ax.spines[:].set_color(grid_c)
    ax.xaxis.grid(True, color=grid_c, linewidth=0.4, linestyle="--")
    ax.set_axisbelow(True)
    legend = [mpatches.Patch(color=cores[p], label=p) for p in lote.keys()]
    ax.legend(handles=legend, loc="lower right", fontsize=7,
              facecolor=bg, edgecolor=grid_c, labelcolor=fg)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SOLVER COM CALLBACK DE ANIMAÇÃO
# ─────────────────────────────────────────────
def solver_com_animacao(df, lote, max_maquinas, lote_minimo, delays,
                         gantt_placeholder, status_placeholder, progresso):
    maquinas  = df["Maquina"].unique()
    produtos  = list(lote.keys())
    tempo     = {(r["Maquina"], r["Produto"]): r["Tempo"] for _, r in df.iterrows()}
    descricao = dict(zip(df["Produto"], df["Descricao"]))

    palette = plt.cm.tab20.colors
    cores   = {p: palette[i % 20] for i, p in enumerate(produtos)}

    # ── Resultado compartilhado entre threads ────────────────────────────────
    resultado = {"done": False, "error": None,
                 "seqA": None, "seqB": None,
                 "assignA": None, "assignB": None,
                 "makespan_val": None, "cargas_b": None}

    def rodar_solver():
        try:
            # SOLVER A
            mA = pl.LpProblem("A", pl.LpMinimize)
            xA = {(m,p): pl.LpVariable(f"x_{m}_{p}",0,None,pl.LpInteger)
                  for m in maquinas for p in produtos if (m,p) in tempo}
            yA = {(m,p): pl.LpVariable(f"y_{m}_{p}",cat="Binary")
                  for m in maquinas for p in produtos if (m,p) in tempo}
            z  = {m: pl.LpVariable(f"z_{m}",cat="Binary") for m in maquinas}
            T  = pl.LpVariable("T",0)
            CA = {m: pl.LpVariable(f"CA_{m}",0) for m in maquinas}
            mA += T+0.01*pl.lpSum(CA[m] for m in maquinas)-0.1*pl.lpSum(yA[(m,p)] for (m,p) in yA)
            for p in produtos:
                mA += pl.lpSum(xA[(m,p)] for m in maquinas if (m,p) in xA) == lote[p]
            for (m,p) in xA:
                mA += xA[(m,p)] <= lote[p]*yA[(m,p)]
                mA += xA[(m,p)] >= lote_minimo[p]*yA[(m,p)]
            for p in produtos:
                mA += pl.lpSum(yA[(m,p)] for m in maquinas if (m,p) in yA) <= max_maquinas[p]
            for m in maquinas:
                mA += pl.lpSum(yA[(m,p)] for p in produtos if (m,p) in yA) <= 1000*z[m]
            for m in maquinas:
                mA += CA[m] == delays[m]+pl.lpSum(tempo[(m,p)]*xA[(m,p)] for p in produtos if (m,p) in xA)
            M_BIG = sum(max((tempo.get((m,p),0)*lote[p]) for p in produtos) for m in maquinas)
            for m in maquinas:
                mA += CA[m] <= T+M_BIG*(1-z[m])
            mA.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=MILP_TIMEOUT))
            if pl.LpStatus[mA.status] != "Optimal":
                raise Exception(f"Solver Makespan: {pl.LpStatus[mA.status]}")

            carga = {m: delays[m] for m in maquinas}
            seqA, linhasA = [], []
            for (m,p),v in xA.items():
                val = v.value()
                if val and val > 0:
                    ini,fim = carga[m], carga[m]+tempo[(m,p)]*val
                    carga[m]=fim
                    seqA.append([m,p,ini,fim,int(round(val))])
                    linhasA.append([m,p,int(round(val))])
            resultado["seqA"]    = pd.DataFrame(seqA,    columns=["Maquina","Produto","Inicio","Fim","Unidades"])
            resultado["assignA"] = pd.DataFrame(linhasA, columns=["Maquina","Produto","Unidades"])
            resultado["makespan_val"] = T.value()

            # SOLVER B
            mB = pl.LpProblem("B", pl.LpMinimize)
            xB = {(m,p): pl.LpVariable(f"xB_{m}_{p}",0,None,pl.LpInteger)
                  for m in maquinas for p in produtos if (m,p) in tempo}
            yB = {(m,p): pl.LpVariable(f"yB_{m}_{p}",cat="Binary")
                  for m in maquinas for p in produtos if (m,p) in tempo}
            C   = {m: pl.LpVariable(f"C_{m}",0) for m in maquinas}
            media   = pl.lpSum(C[m] for m in maquinas)/len(maquinas)
            dev_pos = {m: pl.LpVariable(f"dpos_{m}",0) for m in maquinas}
            dev_neg = {m: pl.LpVariable(f"dneg_{m}",0) for m in maquinas}
            mB += pl.lpSum(dev_pos[m]+dev_neg[m] for m in maquinas)+0.1*pl.lpSum(C[m] for m in maquinas)
            for p in produtos:
                mB += pl.lpSum(xB[(m,p)] for m in maquinas if (m,p) in xB) == lote[p]
            for (m,p) in xB:
                mB += xB[(m,p)] <= lote[p]*yB[(m,p)]
                mB += xB[(m,p)] >= lote_minimo[p]*yB[(m,p)]
            for p in produtos:
                mB += pl.lpSum(yB[(m,p)] for m in maquinas if (m,p) in yB) <= max_maquinas[p]
            for m in maquinas:
                mB += C[m] == delays[m]+pl.lpSum(tempo[(m,p)]*xB[(m,p)] for p in produtos if (m,p) in xB)
            for m in maquinas:
                mB += C[m]-media == dev_pos[m]-dev_neg[m]
            mB.solve(pl.PULP_CBC_CMD(msg=False, timeLimit=MILP_TIMEOUT))
            if pl.LpStatus[mB.status] != "Optimal":
                raise Exception(f"Solver Balanceado: {pl.LpStatus[mB.status]}")

            carga = {m: delays[m] for m in maquinas}
            seqB, linhasB = [], []
            for (m,p),v in xB.items():
                val = v.value()
                if val and val > 0:
                    ini,fim = carga[m], carga[m]+tempo[(m,p)]*val
                    carga[m]=fim
                    seqB.append([m,p,ini,fim,int(round(val))])
                    linhasB.append([m,p,int(round(val))])
            resultado["seqB"]    = pd.DataFrame(seqB,    columns=["Maquina","Produto","Inicio","Fim","Unidades"])
            resultado["assignB"] = pd.DataFrame(linhasB, columns=["Maquina","Produto","Unidades"])
            resultado["cargas_b"] = {m: C[m].value() or 0 for m in maquinas}

        except Exception as e:
            resultado["error"] = str(e)
        finally:
            resultado["done"] = True

    # ── Inicia solver em thread separada ────────────────────────────────────
    t = threading.Thread(target=rodar_solver, daemon=True)
    t.start()

    # ── ANIMAÇÃO: construção visual do Gantt enquanto solver roda ───────────
    # Gera alocações aleatórias progressivas para simular o processo
    bg, fg, grid_c = "#0f1117", "#e8e8e8", "#2a2d3a"

    # Estimativa de carga máxima para o eixo X
    max_tempo_est = max(
        delays.get(m, 0) + sum(tempo.get((m, p), 0) * lote[p] for p in produtos)
        for m in maquinas
    ) if maquinas.size > 0 else 100

    frame = 0
    alocacoes_anim = []  # lista de [maquina, produto, inicio, fim, unidades]
    produtos_restantes = {p: lote[p] for p in produtos}
    cargas_anim = {m: delays.get(m, 0) for m in maquinas}

    fases = [
        "🔍 Analisando restrições do problema…",
        "🧮 Calculando variáveis de decisão…",
        "⚙️  Montando matriz de coeficientes…",
        "🔗 Aplicando restrições de lote mínimo…",
        "🌳 Explorando árvore branch-and-bound…",
        "✂️  Podando ramos infeasíveis…",
        "📐 Refinando solução LP relaxada…",
        "🎯 Convergindo para solução ótima…",
        "🔄 Verificando factibilidade…",
        "📊 Construindo alocação final…",
    ]
    fase_idx = 0
    fase_timer = time.time()

    while not resultado["done"]:
        frame += 1

        # Troca de fase a cada ~4s
        if time.time() - fase_timer > 4:
            fase_idx = (fase_idx + 1) % len(fases)
            fase_timer = time.time()

        # Adiciona 1-3 alocações aleatórias por frame
        prods_com_saldo = [p for p, q in produtos_restantes.items() if q > 0]
        n_add = min(random.randint(1, 3), len(prods_com_saldo))
        for _ in range(n_add):
            if not prods_com_saldo:
                break
            p = random.choice(prods_com_saldo)
            # Máquinas que têm tempo para este produto
            maq_validas = [m for m in maquinas if (m, p) in tempo]
            if not maq_validas:
                continue
            m = random.choice(maq_validas)
            unid = random.randint(
                max(1, lote_minimo.get(p, 1)),
                min(produtos_restantes[p], lote[p])
            )
            duracao = tempo[(m, p)] * unid
            ini = cargas_anim[m]
            fim = ini + duracao
            cargas_anim[m] = fim
            alocacoes_anim.append([m, p, ini, fim, unid])
            produtos_restantes[p] = max(0, produtos_restantes[p] - unid)
            prods_com_saldo = [p for p, q in produtos_restantes.items() if q > 0]

        # Se todos alocados, reinicia animação
        if not prods_com_saldo and alocacoes_anim:
            time.sleep(0.8)
            alocacoes_anim = []
            produtos_restantes = {p: lote[p] for p in produtos}
            cargas_anim = {m: delays.get(m, 0) for m in maquinas}

        # Desenha Gantt animado
        seq_anim = pd.DataFrame(alocacoes_anim, columns=["Maquina","Produto","Inicio","Fim","Unidades"]) \
            if alocacoes_anim else pd.DataFrame(columns=["Maquina","Produto","Inicio","Fim","Unidades"])

        fig, ax = plt.subplots(figsize=(18, max(5, len(maquinas) * 0.55)))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_xlim(0, max_tempo_est * 1.05)

        for i, m in enumerate(maquinas):
            d = delays.get(m, 0)
            if d > 0:
                ax.barh(i, d, color="#333", height=0.6, left=0)
            bloco = seq_anim[seq_anim["Maquina"] == m]
            for j, (_, r) in enumerate(bloco.iterrows()):
                dur = r["Fim"] - r["Inicio"]
                prod_idx = produtos.index(r["Produto"]) if r["Produto"] in produtos else 0
                cor = plt.cm.tab20.colors[prod_idx % 20]
                # Último bloco adicionado pisca (borda branca)
                is_last = (j == len(bloco) - 1) and (frame % 2 == 0)
                ax.barh(i, dur, left=r["Inicio"], height=0.6,
                        color=cor, edgecolor="#ffffff" if is_last else bg,
                        linewidth=2 if is_last else 0.4, alpha=0.9)
                if dur > max_tempo_est * 0.03:
                    ax.text((r["Inicio"]+r["Fim"])/2, i,
                            f"{r['Produto']}\n({int(r['Unidades'])})",
                            ha="center", va="center", fontsize=6.5,
                            color="white", fontweight="bold")

        ax.set_yticks(range(len(maquinas)))
        ax.set_yticklabels(maquinas, fontsize=8, color=fg)
        ax.set_xlabel("Tempo", color=fg, fontsize=9)
        pct_alocado = int(100 * sum(lote[p]-produtos_restantes[p] for p in produtos) / max(1, sum(lote.values())))
        ax.set_title(f"🔄 Explorando soluções… {pct_alocado}% do lote alocado nesta tentativa",
                     color="#00d4aa", fontsize=11, fontweight="bold", pad=12)
        ax.tick_params(colors=fg)
        ax.spines[:].set_color(grid_c)
        ax.xaxis.grid(True, color=grid_c, linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)
        legend = [mpatches.Patch(color=plt.cm.tab20.colors[i%20], label=p)
                  for i, p in enumerate(produtos)]
        ax.legend(handles=legend, loc="lower right", fontsize=7,
                  facecolor=bg, edgecolor=grid_c, labelcolor=fg)
        fig.tight_layout()

        gantt_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)

        status_placeholder.markdown(
            f"<div class='phase-badge'>{fases[fase_idx]}</div>",
            unsafe_allow_html=True
        )
        progresso.progress(min(0.9, 0.1 + frame * 0.008))
        time.sleep(0.6)

    if resultado["error"]:
        raise Exception(resultado["error"])

    return (resultado["assignA"], resultado["seqA"],
            resultado["assignB"], resultado["seqB"],
            descricao, resultado["makespan_val"], resultado["cargas_b"])


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 Alocação Ótima")
    st.markdown("---")
    st.markdown("""
**Como usar:**
1. Faça o upload do arquivo Excel
2. Clique em **Rodar Otimização**
3. Acompanhe o Gantt sendo construído ao vivo
4. Veja o resultado final e baixe os Excel

**Formato esperado:**
- Linha 1: cabeçalhos + nomes das máquinas (G→AJ)
- Linha 2: delays por máquina
- Linha 3+: produtos com quantidades e tempos
""")
    st.markdown("---")
    st.markdown("<small style='color:#555'>V8.7 · MILP via CBC · Animação ao vivo</small>",
                unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏭 Sistema de Alocação Ótima de Produção")
st.markdown("Faça upload do Excel de planejamento. O Gantt será construído **ao vivo** enquanto o solver otimiza.")
st.markdown("---")

uploaded = st.file_uploader(
    "📂 Selecione o arquivo Excel (.xlsx ou .xlsm)",
    type=["xlsx","xlsm"], label_visibility="collapsed"
)

if uploaded:
    st.success(f"✔ Arquivo carregado: **{uploaded.name}**")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        rodar = st.button("⚡ Rodar Otimização", use_container_width=True, type="primary")

    if rodar:
        progresso   = st.progress(0.05)
        status_txt  = st.empty()
        gantt_live  = st.empty()

        try:
            status_txt.markdown("⏳ Lendo e validando arquivo…")
            conteudo = uploaded.read()
            df_banco, lote, max_maq, lote_min, delays, maquinas = carregar_arquivo(conteudo)
            validar_entradas(df_banco, lote, lote_min)

            status_txt.markdown(
                f"<div class='phase-badge'>🚀 Iniciando otimização com {len(lote)} produtos e {len(maquinas)} máquinas…</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.5)

            assignA, seqA, assignB, seqB, desc, makespan_val, cargas_b = solver_com_animacao(
                df_banco, lote, max_maq, lote_min, delays,
                gantt_live, status_txt, progresso
            )

            # Limpa animação
            gantt_live.empty()
            status_txt.empty()
            progresso.progress(0.95)

            assignA = expandir(assignA, maquinas, lote.keys(), desc)
            assignB = expandir(assignB, maquinas, lote.keys(), desc)

            progresso.progress(1.0)
            st.markdown("<span class='status-ok'>✔ Otimização concluída com sucesso!</span>",
                        unsafe_allow_html=True)
            st.markdown("---")

            # Métricas
            n_maq_ativas = len(seqA["Maquina"].unique())
            carga_max_b  = max(cargas_b.values()) if cargas_b else 0
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val in [
                (c1, "Produtos",         len(lote)),
                (c2, "Máquinas Ativas",  n_maq_ativas),
                (c3, "Makespan",         f"{makespan_val:.1f}"),
                (c4, "Carga Máx. (Bal.)",f"{carga_max_b:.1f}"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            tab1, tab2 = st.tabs(["📊 Makespan", "⚖️ Balanceado"])
            with tab1:
                st.markdown("### Gráfico de Gantt — Makespan mínimo")
                fig_a = gerar_gantt(seqA, maquinas, delays, lote, "Makespan — Solução Ótima")
                st.pyplot(fig_a, use_container_width=True)
                st.download_button("⬇ Baixar Excel — Makespan", data=excel_bytes(assignA),
                    file_name="resultado_makespan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with tab2:
                st.markdown("### Gráfico de Gantt — Carga Balanceada")
                fig_b = gerar_gantt(seqB, maquinas, delays, lote, "Balanceado — Solução Ótima")
                st.pyplot(fig_b, use_container_width=True)
                st.download_button("⬇ Baixar Excel — Balanceado", data=excel_bytes(assignB),
                    file_name="resultado_balanceado.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            progresso.empty()
            status_txt.empty()
            gantt_live.empty()
            st.error(f"❌ Erro: {e}")
else:
    st.info("👆 Faça o upload do arquivo Excel para começar.")


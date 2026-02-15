import math
from typing import Tuple, Optional
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =============================
# Константы (№1140 + ограничения демонстратора)
# =============================
R_NORM = 1e-6   # год^-1 (норматив ИПР)
P_E_MAX = 0.999 # по формуле (6) Методики №1140
K_STD = 0.8     # Kобн,i, KСОУЭ,i, KПДЗ,i по №1140: либо 0.8, либо 0
K_MAX = 0.99    # демонстратор: не допускаем 1.00
K_AP_MAX = 0.9  # ограничение интерфейса для Kап

# =============================
# Вспомогательные функции
# =============================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))
def force_rerun():
    try:
        st.rerun()
    except Exception:
        # для старых версий streamlit
        st.experimental_rerun()

def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def ensure_unique_positive_int_ids(df: pd.DataFrame, col: str, start_from: int = 1) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        df.insert(0, col, np.arange(start_from, start_from + len(df), dtype=int))
        return df

    ids = pd.to_numeric(df[col], errors="coerce")
    used = set()
    next_id = start_from
    if ids.notna().any():
        next_id = int(ids.dropna().max()) + 1

    new_ids = []
    for v in ids.to_list():
        if pd.isna(v):
            while next_id in used:
                next_id += 1
            new_ids.append(next_id)
            used.add(next_id)
            next_id += 1
            continue

        iv = int(v)
        if iv <= 0 or iv in used:
            while next_id in used:
                next_id += 1
            new_ids.append(next_id)
            used.add(next_id)
            next_id += 1
        else:
            new_ids.append(iv)
            used.add(iv)

    df[col] = new_ids
    return df

def next_int_id(series: pd.Series, start_from: int = 1) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return start_from
    m = int(s.max())
    return max(start_from, m + 1)

# =============================
# Формулы Методики №1140 (в части ИПР)
# =============================
def p_presence(t_pr_hours: float) -> float:
    # Pпр,i = tпр,i / 24
    return clamp(safe_float(t_pr_hours) / 24.0, 0.0, 1.0)

def p_evac_1140_piecewise(t_p: float, t_bl: float, t_ne: float, t_ck: float) -> float:
    """
    Pэ,i,j по формуле (6) №1140 (кусочная).
    Допускает промежуточные значения при:
      t_p < 0.8*t_bl < t_p + t_ne, при t_ck <= 6.
    """
    t_p = safe_float(t_p)
    t_bl = safe_float(t_bl)
    t_ne = max(1e-9, safe_float(t_ne))  # защита от деления на 0
    t_ck = safe_float(t_ck)

    border = 0.8 * t_bl

    if t_ck > 6.0:
        return 0.0

    if (t_p < border) and (border < (t_p + t_ne)):
        val = P_E_MAX * ((border - t_p) / t_ne)
        return clamp(val, 0.0, P_E_MAX)

    if (t_p + t_ne) <= border:
        return P_E_MAX

    return 0.0

def p_evac_binary(t_p: float, t_bl: float, t_ne: float, t_ck: float) -> float:
    """
    Упрощение: Pэ ∈ {0.999; 0}
    """
    t_p = safe_float(t_p)
    t_bl = safe_float(t_bl)
    t_ne = safe_float(t_ne)
    t_ck = safe_float(t_ck)
    border = 0.8 * t_bl
    if (t_ck <= 6.0) and ((t_p + t_ne) <= border):
        return P_E_MAX
    return 0.0

def k_pz(k_obn: float, k_soue: float, k_pdz: float) -> float:
    """
    Kп.з,i = 1 - (1 - Kобн,i*KСОУЭ,i) * (1 - Kобн,i*KПДЗ,i)  (формула (7) №1140)
    """
    k_obn = clamp(safe_float(k_obn), 0.0, K_MAX)
    k_soue = clamp(safe_float(k_soue), 0.0, K_MAX)
    k_pdz = clamp(safe_float(k_pdz), 0.0, K_MAX)
    val = 1.0 - (1.0 - k_obn * k_soue) * (1.0 - k_obn * k_pdz)
    return clamp(val, 0.0, 1.0)

def r_ij(q_n: float, k_ap: float, p_pr: float, p_e: float, k_pz_i: float) -> float:
    """
    R_i,j = Q_n,i*(1 - Kап,i)*Pпр,i*(1 - Pэ,i,j)*(1 - Kп.з,i) (формула (4) №1140)
    """
    q_n = max(0.0, safe_float(q_n))
    k_ap = clamp(safe_float(k_ap), 0.0, K_AP_MAX)
    p_pr = clamp(safe_float(p_pr), 0.0, 1.0)
    p_e = clamp(safe_float(p_e), 0.0, P_E_MAX)
    k_pz_i = clamp(safe_float(k_pz_i), 0.0, 1.0)
    val = q_n * (1.0 - k_ap) * p_pr * (1.0 - p_e) * (1.0 - k_pz_i)
    return max(0.0, float(val))

# =============================
# Графика: ЛОГ-шкала слева + "как на рисунке" справа (три столбца до Rнорм)
# =============================
def compare_risk_component_html(r_trad: float, r_iot: float, r_norm: float) -> str:
    r_trad = float(r_trad) if (r_trad and r_trad > 0) else 0.0
    r_iot = float(r_iot) if (r_iot and r_iot > 0) else 0.0
    r_norm = float(r_norm) if (r_norm and r_norm > 0) else 1e-12

    # ---------- LEFT: log scale ----------
    values = [v for v in [r_trad, r_iot, r_norm] if v > 0]
    log_min = math.floor(math.log10(min(values)))
    log_max = math.ceil(math.log10(max(values)))

    # делаем шкалу не слишком "узкой"
    if (log_max - log_min) < 6:
        center = (log_max + log_min) / 2.0
        log_min = math.floor(center - 3)
        log_max = math.ceil(center + 3)
    if log_max == log_min:
        log_max = log_min + 1

    def y_log_percent(v: float) -> float:
        v = max(v, 1e-12)
        exp = math.log10(v)
        exp = clamp(exp, log_min, log_max)
        t = (log_max - exp) / (log_max - log_min)
        return clamp(100.0 * t, 0.0, 100.0)

    ticks = list(range(int(log_min), int(log_max) + 1))
    ticks_parts = []
    for p in ticks:
        y = y_log_percent(10.0 ** p)
        ticks_parts.append(
            f"""
<div class="tick" style="top:{y:.6f}%">
  <div class="tick-line"></div>
  <div class="tick-label">10^{p}</div>
</div>
""".strip()
        )
    ticks_html = "\n".join(ticks_parts)

    # Маркеры с управляемым сдвигом подписи:
    #  - Rнорм: сверху линии (сильнее вверх)
    #  - Rтрадиц: сверху линии (чуть вверх)
    #  - RIoT: снизу линии
    def marker(y: float, label: str, cls: str, text_shift_px: int) -> str:
        return f"""
<div class="marker {cls}" style="top:{y:.6f}%">
  <div class="marker-line"></div>
  <div class="marker-text" style="transform: translateY({text_shift_px}px);">{label}</div>
</div>
""".strip()

    markers_parts = []
    markers_parts.append(marker(y_log_percent(r_norm), f"Rнорм = {r_norm:.1e}", "m-norm", -26))
    if r_trad > 0:
        markers_parts.append(marker(y_log_percent(r_trad), f"Rтрадиц = {r_trad:.2e}", "m-trad", -16))
    if r_iot > 0:
        markers_parts.append(marker(y_log_percent(r_iot), f"RIoT = {r_iot:.2e}", "m-iot", 10))
    markers_html = "\n".join(markers_parts)

    # ---------- RIGHT: percent of Rnorm ----------
    pct_norm = 100.0
    pct_trad = (r_trad / r_norm) * 100.0 if r_norm > 0 else 0.0
    pct_iot = (r_iot / r_norm) * 100.0 if r_norm > 0 else 0.0

    pct_max = max(pct_norm, pct_trad, pct_iot, 1.0)

    def h_pct(p: float) -> float:
        return clamp(100.0 * (p / pct_max), 0.0, 100.0)

    h_norm = h_pct(pct_norm)
    h_trad = h_pct(pct_trad)
    h_iot = h_pct(pct_iot)

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --bg: rgba(255,255,255,0.03);
    --bd: rgba(255,255,255,0.14);
    --txt: rgba(255,255,255,0.90);
    --muted: rgba(255,255,255,0.70);
    --grid: rgba(255,255,255,0.14);
    --norm: #ff4d4d;
    --trad: #7aa6ff;
    --iot:  #7dffb3;
  }}
  body {{ margin:0; background:transparent; font-family: system-ui, Segoe UI, Arial; color:var(--txt); }}
  .wrap {{ display:grid; grid-template-columns: 260px 1fr; gap:16px; padding:8px 2px; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:14px; }}
  .title {{ font-size:12px; color:var(--muted); margin:0 0 10px 0; }}

  /* left */
  .scale-box {{ position:relative; height:330px; border-radius:12px; background:rgba(0,0,0,0.15); border:1px solid rgba(255,255,255,0.10); overflow:hidden; }}
  .axis {{ position:absolute; left:16px; top:10px; bottom:10px; width:1px; background:rgba(255,255,255,0.18); }}
  .tick {{ position:absolute; left:18px; right:10px; transform:translateY(-50%); display:flex; gap:8px; align-items:center; }}
  .tick-line {{ height:1px; flex:1; background:var(--grid); }}
  .tick-label {{ font-size:11px; color:var(--muted); white-space:nowrap; }}

  .marker {{ position:absolute; left:10px; right:10px; transform:translateY(-50%); pointer-events:none; }}
  .marker-line {{ height:4px; border-radius:999px; }}
  .marker-text {{
    font-size:11px;
    font-weight:750;
    line-height:1.1;
    white-space:nowrap;
    text-shadow: 0 1px 10px rgba(0,0,0,0.65);
  }}

  .m-norm .marker-line {{ background:var(--norm); }}
  .m-norm .marker-text {{ color:var(--norm); }}
  .m-trad .marker-line {{ background:var(--trad); }}
  .m-trad .marker-text {{ color:var(--trad); }}
  .m-iot .marker-line {{ background:var(--iot); }}
  .m-iot .marker-text {{ color:var(--iot); }}

  /* right percent bars */
  .legend {{ font-size:12px; color:var(--muted); margin-bottom:10px; line-height:1.35; }}
  .bars-box {{ position:relative; height:330px; border-radius:12px; background:rgba(0,0,0,0.15); border:1px solid rgba(255,255,255,0.10); overflow:hidden; padding:12px; }}
  .note-top {{ font-size:11px; color:var(--muted); margin-bottom:6px; }}

  .bars {{ position:absolute; left:12px; right:12px; top:40px; bottom:16px; display:flex; gap:18px; align-items:stretch; }}
  .col {{ flex:1; display:flex; flex-direction:column; height:100%; }}
  .barwrap {{ flex:1; display:flex; align-items:flex-end; justify-content:center; }}
  .bar {{ width:70%; border-radius:12px 12px 8px 8px; box-shadow:0 8px 30px rgba(0,0,0,0.25); }}
  .b-norm {{ background:var(--norm); }}
  .b-trad {{ background:var(--trad); }}
  .b-iot  {{ background:var(--iot); }}

  .lbl {{ margin-top:10px; text-align:center; }}
  .name {{ font-weight:800; font-size:12px; }}
  .val {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .pct {{ font-size:12px; margin-top:2px; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">Сравнительная шкала ИПР (R)</div>
      <div class="scale-box">
        <div class="axis"></div>
        {ticks_html}
        {markers_html}
      </div>
    </div>

    <div class="card">
      <div class="legend">Сравнение в <b>% от Rнорм</b>.</div>
      <div class="bars-box">
        <div class="note-top">Максимум шкалы: {pct_max:.1f}% от Rнорм</div>
        <div class="bars">
          <div class="col">
            <div class="barwrap"><div class="bar b-norm" style="height:{h_norm:.6f}%;"></div></div>
            <div class="lbl">
              <div class="name" style="color:var(--norm)">Rнорм</div>
              <div class="val">{r_norm:.1e} год^-1</div>
              <div class="pct">100%</div>
            </div>
          </div>

          <div class="col">
            <div class="barwrap"><div class="bar b-trad" style="height:{h_trad:.6f}%;"></div></div>
            <div class="lbl">
              <div class="name" style="color:var(--trad)">Rтрадиц</div>
              <div class="val">{r_trad:.6g} год^-1</div>
              <div class="pct">{pct_trad:.1f}%</div>
            </div>
          </div>

          <div class="col">
            <div class="barwrap"><div class="bar b-iot" style="height:{h_iot:.6f}%;"></div></div>
            <div class="lbl">
              <div class="name" style="color:var(--iot)">RIoT</div>
              <div class="val">{r_iot:.6g} год^-1</div>
              <div class="pct">{pct_iot:.1f}%</div>
            </div>
          </div>

        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    return dedent(html)

# =============================
# Streamlit UI
# =============================
st.set_page_config(
    page_title="ИПР: традиционно vs IoT-СОУЭ (№1140 + расширение через Kп.з)",
    layout="wide",
)

st.title("Расчет индивидуального пожарного риска по Приказу МЧС России №1140: традиционный расчёт и вариант с IoT-СОУЭ")
st.caption(
    ""
)

# -----------------------------
# Данные по умолчанию
# -----------------------------
default_scenarios = pd.DataFrame([
    {
        "Сценарий i": 1,
        "Q_n,i (год^-1)": 4.0e-2,
        "t_пр,i (ч/сут)": 12.0,
        "t_бл,i (мин)": 12.0,
        "K_ап,i (0..0.9)": 0.9,
        "ПС соответствует/не требуется/подтверждена? (Kобн=0.8)": True,
        "СОУЭ соответствует/не требуется/подтверждена? (KСОУЭ=0.8)": True,
        "ПДЗ соответствует/не требуется/подтверждена? (KПДЗ=0.8)": True,
    },
])

default_groups = pd.DataFrame([
    {
        "ID": 1,
        "Сценарий i": 1,
        "Группа j": "Основной контингент",
        "t_p,i,j (мин)": 6.0,
        "t_н.э,i,j (мин)": 1.5,
        "t_ск,i,j (мин)": 1.0,
    },
    {
        "ID": 2,
        "Сценарий i": 1,
        "Группа j": "Маломобильные",
        "t_p,i,j (мин)": 7.0,
        "t_н.э,i,j (мин)": 2.0,
        "t_ск,i,j (мин)": 2.5,
    },
])

if "df_scen" not in st.session_state:
    st.session_state.df_scen = default_scenarios.copy()
if "df_grp" not in st.session_state:
    st.session_state.df_grp = default_groups.copy()

st.session_state.df_grp = ensure_unique_positive_int_ids(st.session_state.df_grp, "ID", start_from=1)

# Чтобы выбирать группу для "бегунков" и диагностики
if "selected_group_id" not in st.session_state:
    st.session_state.selected_group_id = int(st.session_state.df_grp["ID"].iloc[0]) if len(st.session_state.df_grp) else 1

# -----------------------------
# Sidebar: управление (БЕЗ вывода "прозрачного расчёта")
# -----------------------------
with st.sidebar:
    st.header("Управление (ползунки)")

    st.markdown("Настройте параметры адаптивности IoT-СОУЭ и надёжности её модулей, а также временные характеристики сценариев и групп. Результаты расчёта ИПР будут отображены на основном полотне справа.")
    st.caption("")


    st.subheader("K_IoT (0…0.99) — шкалы адаптивности")
    k_sensors = st.slider("Сенсоры в ключевых точках", 0.0, K_MAX, 0.90, 0.01)
    k_routing = st.slider("Адаптация маршрутов", 0.0, K_MAX, 0.85, 0.01)
    k_automation = st.slider("Автоматизация решений", 0.0, K_MAX, 0.85, 0.01)
    reaction_sec = st.slider("Время реакции системы (сек)", 0, 60, 10, 1)
    k_reaction = clamp(K_MAX * (1.0 - reaction_sec / 60.0), 0.0, K_MAX)

    st.subheader("Надёжности модулей ioT-СОУЭ (0…0.99)")
    k_det = st.slider("Надёжность обнаружения/локализации", 0.0, K_MAX, 0.95, 0.01)
    k_comm_main = st.slider("Надёжность основного канала связи", 0.0, K_MAX, 0.92, 0.01)
    k_comm_backup = st.slider("Надёжность резервного канала связи", 0.0, K_MAX, 0.90, 0.01)
    k_logic = st.slider("Надёжность логики/маршрутизации", 0.0, K_MAX, 0.93, 0.01)
    k_alert = st.slider("Надёжность оповещения/исполнения", 0.0, K_MAX, 0.94, 0.01)
    k_power = st.slider("Надёжность питания/резерва", 0.0, K_MAX, 0.95, 0.01)
    p_cyber = st.slider("Вероятность киберинцидента", 0.0, 0.50, 0.05, 0.01)

    alpha = st.slider("α (0.10…0.30)", 0.10, 0.30, 0.20, 0.01)

    st.subheader("Бегунки времён (без Pathfinder)")
    df_grp_sel = st.session_state.df_grp.copy()
    if len(df_grp_sel) > 0:
        df_grp_sel["label"] = df_grp_sel.apply(
            lambda r: f"ID {int(r['ID'])} | сценарий {int(r['Сценарий i'])} | {r.get('Группа j','')}",
            axis=1
        )
        labels = df_grp_sel["label"].to_list()
        sel_label = st.selectbox("Выбери группу для настройки", labels, index=0)

        sel_id = int(df_grp_sel.loc[df_grp_sel["label"] == sel_label, "ID"].iloc[0])
        st.session_state.selected_group_id = sel_id

        grp_df = st.session_state.df_grp.copy()
        row_idx = grp_df.index[grp_df["ID"] == sel_id][0]
        scen_id = int(grp_df.at[row_idx, "Сценарий i"])

        scen_df = st.session_state.df_scen.copy()
        scen_df["Сценарий i"] = pd.to_numeric(scen_df["Сценарий i"], errors="coerce").fillna(0).astype(int)
        scen_match = scen_df.index[scen_df["Сценарий i"] == scen_id]

        if len(scen_match) > 0:
            scen_idx = scen_match[0]
            t_bl_cur = safe_float(scen_df.at[scen_idx, "t_бл,i (мин)"], 12.0)

            t_bl_new = st.slider("t_бл,i (мин)", 0.5, 180.0, float(t_bl_cur), 0.5)
            t_p_new = st.slider("t_p,i,j (мин)", 0.0, 180.0, float(grp_df.at[row_idx, "t_p,i,j (мин)"]), 0.5)
            t_ne_new = st.slider("t_н.э,i,j (мин)", 0.0, 60.0, float(grp_df.at[row_idx, "t_н.э,i,j (мин)"]), 0.5)
            t_ck_new = st.slider("t_ск,i,j (мин)", 0.0, 30.0, float(grp_df.at[row_idx, "t_ск,i,j (мин)"]), 0.5)

            scen_df.at[scen_idx, "t_бл,i (мин)"] = t_bl_new
            grp_df.at[row_idx, "t_p,i,j (мин)"] = t_p_new
            grp_df.at[row_idx, "t_н.э,i,j (мин)"] = t_ne_new
            grp_df.at[row_idx, "t_ск,i,j (мин)"] = t_ck_new

            st.session_state.df_scen = scen_df
            st.session_state.df_grp = grp_df
        else:
            st.warning("Для выбранной группы не найден сценарий. Добавь/исправь 'Сценарий i' в таблице сценариев.")
    else:
        st.info("Нет групп. Добавь группу в таблице ниже.")

# -----------------------------
# Промежуточные вычисления IoT (чтобы вывести В ОСНОВНОМ ПОЛОТНЕ)
# -----------------------------
k_iot_score = float(np.mean([k_sensors, k_routing, k_automation, k_reaction]))
k_iot_score = clamp(k_iot_score, 0.0, K_MAX)

cyber_factor = clamp(1.0 - p_cyber, 0.0, K_MAX)
k_comm = 1.0 - (1.0 - k_comm_main) * (1.0 - k_comm_backup)
k_comm = clamp(k_comm, 0.0, K_MAX)

k_rel = k_det * k_comm * k_logic * k_alert * k_power * cyber_factor
k_rel = clamp(k_rel, 0.0, K_MAX)

k_iot_total = clamp(k_iot_score * k_rel, 0.0, K_MAX)
delta_k_soue = clamp(alpha * k_iot_total, 0.0, K_MAX)

# -----------------------------
# Функция расчёта
# -----------------------------
def compute_all(
    df_scen_in: pd.DataFrame,
    df_grp_in: pd.DataFrame,
    alpha: float,
    k_iot_total: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:

    df_scen = df_scen_in.copy()
    df_grp = df_grp_in.copy()

    df_scen["Сценарий i"] = pd.to_numeric(df_scen["Сценарий i"], errors="coerce").fillna(0).astype(int)
    df_grp["Сценарий i"] = pd.to_numeric(df_grp["Сценарий i"], errors="coerce").fillna(0).astype(int)

    if df_scen["Сценарий i"].duplicated().any():
        df_scen = df_scen.drop_duplicates(subset=["Сценарий i"], keep="first").copy()

    for c in ["Q_n,i (год^-1)", "t_пр,i (ч/сут)", "t_бл,i (мин)", "K_ап,i (0..0.9)"]:
        df_scen[c] = pd.to_numeric(df_scen[c], errors="coerce").fillna(0.0)

    for c in ["t_p,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)"]:
        df_grp[c] = pd.to_numeric(df_grp[c], errors="coerce").fillna(0.0)

    # 0 или 0.8 строго по №1140
    ps_ok = df_scen["ПС соответствует/не требуется/подтверждена? (Kобн=0.8)"].astype(bool)
    soue_ok = df_scen["СОУЭ соответствует/не требуется/подтверждена? (KСОУЭ=0.8)"].astype(bool)
    pdz_ok = df_scen["ПДЗ соответствует/не требуется/подтверждена? (KПДЗ=0.8)"].astype(bool)

    df_scen["K_обн,i"] = np.where(ps_ok, K_STD, 0.0)
    df_scen["K_СОУЭ,i (традиц)"] = np.where(soue_ok, K_STD, 0.0)
    df_scen["K_ПДЗ,i"] = np.where(pdz_ok, K_STD, 0.0)

    df_scen["P_пр,i"] = df_scen["t_пр,i (ч/сут)"].apply(p_presence)

    alpha = clamp(safe_float(alpha), 0.0, 1.0)
    k_iot_total = clamp(safe_float(k_iot_total), 0.0, K_MAX)

    # IoT: добавка к KСОУЭ (аддитивно), но не более 0.99
    df_scen["K_СОУЭ,i (IoT)"] = np.clip(
        df_scen["K_СОУЭ,i (традиц)"].astype(float) + alpha * k_iot_total,
        0.0,
        K_MAX
    )

    df_scen["K_п.з,i (традиц)"] = df_scen.apply(
        lambda r: k_pz(r["K_обн,i"], r["K_СОУЭ,i (традиц)"], r["K_ПДЗ,i"]),
        axis=1
    )
    df_scen["K_п.з,i (IoT)"] = df_scen.apply(
        lambda r: k_pz(r["K_обн,i"], r["K_СОУЭ,i (IoT)"], r["K_ПДЗ,i"]),
        axis=1
    )

    df_rows = df_grp.merge(
        df_scen[[
            "Сценарий i", "Q_n,i (год^-1)", "K_ап,i (0..0.9)", "t_бл,i (мин)",
            "P_пр,i", "K_п.з,i (традиц)", "K_п.з,i (IoT)"
        ]],
        on="Сценарий i",
        how="left"
    )

    missing = df_rows["t_бл,i (мин)"].isna() | df_rows["Q_n,i (год^-1)"].isna() | df_rows["P_пр,i"].isna()
    if missing.any():
        df_rows = df_rows.loc[~missing].copy()

    # Pэ — всегда по формуле №1140 (возможны промежуточные значения)
    df_rows["P_э,i,j"] = df_rows.apply(
        lambda r: p_evac_1140_piecewise(
            r["t_p,i,j (мин)"], r["t_бл,i (мин)"], r["t_н.э,i,j (мин)"], r["t_ск,i,j (мин)"]
        ),
        axis=1
    )


    df_rows["R_i,j (традиц)"] = df_rows.apply(
        lambda r: r_ij(
            r["Q_n,i (год^-1)"], r["K_ап,i (0..0.9)"], r["P_пр,i"], r["P_э,i,j"], r["K_п.з,i (традиц)"]
        ),
        axis=1
    )
    df_rows["R_i,j (IoT)"] = df_rows.apply(
        lambda r: r_ij(
            r["Q_n,i (год^-1)"], r["K_ап,i (0..0.9)"], r["P_пр,i"], r["P_э,i,j"], r["K_п.з,i (IoT)"]
        ),
        axis=1
    )

    agg = df_rows.groupby("Сценарий i", as_index=False).agg(
        **{
            "R_i (традиц) = max_j": ("R_i,j (традиц)", "max"),
            "R_i (IoT) = max_j": ("R_i,j (IoT)", "max"),
        }
    )

    r_total_trad = float(agg["R_i (традиц) = max_j"].max()) if len(agg) else 0.0
    r_total_iot = float(agg["R_i (IoT) = max_j"].max()) if len(agg) else 0.0

    agg["Проходит? (традиц)"] = agg["R_i (традиц) = max_j"].apply(lambda x: "Да" if x <= R_NORM else "Нет")
    agg["Проходит? (IoT)"] = agg["R_i (IoT) = max_j"].apply(lambda x: "Да" if x <= R_NORM else "Нет")

    total_row = pd.DataFrame([{
        "Сценарий i": "ИТОГО (R = max_i)",
        "R_i (традиц) = max_j": r_total_trad,
        "R_i (IoT) = max_j": r_total_iot,
        "Проходит? (традиц)": "Да" if r_total_trad <= R_NORM else "Нет",
        "Проходит? (IoT)": "Да" if r_total_iot <= R_NORM else "Нет",
    }])
    agg_out = pd.concat([agg, total_row], ignore_index=True)

    return df_scen, df_rows, agg_out, r_total_trad, r_total_iot

# -----------------------------
# Таблицы ввода + кнопки добавления строк
# -----------------------------
st.subheader("Исходные данные (редактируемые)")

tab1, tab2 = st.tabs(["Сценарии i", "Группы j (по сценариям)"])

# ==========================
# TAB1: Сценарии
# ==========================
with tab1:
    st.caption(
        "K_обн,i, K_СОУЭ,i, K_ПДЗ,i задаются строго по №1140: либо 0.8 (выполняется хотя бы одно условие), либо 0."
    )

    # --- Текущая таблица сценариев ---
    df_scen_raw = st.session_state.df_scen.copy()
    if "Сценарий i" not in df_scen_raw.columns:
        df_scen_raw["Сценарий i"] = np.arange(1, len(df_scen_raw) + 1, dtype=int)

    df_scen_raw["Сценарий i"] = pd.to_numeric(df_scen_raw["Сценарий i"], errors="coerce").fillna(0).astype(int)
    df_scen_raw = df_scen_raw.loc[df_scen_raw["Сценарий i"] > 0].copy()

    scen_list = sorted(df_scen_raw["Сценарий i"].unique().tolist())
    if len(scen_list) == 0:
        scen_list = [1]

    # --- Кнопки управления (ДО data_editor) ---
    c1, c2, c3 = st.columns([1.2, 1.8, 2.0])

    with c1:
        if st.button("➕ Добавить сценарий", use_container_width=True):
            df = st.session_state.df_scen.copy()
            if len(df) == 0:
                next_i = 1
            else:
                df["Сценарий i"] = pd.to_numeric(df["Сценарий i"], errors="coerce").fillna(0).astype(int)
                next_i = int(df["Сценарий i"].max()) + 1

            new_row = {
                "Сценарий i": next_i,
                "Q_n,i (год^-1)": 4.0e-2,
                "t_пр,i (ч/сут)": 12.0,
                "t_бл,i (мин)": 12.0,
                "K_ап,i (0..0.9)": 0.9,
                "ПС соответствует/не требуется/подтверждена? (Kобн=0.8)": True,
                "СОУЭ соответствует/не требуется/подтверждена? (KСОУЭ=0.8)": True,
                "ПДЗ соответствует/не требуется/подтверждена? (KПДЗ=0.8)": True,
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df_scen = df
            force_rerun()

    with c2:
        scen_del = st.selectbox("Удалить сценарий", scen_list, key="scen_del_select")
        if st.button("🗑️ Удалить выбранный сценарий", use_container_width=True):
            # удаляем сценарий
            df_s = st.session_state.df_scen.copy()
            df_s["Сценарий i"] = pd.to_numeric(df_s["Сценарий i"], errors="coerce").fillna(0).astype(int)
            df_s = df_s.loc[df_s["Сценарий i"] != int(scen_del)].copy()
            st.session_state.df_scen = df_s

            # удаляем все группы, которые ссылались на этот сценарий
            df_g = st.session_state.df_grp.copy()
            df_g["Сценарий i"] = pd.to_numeric(df_g["Сценарий i"], errors="coerce").fillna(0).astype(int)
            df_g = df_g.loc[df_g["Сценарий i"] != int(scen_del)].copy()
            st.session_state.df_grp = df_g

            force_rerun()

    with c3:
        st.info(
            "Удаление сценария удалает связанные группы."
        )

    # --- Preview столбцы (read-only) ---
    df_scen_raw = st.session_state.df_scen.copy()
    df_scen_raw["Сценарий i"] = pd.to_numeric(df_scen_raw["Сценарий i"], errors="coerce").fillna(0).astype(int)
    df_scen_raw = df_scen_raw.loc[df_scen_raw["Сценарий i"] > 0].copy()

    # добавляем расчетные столбцы
    df_scen_preview = df_scen_raw.copy()
    df_scen_preview["K_обн,i (расч.)"] = df_scen_preview["ПС соответствует/не требуется/подтверждена? (Kобн=0.8)"].astype(bool).map(lambda x: K_STD if x else 0.0)
    df_scen_preview["K_СОУЭ,i (расч., традиц)"] = df_scen_preview["СОУЭ соответствует/не требуется/подтверждена? (KСОУЭ=0.8)"].astype(bool).map(lambda x: K_STD if x else 0.0)
    df_scen_preview["K_ПДЗ,i (расч.)"] = df_scen_preview["ПДЗ соответствует/не требуется/подтверждена? (KПДЗ=0.8)"].astype(bool).map(lambda x: K_STD if x else 0.0)

    df_scen_edit = st.data_editor(
        df_scen_preview,
        num_rows="dynamic",
        use_container_width=True,
        disabled=["K_обн,i (расч.)", "K_СОУЭ,i (расч., традиц)", "K_ПДЗ,i (расч.)"],
        column_config={
            "Сценарий i": st.column_config.NumberColumn(min_value=1, step=1),
            "Q_n,i (год^-1)": st.column_config.NumberColumn(format="%.6g"),
            "t_пр,i (ч/сут)": st.column_config.NumberColumn(format="%.3f"),
            "t_бл,i (мин)": st.column_config.NumberColumn(format="%.3f"),
            "K_ап,i (0..0.9)": st.column_config.NumberColumn(format="%.3f"),
            "ПС соответствует/не требуется/подтверждена? (Kобн=0.8)": st.column_config.CheckboxColumn(),
            "СОУЭ соответствует/не требуется/подтверждена? (KСОУЭ=0.8)": st.column_config.CheckboxColumn(),
            "ПДЗ соответствует/не требуется/подтверждена? (KПДЗ=0.8)": st.column_config.CheckboxColumn(),
        },
        key="editor_scenarios"
    )

    # сохраняем без preview-столбцов
    drop_cols = ["K_обн,i (расч.)", "K_СОУЭ,i (расч., традиц)", "K_ПДЗ,i (расч.)"]
    df_scen_store = df_scen_edit.drop(columns=[c for c in drop_cols if c in df_scen_edit.columns], errors="ignore").copy()
    df_scen_store["Сценарий i"] = pd.to_numeric(df_scen_store["Сценарий i"], errors="coerce").fillna(0).astype(int)
    df_scen_store = df_scen_store.loc[df_scen_store["Сценарий i"] > 0].drop_duplicates(subset=["Сценарий i"], keep="first").copy()
    st.session_state.df_scen = df_scen_store


# ==========================
# TAB2: Группы
# ==========================
with tab2:
    st.caption(
        "t_p,i,j — расчётное время эвакуации; t_н.э,i,j — время начала эвакуации; "
        "t_ск,i,j — время существования скоплений. Эти величины влияют на P_э,i,j по формуле (6) №1140."
    )

    # актуальный список сценариев для привязки групп
    df_scen_for_groups = st.session_state.df_scen.copy()
    df_scen_for_groups["Сценарий i"] = pd.to_numeric(df_scen_for_groups["Сценарий i"], errors="coerce").fillna(0).astype(int)
    scen_list2 = sorted(df_scen_for_groups.loc[df_scen_for_groups["Сценарий i"] > 0, "Сценарий i"].unique().tolist())
    if len(scen_list2) == 0:
        scen_list2 = [1]

    # таблица групп
    df_grp_raw = st.session_state.df_grp.copy()
    df_grp_raw = ensure_unique_positive_int_ids(df_grp_raw, "ID", start_from=1)
    df_grp_raw["Сценарий i"] = pd.to_numeric(df_grp_raw["Сценарий i"], errors="coerce").fillna(scen_list2[0]).astype(int)
    st.session_state.df_grp = df_grp_raw

    id_list = sorted(pd.to_numeric(df_grp_raw["ID"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(id_list) == 0:
        id_list = [1]

    # --- Кнопки управления (ДО data_editor) ---
    g1, g2, g3 = st.columns([1.3, 1.7, 2.0])

    with g1:
        scen_for_new_group = st.selectbox("Сценарий для новой группы", scen_list2, key="add_group_scen")
        if st.button("➕ Добавить группу", use_container_width=True):
            df = st.session_state.df_grp.copy()
            df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
            next_id = int(pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int).max()) + 1 if len(df) else 1

            new_row = {
                "ID": next_id,
                "Сценарий i": int(scen_for_new_group),
                "Группа j": "Новая группа",
                "t_p,i,j (мин)": 6.0,
                "t_н.э,i,j (мин)": 1.5,
                "t_ск,i,j (мин)": 1.0,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.df_grp = df
            st.session_state.selected_group_id = int(next_id)
            force_rerun()

    with g2:
        df_tmp = st.session_state.df_grp.copy()
        df_tmp = ensure_unique_positive_int_ids(df_tmp, "ID", start_from=1)
        id_list2 = sorted(pd.to_numeric(df_tmp["ID"], errors="coerce").dropna().astype(int).unique().tolist())
        gid_del = st.selectbox("Удалить группу по ID", id_list2, key="del_group_id")
        if st.button("🗑️ Удалить выбранную группу", use_container_width=True):
            df = st.session_state.df_grp.copy()
            df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)
            df = df.loc[df["ID"] != int(gid_del)].copy()
            df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
            st.session_state.df_grp = df

            if len(df) > 0:
                st.session_state.selected_group_id = int(df["ID"].iloc[0])
            force_rerun()

    with g3:
        st.info(
            "Можно добавлять/удалять группы кнопками или напрямую через таблицу (num_rows='dynamic').",
            icon="ℹ️"
        )

    # --- Таблица редактирования групп ---
    df_grp_raw2 = st.session_state.df_grp.copy()
    df_grp_raw2 = ensure_unique_positive_int_ids(df_grp_raw2, "ID", start_from=1)

    df_grp_edit = st.data_editor(
        df_grp_raw2,
        num_rows="dynamic",
        use_container_width=True,
        disabled=["ID"],
        column_config={
            "ID": st.column_config.NumberColumn(min_value=1, step=1),
            "Сценарий i": st.column_config.NumberColumn(min_value=1, step=1),
            "Группа j": st.column_config.TextColumn(),
            "t_p,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
            "t_н.э,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
            "t_ск,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
        },
        key="editor_groups"
    )

    df_grp_edit = ensure_unique_positive_int_ids(df_grp_edit, "ID", start_from=1)
    df_grp_edit["Сценарий i"] = pd.to_numeric(df_grp_edit["Сценарий i"], errors="coerce").fillna(scen_list2[0]).astype(int)
    st.session_state.df_grp = df_grp_edit

    def fmt_sci(x: float, digits: int = 2) -> str:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
             return ""
            return f"{v:.{digits}e}"
        except Exception:
            return str(x)

    def format_df_scientific(df: pd.DataFrame, sci_cols: list[str], digits: int = 2) -> pd.DataFrame:
        """Делает копию df и форматирует указанные числовые столбцы в e-нотацию (только для отображения)."""
        out = df.copy()
        for c in sci_cols:
            if c in out.columns:
                out[c] = out[c].apply(lambda v: fmt_sci(v, digits=digits) if pd.notna(v) else "")
        return out
    
# -----------------------------
# Расчёт
# -----------------------------
df_scen_calc, df_rows_calc, df_agg, r_trad, r_iot = compute_all(
    st.session_state.df_scen,
    st.session_state.df_grp,
    alpha=alpha,
    k_iot_total=k_iot_total,
)

# -----------------------------
# Основное полотно: результаты + графика + "прозрачный расчёт" (ПО ТВОЕЙ ПРОСЬБЕ)
# -----------------------------
st.subheader("Результаты расчёта")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("R (традиц), год^-1", f"{r_trad:.6g}")
with m2:
    delta_val = r_iot - r_trad  # отрицательное = риск снизился (это хорошо)
    st.metric(
        "R (IoT-СОУЭ), год^-1",
        f"{r_iot:.6g}",
        delta=f"{delta_val:.2e}",
        delta_color="inverse"   # минус будет зелёным
    )

with m3:
    if r_trad > 0:
        st.metric("Снижение, %", f"{(1.0 - r_iot / r_trad) * 100.0:.2f}%")
    else:
        st.metric("Снижение, %", "—")

components.html(compare_risk_component_html(r_trad, r_iot, R_NORM), height=420, scrolling=False)

# --- Прозрачный расчёт в основном полотне ---
st.subheader("Динамический расчет коэффициентов адаптивности и надежности IoT-СОУЭ")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("K_IoT (по шкалам)", f"{k_iot_score:.3f}")
c2.metric("K_связь (OR)", f"{k_comm:.3f}")
c3.metric("K_над", f"{k_rel:.3f}")
c4.metric("K_IoT_итог", f"{k_iot_total:.3f}")
c5.metric("α·K_IoT_итог", f"{delta_k_soue:.3f}")

# --- Вывод текущих "бегунков" (какие сейчас времена для выбранной группы) ---
st.subheader("Значения времён для выбранной группы для P_э,i,j (формула (6) №1140)")

sel_id = int(st.session_state.selected_group_id)
dfg = st.session_state.df_grp.copy()
dfs = st.session_state.df_scen.copy()
dfs["Сценарий i"] = pd.to_numeric(dfs["Сценарий i"], errors="coerce").fillna(0).astype(int)

row_g = dfg.loc[dfg["ID"].astype(int) == sel_id]
if len(row_g) > 0:
    row_g = row_g.iloc[0]
    scen_id = int(row_g["Сценарий i"])
    row_s = dfs.loc[dfs["Сценарий i"] == scen_id]
    t_bl = safe_float(row_s.iloc[0]["t_бл,i (мин)"]) if len(row_s) > 0 else float("nan")

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("ID группы", f"{sel_id}")
    cc2.metric("Сценарий i", f"{scen_id}")
    cc3.metric("t_бл,i (мин)", f"{t_bl:.3f}" if not math.isnan(t_bl) else "—")
    cc4.metric("t_p,i,j (мин)", f"{safe_float(row_g['t_p,i,j (мин)']):.3f}")
    cc5.metric("t_н.э,i,j (мин)", f"{safe_float(row_g['t_н.э,i,j (мин)']):.3f}")
    st.metric("t_ск,i,j (мин)", f"{safe_float(row_g['t_ск,i,j (мин)']):.3f}")
else:
    st.info("Выбранная группа не найдена (возможно, ты удалил строку). Выбери группу в левом меню.")

# -----------------------------
# Таблицы результатов
# -----------------------------
st.markdown("### Агрегирование по сценариям и итог (формулы (2)–(3) №1140)")
df_agg_view = format_df_scientific(
    df_agg,
    sci_cols=["R_i (традиц) = max_j", "R_i (IoT) = max_j"],
    digits=2
)
st.dataframe(df_agg_view, use_container_width=True)

st.markdown("### Расчёт ИПР по группам (формула (4) №1140)")
cols_show = [
    "ID", "Сценарий i", "Группа j",
    "t_бл,i (мин)", "t_p,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)",
    "P_э,i,j",
    "K_п.з,i (традиц)", "K_п.з,i (IoT)",
    "R_i,j (традиц)", "R_i,j (IoT)"
]
cols_show = [c for c in cols_show if c in df_rows_calc.columns]
df_rows_view = df_rows_calc[cols_show].copy()
df_rows_view = format_df_scientific(
    df_rows_view,
    sci_cols=["R_i,j (традиц)", "R_i,j (IoT)", "Q_n,i (год^-1)"],
    digits=2
)
st.dataframe(df_rows_view, use_container_width=True)

st.markdown("### Промежуточные коэффициенты по сценариям")
cols_scen = [
    "Сценарий i",
    "K_обн,i",
    "K_СОУЭ,i (традиц)",
    "K_СОУЭ,i (IoT)",
    "K_ПДЗ,i",
    "K_п.з,i (традиц)",
    "K_п.з,i (IoT)",
    "P_пр,i",
]
cols_scen = [c for c in cols_scen if c in df_scen_calc.columns]
df_scen_view = df_scen_calc[cols_scen].copy()
df_scen_view = format_df_scientific(
    df_scen_view,
    sci_cols=["Q_n,i (год^-1)"] if "Q_n,i (год^-1)" in df_scen_view.columns else [],
    digits=2
)
st.dataframe(df_scen_view, use_container_width=True)
# -----------------------------
# Формулы — свёрнутый блок
# -----------------------------
st.subheader("Сравнительные формулы для расчёта ИПР по традиционному методу и с IoT-СОУЭ")

with st.expander("Показать/скрыть формулы", expanded=False):
    st.markdown("#### Традиционный расчёт по №1140")
    st.latex(r"R \le R_{\text{norm}}, \quad R_{\text{norm}} = 10^{-6}\ \text{год}^{-1}")
    st.latex(r"R = \max\{R_1, \dots, R_i, \dots, R_K\}, \quad R_i = \max\{R_{i,1}, \dots, R_{i,m}\}")
    st.latex(r"R_{i,j}=Q_{n,i}\cdot(1-K_{\text{ап},i})\cdot P_{\text{пр},i}\cdot(1-P_{\text{э},i,j})\cdot(1-K_{\text{п.з},i})")
    st.latex(r"P_{\text{пр},i} = \frac{t_{\text{пр},i}}{24}")

    st.markdown("**Pэ по формуле (6) №1140**")
    st.latex(r"""
P_{\text{э},i,j}=
\begin{cases}
0{.}999\cdot\dfrac{0{.}8\,t_{\text{бл},i}-t_{p,i,j}}{t_{\text{н.э},i,j}}, & \text{если } t_{p,i,j}<0{.}8\,t_{\text{бл},i}<t_{p,i,j}+t_{\text{н.э},i,j}\ \text{и}\ t_{\text{ск},i,j}\le 6\\[6pt]
0{.}999, & \text{если } t_{p,i,j}+t_{\text{н.э},i,j}\le 0{.}8\,t_{\text{бл},i}\ \text{и}\ t_{\text{ск},i,j}\le 6\\[6pt]
0, & \text{если } t_{p,i,j}\ge 0{.}8\,t_{\text{бл},i}\ \text{или}\ t_{\text{ск},i,j}> 6
\end{cases}
""")

    st.markdown("**Kп.з по формуле (7) №1140**")
    st.latex(r"K_{\text{п.з,i}}=1-(1-K_{\text{обн},i}\cdot K_{\text{СОУЭ},i})\cdot(1-K_{\text{обн},i}\cdot K_{\text{ПДЗ},i})")

    st.markdown("#### Расчёт с IoT-СОУЭ (аддитивный вариант в демонстраторе)")
    st.latex(r"K^{(\text{IoT})}_{\text{СОУЭ},i}=\min\left(0{.}99,\ K_{\text{СОУЭ},i}+\alpha\cdot K^{\text{итог}}_{\text{IoT}}\right)")
    st.latex(r"K^{\text{итог}}_{\text{IoT}}=\min\left(0{.}99,\ K_{\text{IoT}}\cdot K_{\text{над}}\right)")
    st.latex(r"K_{\text{связь}} = 1-(1-K_{\text{канал осн}})\cdot(1-K_{\text{канал рез}})")
    st.latex(r"K_{\text{над}} = K_{\text{обнаруж}}\cdot K_{\text{связь}}\cdot K_{\text{логика}}\cdot K_{\text{оповещ}}\cdot K_{\text{питание}}\cdot(1-p_{\text{cyber}})")
    st.latex(r"K_{\text{п.з},i}^{(\text{IoT})}=1-(1-K_{\text{обн},i}\cdot K^{(\text{IoT})}_{\text{СОУЭ},i})\cdot(1-K_{\text{обн},i}\cdot K_{\text{ПДЗ},i})")

# -----------------------------
# Диагностика Pэ по выбранной группе
# -----------------------------
st.subheader("Диагностика P_э,i,j по выбранной группе")

df_rows_diag = df_rows_calc.copy()
if len(df_rows_diag) > 0 and "ID" in df_rows_diag.columns:
    ids = df_rows_diag["ID"].astype(int).to_list()
    # выберем по умолчанию текущую группу
    default_index = ids.index(sel_id) if sel_id in ids else 0
    sel_diag_id = st.selectbox("Выбери ID строки для объяснения P_э", ids, index=default_index)

    row = df_rows_diag.loc[df_rows_diag["ID"].astype(int) == int(sel_diag_id)].iloc[0]
    t_bl = safe_float(row.get("t_бл,i (мин)", 0.0))
    t_p = safe_float(row.get("t_p,i,j (мин)", 0.0))
    t_ne = safe_float(row.get("t_н.э,i,j (мин)", 0.0))
    t_ck = safe_float(row.get("t_ск,i,j (мин)", 0.0))
    border = 0.8 * t_bl
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("0.8·t_бл, мин", f"{border:.3f}")
    c2.metric("t_p, мин", f"{t_p:.3f}")
    c3.metric("t_p+t_н.э, мин", f"{(t_p+t_ne):.3f}")
    c4.metric("t_ск, мин", f"{t_ck:.3f}")

    st.metric("P_э,i,j", f"{safe_float(row.get('P_э,i,j', 0.0)):.3f}")


    if t_ck > 6:
        st.warning("t_ск > 6 мин ⇒ по формуле (6) P_э = 0.")
    else:
        if (t_p + t_ne) <= border:
            st.success("t_p + t_н.э ≤ 0.8·t_бл и t_ск ≤ 6 ⇒ P_э = 0.999.")
        elif (t_p < border) and (border < (t_p + t_ne)):
            st.info(
                "0.8·t_бл попало между t_p и t_p + t_н.э ⇒ возможны промежуточные значения (ветвь №1140)."
            )
        else:
            st.warning("t_p ≥ 0.8·t_бл ⇒ по формуле (6) P_э = 0.")
else:
    st.info("Нет строк групп для диагностики (добавьте группы).")

# -----------------------------
# Выгрузка
# -----------------------------
st.subheader("Выгрузка результатов (CSV)")

csv_rows = df_rows_calc.to_csv(index=False).encode("utf-8-sig")
csv_scen = df_scen_calc.to_csv(index=False).encode("utf-8-sig")
csv_agg = df_agg.to_csv(index=False).encode("utf-8-sig")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("Скачать CSV: построчно (группы)", data=csv_rows, file_name="calc_rows.csv", mime="text/csv")
with c2:
    st.download_button("Скачать CSV: сценарии (коэффициенты)", data=csv_scen, file_name="calc_scenarios.csv", mime="text/csv")
with c3:
    st.download_button("Скачать CSV: агрегирование", data=csv_agg, file_name="calc_aggregate.csv", mime="text/csv")

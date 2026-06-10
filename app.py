import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE_DIR, 'Gross_Profit_vs_Competitor_0514_Copliot.xlsx')

# ══════════════════════════════════════════════
#  全域 CSS
# ══════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* 整體背景 */
    .stApp { background-color: #EAF4F4; }

    /* Sidebar 背景 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D0E8E8;
    }

    /* 漢堡選單按鈕 - 收合狀態 */
    button[data-testid="stBaseButton-headerNoPadding"] > div > svg,
    button[data-testid="stSidebarNavToggleButton"] > div > svg,
    [data-testid="collapsedControl"] svg {
        display: none !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]::before,
    button[data-testid="stSidebarNavToggleButton"]::before,
    [data-testid="collapsedControl"]::before {
        content: "☰";
        font-size: 1.3rem;
        color: #2A7F7F;
        font-weight: 400;
    }

    /* 卡片樣式 */
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* 頁面標題 */
    .page-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1A5C5C;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #6B8F8F;
        margin-bottom: 24px;
    }

    /* 區塊標題 */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #2A7F7F;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #C2E0E0;
    }

    /* 主要按鈕 */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #2A9D8F;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        letter-spacing: 0.3px;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #21867A;
    }

    /* Metric 卡片 */
    [data-testid="stMetric"] {
        background: #F4FAFA;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #D4EEEE;
    }
    [data-testid="stMetricLabel"] { color: #6B8F8F; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #1A5C5C; font-weight: 700; }

    /* Info / Success / Warning / Error 訊息 */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* Sidebar 文字 */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        color: #2C4A4A;
    }

    /* 輸入框 */
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select {
        border-radius: 8px;
        border-color: #C2E0E0;
    }

    /* Divider */
    hr { border-color: #D0E8E8; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* 移除 Streamlit watermark */
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

def card(content_fn, *args, **kwargs):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  特徵工程
# ══════════════════════════════════════════════
def build_features(df):
    comp_cols = ['Competitor A', 'Competitor B', 'Competitor C']
    df = df.copy()
    df['competitor_count']   = df[comp_cols].notna().sum(axis=1)
    df['no_competitor_info'] = df[comp_cols].isna().all(axis=1).astype(int)
    df['min_competitor_price'] = df[comp_cols].min(axis=1)
    df['Diff_A'] = df['Unit Price'] - df['Competitor A']
    df['Diff_B'] = df['Unit Price'] - df['Competitor B']
    df['Diff_C'] = df['Unit Price'] - df['Competitor C']
    df['Grant_Ratio'] = df['Energy grant amount'] / df['Subtotal Price']
    df['Grant_Ratio'] = df['Grant_Ratio'].where(np.isfinite(df['Grant_Ratio']), 0)
    df['Convert_to_Order'] = df['Convert to Order (0:Success, 1:Fail)']
    df['highest_competitor_per_row'] = df[comp_cols].max(axis=1)
    product_ceiling_temp = df.groupby('Product')['highest_competitor_per_row'].transform('max').mul(1.5)
    global_fallback_temp = df['Unit Price'].median() * 1.5
    df['rational_imputed_price'] = product_ceiling_temp.fillna(global_fallback_temp)
    df['min_competitor_price_filled'] = df['min_competitor_price'].fillna(df['rational_imputed_price'])
    df['price_diff']  = df['Unit Price'] - df['min_competitor_price_filled']
    df['price_ratio'] = df['Unit Price'] / df['min_competitor_price_filled']
    return df

# ══════════════════════════════════════════════
#  訓練模型
# ══════════════════════════════════════════════
@st.cache_resource
def train_models():
    df_raw = pd.read_excel(XLSX_PATH, sheet_name='raw data')
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw.drop_duplicates()
    df = build_features(df_raw)

    features = [
        'Kw', 'Qty', 'Subtotal Price', 'Gross Margin Rate',
        'Energy grant amount', 'Grant_Ratio',
        'competitor_count', 'no_competitor_info',
        'Diff_A', 'Diff_B', 'Diff_C', 'price_diff', 'price_ratio'
    ]

    X = df[features]
    y = df['Convert_to_Order']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    train_idx, test_idx = y_train.index, y_test.index
    train_full = df.loc[train_idx].copy()
    test_full  = df.loc[test_idx].copy()
    product_ceiling = train_full.groupby('Product')['highest_competitor_per_row'].max().mul(1.5)
    global_fallback = train_full['Unit Price'].median() * 1.5

    for idx_set, full_set in [(train_idx, train_full), (test_idx, test_full)]:
        imputed = full_set['Product'].map(product_ceiling).fillna(global_fallback)
        df.loc[idx_set, 'rational_imputed_price']      = imputed.values
        df.loc[idx_set, 'min_competitor_price_filled'] = df.loc[idx_set, 'min_competitor_price'].fillna(imputed).values
        df.loc[idx_set, 'price_diff']  = (df.loc[idx_set, 'Unit Price'] - df.loc[idx_set, 'min_competitor_price_filled']).values
        df.loc[idx_set, 'price_ratio'] = (df.loc[idx_set, 'Unit Price'] / df.loc[idx_set, 'min_competitor_price_filled']).values

    X = df[features]
    X_train = X.loc[train_idx]
    X_test  = X.loc[test_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.replace([np.inf,-np.inf], np.nan).fillna(0))

    neg_count, pos_count = Counter(y_train)[0], Counter(y_train)[1]
    lr  = LogisticRegression(max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                  min_samples_split=10, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', n_estimators=200, max_depth=6,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=0.1, reg_lambda=1.0,
                         scale_pos_weight=neg_count/pos_count, random_state=42)
    lr.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)
    xgb.fit(X_train_s, y_train)

    return dict(lr=lr, rf=rf, xgb=xgb, scaler=scaler, features=features,
                product_ceiling=product_ceiling, global_fallback=global_fallback,
                product_list=sorted(df['Product'].dropna().unique().tolist()))

@st.cache_data
def load_bcg_data():
    df_raw = pd.read_excel(XLSX_PATH, sheet_name='raw data')
    df_raw.columns = df_raw.columns.str.strip()
    df_raw = df_raw.drop_duplicates()
    df = build_features(df_raw)

    bcg_df = df.groupby('Product').agg(
        avg_gross_margin=('Gross Margin Rate', 'mean'),
        win_rate=('Convert_to_Order', lambda x: (x==0).mean()),
        sample_size=('Convert_to_Order', 'count'),
        avg_price_diff=('price_diff', 'mean'),
        total_revenue=('Subtotal Price', 'sum'),
    ).reset_index()

    gm_med = bcg_df['avg_gross_margin'].median()
    wr_med = bcg_df['win_rate'].median()

    def quadrant(row):
        hg = row['avg_gross_margin'] >= gm_med
        hw = row['win_rate'] >= wr_med
        if hg and hw:   return 'Star'
        if hg and not hw: return 'Question Mark'
        if not hg and hw: return 'Cash Cow'
        return 'Dog'

    bcg_df['Quadrant'] = bcg_df.apply(quadrant, axis=1)
    return bcg_df, gm_med, wr_med, df

def preprocess_input(ckpt, product, kw, qty, unit_price, subtotal,
                     gross_margin, energy_grant, comp_a, comp_b, comp_c):
    features        = ckpt['features']
    scaler          = ckpt['scaler']
    product_ceiling = ckpt['product_ceiling']
    global_fallback = ckpt['global_fallback']

    comp_notna = [v for v in [comp_a, comp_b, comp_c] if v is not None]
    competitor_count   = len(comp_notna)
    no_competitor_info = 1 if competitor_count == 0 else 0
    min_comp = min(comp_notna) if comp_notna else None

    if min_comp is None:
        min_comp_filled = product_ceiling.get(product, global_fallback)
    else:
        min_comp_filled = min_comp

    price_diff  = unit_price - min_comp_filled
    price_ratio = (unit_price / min_comp_filled) if min_comp_filled != 0 else 0
    grant_ratio = (energy_grant / subtotal) if subtotal != 0 else 0
    if not np.isfinite(grant_ratio): grant_ratio = 0

    row = {
        'Kw': kw, 'Qty': qty, 'Subtotal Price': subtotal,
        'Gross Margin Rate': gross_margin,
        'Energy grant amount': energy_grant, 'Grant_Ratio': grant_ratio,
        'competitor_count': competitor_count, 'no_competitor_info': no_competitor_info,
        'Diff_A': (unit_price - comp_a) if comp_a is not None else np.nan,
        'Diff_B': (unit_price - comp_b) if comp_b is not None else np.nan,
        'Diff_C': (unit_price - comp_c) if comp_c is not None else np.nan,
        'price_diff': price_diff, 'price_ratio': price_ratio,
    }
    df_row = pd.DataFrame([row])[features].replace([np.inf,-np.inf], np.nan).fillna(0)
    return scaler.transform(df_row)

# ══════════════════════════════════════════════
#  App 入口
# ══════════════════════════════════════════════
st.set_page_config(page_title="銷售分析系統", page_icon="📊", layout="wide")
inject_css()

with st.spinner("模型初始化中，請稍候..."):
    ckpt = train_models()

lr, rf, xgb  = ckpt['lr'], ckpt['rf'], ckpt['xgb']
product_list = ckpt['product_list']
bcg_df, gm_med, wr_med, raw_df = load_bcg_data()

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("### 銷售分析系統")
    st.markdown("---")
    page = st.radio("", ["接單成功率預測", "BCG 產品策略分析"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("LR  AUC 0.856\nRF  AUC 0.944\nXGB AUC 0.951")

# ══════════════════════════════════════════════
#  頁面一：預測
# ══════════════════════════════════════════════
if page == "接單成功率預測":
    st.markdown('<p class="page-title">接單成功率預測</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">輸入報價資訊，預測此筆詢單的接單成功機率</p>', unsafe_allow_html=True)

    # 產品 & 價格
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<p class="section-title">產品資訊</p>', unsafe_allow_html=True)
        product      = st.selectbox("產品型號", product_list, label_visibility="visible")
        kw           = st.number_input("功率 (Kw)", min_value=1, max_value=2000, value=75)
        qty          = st.number_input("數量 (Qty)", min_value=1, max_value=100, value=1)
    with col2:
        st.markdown('<p class="section-title">報價資訊</p>', unsafe_allow_html=True)
        unit_price   = st.number_input("單價 (Unit Price)", min_value=1, value=700000, step=1000)
        subtotal     = st.number_input("小計 (Subtotal Price)", min_value=1, value=700000, step=1000)
        gross_margin = st.number_input("毛利率 (Gross Margin Rate)", min_value=0.0, max_value=1.0,
                                       value=0.35, step=0.01, format="%.4f")
        energy_grant = st.number_input("能源補助金額", min_value=0, value=375000, step=1000)

    # 競爭對手
    st.markdown('<p class="section-title" style="margin-top:8px">競爭對手報價（選填）</p>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3, gap="large")
    with col3:
        use_a = st.checkbox("Competitor A")
        comp_a_val = st.number_input("A 報價", min_value=0, value=0, step=1000,
                                      disabled=not use_a, label_visibility="collapsed")
    with col4:
        use_b = st.checkbox("Competitor B")
        comp_b_val = st.number_input("B 報價", min_value=0, value=0, step=1000,
                                      disabled=not use_b, label_visibility="collapsed")
    with col5:
        use_c = st.checkbox("Competitor C")
        comp_c_val = st.number_input("C 報價", min_value=0, value=0, step=1000,
                                      disabled=not use_c, label_visibility="collapsed")

    comp_a = comp_a_val if use_a else None
    comp_b = comp_b_val if use_b else None
    comp_c = comp_c_val if use_c else None

    # 產品歷史提示
    prod_row = bcg_df[bcg_df['Product'] == product]
    if not prod_row.empty:
        r = prod_row.iloc[0]
        q_label = {'Star':'Star（高毛利 × 高轉換）','Question Mark':'Question Mark（高毛利 × 低轉換）',
                   'Cash Cow':'Cash Cow（低毛利 × 高轉換）','Dog':'Dog（低毛利 × 低轉換）'}
        st.info(f"產品 {product} 歷史定位：{q_label[r['Quadrant']]}　｜　歷史勝率 {r['win_rate']:.1%}　｜　平均毛利率 {r['avg_gross_margin']:.1%}")

    st.markdown("")
    if st.button("預測接單成功率", use_container_width=True, type="primary"):
        X_input  = preprocess_input(ckpt, product, kw, qty, unit_price, subtotal,
                                    gross_margin, energy_grant, comp_a, comp_b, comp_c)
        prob_lr  = lr.predict_proba(X_input)[0][0]
        prob_rf  = rf.predict_proba(X_input)[0][0]
        prob_xgb = xgb.predict_proba(X_input)[0][0]
        avg_prob = (prob_lr + prob_rf + prob_xgb) / 3

        st.markdown("---")
        st.markdown('<p class="section-title">預測結果</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Logistic Regression", f"{prob_lr:.1%}")
        c2.metric("Random Forest",       f"{prob_rf:.1%}")
        c3.metric("XGBoost  ★",          f"{prob_xgb:.1%}")

        color = "#2A9D8F" if avg_prob >= 0.6 else "#E9C46A" if avg_prob >= 0.4 else "#E76F51"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob * 100,
            title={'text': "綜合接單成功率", 'font': {'size': 16, 'color': '#1A5C5C'}},
            number={'suffix': '%', 'font': {'size': 44, 'color': '#1A5C5C'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6B8F8F'},
                'bar': {'color': color},
                'bgcolor': '#F4FAFA',
                'bordercolor': '#D4EEEE',
                'steps': [
                    {'range': [0,  40], 'color': '#FAE8E4'},
                    {'range': [40, 60], 'color': '#FDF6E3'},
                    {'range': [60,100], 'color': '#E4F5F2'},
                ],
            }
        ))
        fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        if avg_prob >= 0.6:
            st.success(f"綜合三模型平均接單成功率 {avg_prob:.1%}，建議積極跟進。")
        elif avg_prob >= 0.4:
            st.warning(f"綜合三模型平均接單成功率 {avg_prob:.1%}，建議評估調整報價條件。")
        else:
            st.error(f"綜合三模型平均接單成功率 {avg_prob:.1%}，接單難度較高，建議重新評估策略。")

# ══════════════════════════════════════════════
#  頁面二：BCG
# ══════════════════════════════════════════════
else:
    st.markdown('<p class="page-title">BCG 產品策略分析</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">以毛利率（獲利能力）× 接單轉換率（市場競爭力）分析各產品線策略定位</p>', unsafe_allow_html=True)

    # 象限說明
    q1, q2, q3, q4 = st.columns(4, gap="small")
    q1.success("**Star**\n\n高毛利 × 高轉換\n\n優先擴大投入")
    q2.info("**Question Mark**\n\n高毛利 × 低轉換\n\n分析競爭障礙")
    q3.warning("**Cash Cow**\n\n低毛利 × 高轉換\n\n嘗試適度調價")
    q4.error("**Dog**\n\n低毛利 × 低轉換\n\n重新評估投入")

    st.markdown("---")

    # 篩選
    f1, f2 = st.columns([2,1])
    with f1:
        q_filter = st.multiselect("篩選象限", ['Star','Question Mark','Cash Cow','Dog'],
                                   default=['Star','Question Mark','Cash Cow','Dog'])
    with f2:
        min_n = st.slider("最少樣本數", 1, 100, 5)

    filtered = bcg_df[(bcg_df['Quadrant'].isin(q_filter)) & (bcg_df['sample_size'] >= min_n)]

    color_map = {'Star':'#2A9D8F','Question Mark':'#457B9D','Cash Cow':'#E9C46A','Dog':'#E76F51'}

    fig = go.Figure()
    for q, grp in filtered.groupby('Quadrant'):
        fig.add_trace(go.Scatter(
            x=grp['win_rate'], y=grp['avg_gross_margin'],
            mode='markers+text', name=q,
            text=grp['Product'], textposition='top center',
            textfont=dict(size=11, color='#2C4A4A'),
            marker=dict(size=np.sqrt(grp['sample_size'])*4, color=color_map[q],
                        opacity=0.8, line=dict(color='white', width=2)),
            customdata=np.stack([grp['sample_size'], grp['total_revenue'], grp['avg_price_diff']], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>轉換率：%{x:.1%}<br>毛利率：%{y:.1%}<br>"
                "樣本數：%{customdata[0]}<br>總營收：%{customdata[1]:,.0f}<extra></extra>")
        ))

    fig.add_vline(x=wr_med, line_dash="dot", line_color="#A0C4C4", line_width=1.5)
    fig.add_hline(y=gm_med, line_dash="dot", line_color="#A0C4C4", line_width=1.5)
    fig.update_xaxes(title="接單轉換率", tickformat=".0%", gridcolor="#E8F4F4")
    fig.update_yaxes(title="平均毛利率",  tickformat=".0%", gridcolor="#E8F4F4")
    fig.update_layout(
        title=dict(text="BCG 矩陣（氣泡大小 = 詢價筆數）", font=dict(size=15, color='#1A5C5C')),
        height=560, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE',
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        hovermode='closest', font=dict(color='#2C4A4A')
    )
    st.plotly_chart(fig, use_container_width=True)

    # 產品明細
    st.markdown('<p class="section-title">產品明細</p>', unsafe_allow_html=True)
    disp = filtered[['Product','Quadrant','avg_gross_margin','win_rate','sample_size','total_revenue']].copy()
    disp = disp.sort_values('total_revenue', ascending=False).reset_index(drop=True)
    disp.columns = ['產品','象限','平均毛利率','轉換率','詢價筆數','總營收']
    st.dataframe(
        disp.style.format({'平均毛利率':'{:.1%}','轉換率':'{:.1%}','總營收':'{:,.0f}'}),
        use_container_width=True, height=380
    )

    # 單一產品深入
    st.markdown("---")
    st.markdown('<p class="section-title">單一產品深入分析</p>', unsafe_allow_html=True)
    sel = st.selectbox("選擇產品", filtered['Product'].tolist())
    pi  = filtered[filtered['Product']==sel].iloc[0]
    pr  = raw_df[raw_df['Product']==sel]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("象限",      pi['Quadrant'])
    m2.metric("平均毛利率", f"{pi['avg_gross_margin']:.1%}")
    m3.metric("接單轉換率", f"{pi['win_rate']:.1%}")
    m4.metric("詢價筆數",   int(pi['sample_size']))

    h1, h2 = st.columns(2)
    with h1:
        fig_h = px.histogram(pr, x='Gross Margin Rate', color='Convert_to_Order',
                              color_discrete_map={0:'#2A9D8F',1:'#E76F51'},
                              labels={'Convert_to_Order':'結果','Gross Margin Rate':'毛利率'},
                              title=f"{sel}：毛利率分布（綠=成交 紅=未成交）",
                              barmode='overlay', opacity=0.75)
        fig_h.update_xaxes(tickformat=".0%")
        fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE',
                             font=dict(color='#2C4A4A'))
        st.plotly_chart(fig_h, use_container_width=True)
    with h2:
        fig_b = px.box(pr, x='Convert_to_Order', y='price_diff',
                        color='Convert_to_Order',
                        color_discrete_map={0:'#2A9D8F',1:'#E76F51'},
                        labels={'Convert_to_Order':'結果（0=成交）','price_diff':'與競爭者價差'},
                        title=f"{sel}：價差分布")
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE',
                             font=dict(color='#2C4A4A'))
        st.plotly_chart(fig_b, use_container_width=True)

    strategy_map = {
        'Star':          '此產品兼具高獲利與高競爭力，建議維持現有定價策略並積極擴大銷售量。',
        'Cash Cow':      '市場接受度高但毛利偏低。建議評估適度調漲空間，在不明顯影響成交率的前提下提升獲利。',
        'Question Mark': '獲利能力佳但成交率偏低。建議深入分析競爭者條件，評估降價或強化補助方案以提升轉換率。',
        'Dog':           '毛利與轉換率雙低，建議檢討此產品線的資源配置，考慮縮減報價投入或重新定位。',
    }
    st.info(f"策略建議（{pi['Quadrant']}）：{strategy_map[pi['Quadrant']]}")

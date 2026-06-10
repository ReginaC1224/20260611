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
#  語言字典
# ══════════════════════════════════════════════
TEXT = {
    'zh': {
        'app_title':       '銷售分析系統',
        'nav_predict':     '接單成功率預測',
        'nav_bcg':         'BCG 產品策略分析',
        'model_perf':      '模型效能',
        'loading':         '模型初始化中，請稍候...',
        # 預測頁
        'pred_title':      '接單成功率預測',
        'pred_subtitle':   '輸入報價資訊，預測此筆詢單的接單成功機率',
        'sec_product':     '產品資訊',
        'sec_price':       '報價資訊',
        'sec_competitor':  '競爭對手報價（選填）',
        'lbl_product':     '產品型號',
        'lbl_kw':          '功率 (Kw)',
        'lbl_qty':         '數量 (Qty)',
        'lbl_unit_price':  '單價 (Unit Price)',
        'lbl_subtotal':    '小計 (Subtotal Price)',
        'lbl_gm':          '毛利率 (Gross Margin Rate)',
        'lbl_grant':       '能源補助金額',
        'chk_a':           '有 Competitor A 資料',
        'chk_b':           '有 Competitor B 資料',
        'chk_c':           '有 Competitor C 資料',
        'btn_predict':     '預測接單成功率',
        'sec_result':      '預測結果',
        'gauge_title':     '綜合接單成功率',
        'hist_product':    '產品 {p} 歷史定位：{q}　｜　歷史勝率 {wr}　｜　平均毛利率 {gm}',
        'msg_high':        '綜合三模型平均接單成功率 {p}，建議積極跟進。',
        'msg_mid':         '綜合三模型平均接單成功率 {p}，建議評估調整報價條件。',
        'msg_low':         '綜合三模型平均接單成功率 {p}，接單難度較高，建議重新評估策略。',
        # BCG 頁
        'bcg_title':       'BCG 產品策略分析',
        'bcg_subtitle':    '以毛利率（獲利能力）× 接單轉換率（市場競爭力）分析各產品線策略定位',
        'q_star_title':    'Star',
        'q_star_desc':     '高毛利 × 高轉換\n\n優先擴大投入',
        'q_qm_title':      'Question Mark',
        'q_qm_desc':       '高毛利 × 低轉換\n\n分析競爭障礙',
        'q_cc_title':      'Cash Cow',
        'q_cc_desc':       '低毛利 × 高轉換\n\n嘗試適度調價',
        'q_dog_title':     'Dog',
        'q_dog_desc':      '低毛利 × 低轉換\n\n重新評估投入',
        'filter_q':        '篩選象限',
        'filter_n':        '最少樣本數',
        'chart_title':     'BCG 矩陣（氣泡大小 = 詢價筆數）',
        'x_axis':          '接單轉換率',
        'y_axis':          '平均毛利率',
        'sec_table':       '產品明細',
        'col_product':     '產品',
        'col_quadrant':    '象限',
        'col_gm':          '平均毛利率',
        'col_wr':          '轉換率',
        'col_n':           '詢價筆數',
        'col_rev':         '總營收',
        'sec_deep':        '單一產品深入分析',
        'sel_product':     '選擇產品',
        'metric_q':        '象限',
        'metric_gm':       '平均毛利率',
        'metric_wr':       '接單轉換率',
        'metric_n':        '詢價筆數',
        'hist_chart':      '{p}：毛利率分布（綠=成交 紅=未成交）',
        'box_chart':       '{p}：價差分布',
        'lbl_result':      '結果',
        'lbl_gm_x':        '毛利率',
        'lbl_result_0':    '結果（0=成交）',
        'lbl_price_diff':  '與競爭者價差',
        'strategy_label':  '策略建議（{q}）：{s}',
        'strategy': {
            'Star':          '此產品兼具高獲利與高競爭力，建議維持現有定價策略並積極擴大銷售量。',
            'Question Mark': '獲利能力佳但成交率偏低。建議深入分析競爭者條件，評估降價或強化補助方案以提升轉換率。',
            'Cash Cow':      '市場接受度高但毛利偏低。建議評估適度調漲空間，在不明顯影響成交率的前提下提升獲利。',
            'Dog':           '毛利與轉換率雙低，建議檢討此產品線的資源配置，考慮縮減報價投入或重新定位。',
        },
        'q_label': {
            'Star':          'Star（高毛利 × 高轉換）',
            'Question Mark': 'Question Mark（高毛利 × 低轉換）',
            'Cash Cow':      'Cash Cow（低毛利 × 高轉換）',
            'Dog':           'Dog（低毛利 × 低轉換）',
        },
    },
    'en': {
        'app_title':       'Sales Analytics System',
        'nav_predict':     'Win Rate Prediction',
        'nav_bcg':         'BCG Product Analysis',
        'model_perf':      'Model Performance',
        'loading':         'Initializing models, please wait...',
        # Prediction page
        'pred_title':      'Win Rate Prediction',
        'pred_subtitle':   'Enter quote details to predict the probability of winning this order.',
        'sec_product':     'Product Info',
        'sec_price':       'Pricing Info',
        'sec_competitor':  'Competitor Pricing (Optional)',
        'lbl_product':     'Product Code',
        'lbl_kw':          'Power (Kw)',
        'lbl_qty':         'Quantity (Qty)',
        'lbl_unit_price':  'Unit Price',
        'lbl_subtotal':    'Subtotal Price',
        'lbl_gm':          'Gross Margin Rate',
        'lbl_grant':       'Energy Grant Amount',
        'chk_a':           'Have Competitor A data',
        'chk_b':           'Have Competitor B data',
        'chk_c':           'Have Competitor C data',
        'btn_predict':     'Predict Win Rate',
        'sec_result':      'Prediction Results',
        'gauge_title':     'Overall Win Probability',
        'hist_product':    'Product {p} Historical: {q}  |  Win Rate {wr}  |  Avg. Margin {gm}',
        'msg_high':        'Average win probability {p} across all models — recommend pursuing actively.',
        'msg_mid':         'Average win probability {p} — consider adjusting the quote.',
        'msg_low':         'Average win probability {p} — low chance of winning, recommend reassessing strategy.',
        # BCG page
        'bcg_title':       'BCG Product Strategy Analysis',
        'bcg_subtitle':    'Analyze product lines by Gross Margin (profitability) × Win Rate (competitiveness)',
        'q_star_title':    'Star',
        'q_star_desc':     'High margin × High win rate\n\nPrioritize & scale',
        'q_qm_title':      'Question Mark',
        'q_qm_desc':       'High margin × Low win rate\n\nAnalyze barriers to conversion',
        'q_cc_title':      'Cash Cow',
        'q_cc_desc':       'Low margin × High win rate\n\nExplore price increases',
        'q_dog_title':     'Dog',
        'q_dog_desc':      'Low margin × Low win rate\n\nReconsider resource allocation',
        'filter_q':        'Filter Quadrant',
        'filter_n':        'Minimum Sample Size',
        'chart_title':     'BCG Matrix (Bubble size = Number of quotes)',
        'x_axis':          'Win Rate',
        'y_axis':          'Avg. Gross Margin',
        'sec_table':       'Product Summary',
        'col_product':     'Product',
        'col_quadrant':    'Quadrant',
        'col_gm':          'Avg. Gross Margin',
        'col_wr':          'Win Rate',
        'col_n':           'No. of Quotes',
        'col_rev':         'Total Revenue',
        'sec_deep':        'Product Deep Dive',
        'sel_product':     'Select Product',
        'metric_q':        'Quadrant',
        'metric_gm':       'Avg. Gross Margin',
        'metric_wr':       'Win Rate',
        'metric_n':        'No. of Quotes',
        'hist_chart':      '{p}: Gross Margin Distribution (Green=Won  Red=Lost)',
        'box_chart':       '{p}: Price Gap Distribution',
        'lbl_result':      'Outcome',
        'lbl_gm_x':        'Gross Margin Rate',
        'lbl_result_0':    'Outcome (0=Won)',
        'lbl_price_diff':  'Price Gap vs. Competitor',
        'strategy_label':  'Strategic Recommendation ({q}): {s}',
        'strategy': {
            'Star':          'This product has both high profitability and strong competitiveness. Maintain current pricing and scale sales volume.',
            'Question Mark': 'Good margin but low conversion. Investigate competitor conditions and evaluate whether a price adjustment or enhanced grant offer could improve win rate.',
            'Cash Cow':      'High market acceptance but thin margins. Explore moderate price increases without significantly impacting win rate.',
            'Dog':           'Both margin and win rate are low. Review resource allocation for this product line and consider reducing quoting effort or repositioning.',
        },
        'q_label': {
            'Star':          'Star (High margin × High win rate)',
            'Question Mark': 'Question Mark (High margin × Low win rate)',
            'Cash Cow':      'Cash Cow (Low margin × High win rate)',
            'Dog':           'Dog (Low margin × Low win rate)',
        },
    },
}

# ══════════════════════════════════════════════
#  全域 CSS
# ══════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #EAF4F4; }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D0E8E8;
    }
    button[data-testid="stBaseButton-headerNoPadding"] > div > svg,
    button[data-testid="stSidebarNavToggleButton"] > div > svg,
    [data-testid="collapsedControl"] svg { display: none !important; }
    button[data-testid="stBaseButton-headerNoPadding"]::before,
    button[data-testid="stSidebarNavToggleButton"]::before,
    [data-testid="collapsedControl"]::before {
        content: "☰"; font-size: 1.3rem; color: #2A7F7F; font-weight: 400;
    }
    .page-title {
        font-size: 1.7rem; font-weight: 700; color: #1A5C5C;
        margin-bottom: 4px; letter-spacing: -0.3px;
    }
    .page-subtitle { font-size: 0.95rem; color: #6B8F8F; margin-bottom: 24px; }
    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #2A7F7F;
        margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #C2E0E0;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #2A9D8F; color: white; border: none;
        border-radius: 8px; font-size: 1rem; font-weight: 600;
        padding: 0.6rem 1.2rem; letter-spacing: 0.3px;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #21867A; }
    [data-testid="stMetric"] {
        background: #F4FAFA; border-radius: 10px;
        padding: 14px 18px; border: 1px solid #D4EEEE;
    }
    [data-testid="stMetricLabel"] { color: #6B8F8F; font-size: 0.85rem; }
    [data-testid="stMetricValue"] { color: #1A5C5C; font-weight: 700; }
    hr { border-color: #D0E8E8; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

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
    pct = df.groupby('Product')['highest_competitor_per_row'].transform('max').mul(1.5)
    gfb = df['Unit Price'].median() * 1.5
    df['rational_imputed_price']      = pct.fillna(gfb)
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
        'Kw','Qty','Subtotal Price','Gross Margin Rate','Energy grant amount','Grant_Ratio',
        'competitor_count','no_competitor_info','Diff_A','Diff_B','Diff_C','price_diff','price_ratio'
    ]
    X = df[features]; y = df['Convert_to_Order']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_idx, test_idx = y_train.index, y_test.index
    train_full = df.loc[train_idx].copy()
    product_ceiling = train_full.groupby('Product')['highest_competitor_per_row'].max().mul(1.5)
    global_fallback = train_full['Unit Price'].median() * 1.5
    for idx_set, full_set in [(train_idx, train_full), (test_idx, df.loc[test_idx].copy())]:
        imputed = full_set['Product'].map(product_ceiling).fillna(global_fallback)
        df.loc[idx_set,'rational_imputed_price']      = imputed.values
        df.loc[idx_set,'min_competitor_price_filled'] = df.loc[idx_set,'min_competitor_price'].fillna(imputed).values
        df.loc[idx_set,'price_diff']  = (df.loc[idx_set,'Unit Price'] - df.loc[idx_set,'min_competitor_price_filled']).values
        df.loc[idx_set,'price_ratio'] = (df.loc[idx_set,'Unit Price'] / df.loc[idx_set,'min_competitor_price_filled']).values
    X = df[features]; X_train = X.loc[train_idx]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.replace([np.inf,-np.inf],np.nan).fillna(0))
    neg_count, pos_count = Counter(y_train)[0], Counter(y_train)[1]
    lr  = LogisticRegression(max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                  min_samples_split=10, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(eval_metric='logloss', n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                         scale_pos_weight=neg_count/pos_count, random_state=42)
    lr.fit(X_train_s, y_train); rf.fit(X_train_s, y_train); xgb.fit(X_train_s, y_train)
    return dict(lr=lr, rf=rf, xgb=xgb, scaler=scaler, features=features,
                product_ceiling=product_ceiling, global_fallback=global_fallback,
                product_list=sorted(df['Product'].dropna().unique().tolist()))

@st.cache_data
def load_bcg_data():
    df_raw = pd.read_excel(XLSX_PATH, sheet_name='raw data')
    df_raw.columns = df_raw.columns.str.strip()
    df = build_features(df_raw.drop_duplicates())
    bcg_df = df.groupby('Product').agg(
        avg_gross_margin=('Gross Margin Rate','mean'),
        win_rate=('Convert_to_Order', lambda x: (x==0).mean()),
        sample_size=('Convert_to_Order','count'),
        avg_price_diff=('price_diff','mean'),
        total_revenue=('Subtotal Price','sum'),
    ).reset_index()
    gm_med = bcg_df['avg_gross_margin'].median()
    wr_med = bcg_df['win_rate'].median()
    def quad(row):
        hg = row['avg_gross_margin'] >= gm_med
        hw = row['win_rate'] >= wr_med
        if hg and hw: return 'Star'
        if hg: return 'Question Mark'
        if hw: return 'Cash Cow'
        return 'Dog'
    bcg_df['Quadrant'] = bcg_df.apply(quad, axis=1)
    return bcg_df, gm_med, wr_med, df

def preprocess_input(ckpt, product, kw, qty, unit_price, subtotal,
                     gross_margin, energy_grant, comp_a, comp_b, comp_c):
    features = ckpt['features']; scaler = ckpt['scaler']
    product_ceiling = ckpt['product_ceiling']; global_fallback = ckpt['global_fallback']
    comp_notna = [v for v in [comp_a,comp_b,comp_c] if v is not None]
    competitor_count   = len(comp_notna)
    no_competitor_info = 1 if competitor_count == 0 else 0
    min_comp = min(comp_notna) if comp_notna else None
    min_comp_filled = min_comp if min_comp is not None else product_ceiling.get(product, global_fallback)
    price_diff  = unit_price - min_comp_filled
    price_ratio = (unit_price / min_comp_filled) if min_comp_filled != 0 else 0
    grant_ratio = (energy_grant / subtotal) if subtotal != 0 else 0
    if not np.isfinite(grant_ratio): grant_ratio = 0
    row = {
        'Kw':kw,'Qty':qty,'Subtotal Price':subtotal,'Gross Margin Rate':gross_margin,
        'Energy grant amount':energy_grant,'Grant_Ratio':grant_ratio,
        'competitor_count':competitor_count,'no_competitor_info':no_competitor_info,
        'Diff_A':(unit_price-comp_a) if comp_a is not None else np.nan,
        'Diff_B':(unit_price-comp_b) if comp_b is not None else np.nan,
        'Diff_C':(unit_price-comp_c) if comp_c is not None else np.nan,
        'price_diff':price_diff,'price_ratio':price_ratio,
    }
    return scaler.transform(pd.DataFrame([row])[features].replace([np.inf,-np.inf],np.nan).fillna(0))

# ══════════════════════════════════════════════
#  App 入口
# ══════════════════════════════════════════════
st.set_page_config(page_title="Sales Analytics", page_icon="📊", layout="wide")
inject_css()

with st.spinner("Initializing..."):
    ckpt = train_models()

lr, rf, xgb  = ckpt['lr'], ckpt['rf'], ckpt['xgb']
product_list = ckpt['product_list']
bcg_df, gm_med, wr_med, raw_df = load_bcg_data()

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    # 語言切換
    lang = st.radio("Language / 語言", ["中文", "English"],
                    horizontal=True, label_visibility="collapsed")
    L = TEXT['zh'] if lang == "中文" else TEXT['en']

    st.markdown(f"### {L['app_title']}")
    st.markdown("---")
    page = st.radio("", [L['nav_predict'], L['nav_bcg']], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"{L['model_perf']}\nLR  AUC 0.856\nRF  AUC 0.944\nXGB AUC 0.951")

color_map = {'Star':'#2A9D8F','Question Mark':'#457B9D','Cash Cow':'#E9C46A','Dog':'#E76F51'}

# ══════════════════════════════════════════════
#  頁面一：預測
# ══════════════════════════════════════════════
if page == L['nav_predict']:
    st.markdown(f'<p class="page-title">{L["pred_title"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{L["pred_subtitle"]}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f'<p class="section-title">{L["sec_product"]}</p>', unsafe_allow_html=True)
        product      = st.selectbox(L['lbl_product'], product_list)
        kw           = st.number_input(L['lbl_kw'],  min_value=1, max_value=2000, value=75)
        qty          = st.number_input(L['lbl_qty'], min_value=1, max_value=100, value=1)
    with col2:
        st.markdown(f'<p class="section-title">{L["sec_price"]}</p>', unsafe_allow_html=True)
        unit_price   = st.number_input(L['lbl_unit_price'], min_value=1, value=700000, step=1000)
        subtotal     = st.number_input(L['lbl_subtotal'],   min_value=1, value=700000, step=1000)
        gross_margin = st.number_input(L['lbl_gm'], min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.4f")
        energy_grant = st.number_input(L['lbl_grant'], min_value=0, value=375000, step=1000)

    st.markdown(f'<p class="section-title" style="margin-top:8px">{L["sec_competitor"]}</p>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3, gap="large")
    with col3:
        use_a = st.checkbox(L['chk_a'])
        comp_a_val = st.number_input("A", min_value=0, value=0, step=1000, disabled=not use_a, label_visibility="collapsed")
    with col4:
        use_b = st.checkbox(L['chk_b'])
        comp_b_val = st.number_input("B", min_value=0, value=0, step=1000, disabled=not use_b, label_visibility="collapsed")
    with col5:
        use_c = st.checkbox(L['chk_c'])
        comp_c_val = st.number_input("C", min_value=0, value=0, step=1000, disabled=not use_c, label_visibility="collapsed")

    comp_a = comp_a_val if use_a else None
    comp_b = comp_b_val if use_b else None
    comp_c = comp_c_val if use_c else None

    prod_row = bcg_df[bcg_df['Product'] == product]
    if not prod_row.empty:
        r = prod_row.iloc[0]
        st.info(L['hist_product'].format(p=product, q=L['q_label'][r['Quadrant']],
                                          wr=f"{r['win_rate']:.1%}", gm=f"{r['avg_gross_margin']:.1%}"))

    st.markdown("")
    if st.button(L['btn_predict'], use_container_width=True, type="primary"):
        X_input  = preprocess_input(ckpt, product, kw, qty, unit_price, subtotal,
                                    gross_margin, energy_grant, comp_a, comp_b, comp_c)
        prob_lr  = lr.predict_proba(X_input)[0][0]
        prob_rf  = rf.predict_proba(X_input)[0][0]
        prob_xgb = xgb.predict_proba(X_input)[0][0]
        avg_prob = (prob_lr + prob_rf + prob_xgb) / 3

        st.markdown("---")
        st.markdown(f'<p class="section-title">{L["sec_result"]}</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Logistic Regression", f"{prob_lr:.1%}")
        c2.metric("Random Forest",       f"{prob_rf:.1%}")
        c3.metric("XGBoost  ★",          f"{prob_xgb:.1%}")

        bar_color = "#2A9D8F" if avg_prob >= 0.6 else "#E9C46A" if avg_prob >= 0.4 else "#E76F51"
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=avg_prob*100,
            title={'text': L['gauge_title'], 'font': {'size': 16, 'color': '#1A5C5C'}},
            number={'suffix': '%', 'font': {'size': 44, 'color': '#1A5C5C'}},
            gauge={
                'axis': {'range': [0,100], 'tickcolor': '#6B8F8F'},
                'bar': {'color': bar_color}, 'bgcolor': '#F4FAFA', 'bordercolor': '#D4EEEE',
                'steps': [{'range':[0,40],'color':'#FAE8E4'},{'range':[40,60],'color':'#FDF6E3'},
                           {'range':[60,100],'color':'#E4F5F2'}],
            }
        ))
        fig.update_layout(height=280, margin=dict(t=50,b=10,l=30,r=30),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        p_str = f"{avg_prob:.1%}"
        if avg_prob >= 0.6:   st.success(L['msg_high'].format(p=p_str))
        elif avg_prob >= 0.4: st.warning(L['msg_mid'].format(p=p_str))
        else:                 st.error(L['msg_low'].format(p=p_str))

# ══════════════════════════════════════════════
#  頁面二：BCG
# ══════════════════════════════════════════════
else:
    st.markdown(f'<p class="page-title">{L["bcg_title"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{L["bcg_subtitle"]}</p>', unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4, gap="small")
    q1.success(f"**{L['q_star_title']}**\n\n{L['q_star_desc']}")
    q2.info(f"**{L['q_qm_title']}**\n\n{L['q_qm_desc']}")
    q3.warning(f"**{L['q_cc_title']}**\n\n{L['q_cc_desc']}")
    q4.error(f"**{L['q_dog_title']}**\n\n{L['q_dog_desc']}")

    st.markdown("---")
    f1, f2 = st.columns([2,1])
    with f1:
        q_filter = st.multiselect(L['filter_q'], ['Star','Question Mark','Cash Cow','Dog'],
                                   default=['Star','Question Mark','Cash Cow','Dog'])
    with f2:
        min_n = st.slider(L['filter_n'], 1, 100, 5)

    filtered = bcg_df[(bcg_df['Quadrant'].isin(q_filter)) & (bcg_df['sample_size'] >= min_n)]

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
            hovertemplate="<b>%{text}</b><br>"+L['x_axis']+": %{x:.1%}<br>"+L['y_axis']+
                          ": %{y:.1%}<br>"+L['col_n']+": %{customdata[0]}<extra></extra>"
        ))
    fig.add_vline(x=wr_med,  line_dash="dot", line_color="#A0C4C4", line_width=1.5)
    fig.add_hline(y=gm_med, line_dash="dot", line_color="#A0C4C4", line_width=1.5)
    fig.update_xaxes(title=L['x_axis'], tickformat=".0%", gridcolor="#E8F4F4")
    fig.update_yaxes(title=L['y_axis'],  tickformat=".0%", gridcolor="#E8F4F4")
    fig.update_layout(
        title=dict(text=L['chart_title'], font=dict(size=15, color='#1A5C5C')),
        height=560, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE',
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        hovermode='closest', font=dict(color='#2C4A4A')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<p class="section-title">{L["sec_table"]}</p>', unsafe_allow_html=True)
    disp = filtered[['Product','Quadrant','avg_gross_margin','win_rate','sample_size','total_revenue']].copy()
    disp = disp.sort_values('total_revenue', ascending=False).reset_index(drop=True)
    disp.columns = [L['col_product'],L['col_quadrant'],L['col_gm'],L['col_wr'],L['col_n'],L['col_rev']]
    st.dataframe(
        disp.style.format({L['col_gm']:'{:.1%}', L['col_wr']:'{:.1%}', L['col_rev']:'{:,.0f}'}),
        use_container_width=True, height=380
    )

    st.markdown("---")
    st.markdown(f'<p class="section-title">{L["sec_deep"]}</p>', unsafe_allow_html=True)
    sel = st.selectbox(L['sel_product'], filtered['Product'].tolist())
    pi  = filtered[filtered['Product']==sel].iloc[0]
    pr  = raw_df[raw_df['Product']==sel]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(L['metric_q'],  pi['Quadrant'])
    m2.metric(L['metric_gm'], f"{pi['avg_gross_margin']:.1%}")
    m3.metric(L['metric_wr'], f"{pi['win_rate']:.1%}")
    m4.metric(L['metric_n'],  int(pi['sample_size']))

    h1, h2 = st.columns(2)
    with h1:
        fig_h = px.histogram(pr, x='Gross Margin Rate', color='Convert_to_Order',
                              color_discrete_map={0:'#2A9D8F',1:'#E76F51'},
                              labels={'Convert_to_Order':L['lbl_result'],'Gross Margin Rate':L['lbl_gm_x']},
                              title=L['hist_chart'].format(p=sel), barmode='overlay', opacity=0.75)
        fig_h.update_xaxes(tickformat=".0%")
        fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE', font=dict(color='#2C4A4A'))
        st.plotly_chart(fig_h, use_container_width=True)
    with h2:
        fig_b = px.box(pr, x='Convert_to_Order', y='price_diff', color='Convert_to_Order',
                        color_discrete_map={0:'#2A9D8F',1:'#E76F51'},
                        labels={'Convert_to_Order':L['lbl_result_0'],'price_diff':L['lbl_price_diff']},
                        title=L['box_chart'].format(p=sel))
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FAFEFE', font=dict(color='#2C4A4A'))
        st.plotly_chart(fig_b, use_container_width=True)

    st.info(L['strategy_label'].format(q=pi['Quadrant'], s=L['strategy'][pi['Quadrant']]))

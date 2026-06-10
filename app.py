import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ══════════════════════════════════════════════
#  載入模型 & 資料
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    with open('model_checkpoint.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_bcg_data():
    df = pd.read_excel('Gross_Profit_vs_Competitor_0514_Copliot.xlsx', sheet_name='raw data')
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    comp_cols = ['Competitor A', 'Competitor B', 'Competitor C']
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

    bcg_df = df.groupby('Product').agg(
        avg_gross_margin = ('Gross Margin Rate', 'mean'),
        win_rate         = ('Convert_to_Order', lambda x: (x == 0).mean()),
        sample_size      = ('Convert_to_Order', 'count'),
        avg_price_diff   = ('price_diff', 'mean'),
        avg_grant_ratio  = ('Grant_Ratio', 'mean'),
        total_revenue    = ('Subtotal Price', 'sum'),
    ).reset_index()

    gm_median = bcg_df['avg_gross_margin'].median()
    wr_median = bcg_df['win_rate'].median()

    def assign_quadrant(row):
        high_gm = row['avg_gross_margin'] >= gm_median
        high_wr = row['win_rate'] >= wr_median
        if high_gm and high_wr:        return 'Star ⭐'
        elif high_gm and not high_wr:  return 'Question Mark ❓'
        elif not high_gm and high_wr:  return 'Cash Cow 🐄'
        else:                           return 'Dog 🐕'

    bcg_df['Quadrant'] = bcg_df.apply(assign_quadrant, axis=1)
    return bcg_df, gm_median, wr_median, df

# ══════════════════════════════════════════════
#  前處理（預測用）
# ══════════════════════════════════════════════
def preprocess(ckpt, product, kw, qty, unit_price, subtotal_price,
               gross_margin_rate, energy_grant, comp_a, comp_b, comp_c):
    features        = ckpt['features']
    scaler          = ckpt['scaler']
    product_ceiling = ckpt['product_ceiling']
    global_fallback = ckpt['global_fallback']

    comp_notna = [v for v in [comp_a, comp_b, comp_c] if v is not None]
    competitor_count   = len(comp_notna)
    no_competitor_info = 1 if competitor_count == 0 else 0
    min_comp = min(comp_notna) if comp_notna else None

    if min_comp is None:
        ceiling = product_ceiling.get(product, None)
        min_comp_filled = ceiling if ceiling is not None else global_fallback
    else:
        min_comp_filled = min_comp

    price_diff  = unit_price - min_comp_filled
    price_ratio = unit_price / min_comp_filled if min_comp_filled != 0 else 0
    grant_ratio = energy_grant / subtotal_price if subtotal_price != 0 else 0
    if not np.isfinite(grant_ratio):
        grant_ratio = 0

    diff_a = (unit_price - comp_a) if comp_a is not None else np.nan
    diff_b = (unit_price - comp_b) if comp_b is not None else np.nan
    diff_c = (unit_price - comp_c) if comp_c is not None else np.nan

    row = {
        'Kw': kw, 'Qty': qty, 'Subtotal Price': subtotal_price,
        'Gross Margin Rate': gross_margin_rate,
        'Energy grant amount': energy_grant, 'Grant_Ratio': grant_ratio,
        'competitor_count': competitor_count, 'no_competitor_info': no_competitor_info,
        'Diff_A': diff_a, 'Diff_B': diff_b, 'Diff_C': diff_c,
        'price_diff': price_diff, 'price_ratio': price_ratio,
    }
    df_row = pd.DataFrame([row])[features]
    df_row = df_row.replace([np.inf, -np.inf], np.nan).fillna(0)
    return scaler.transform(df_row)

# ══════════════════════════════════════════════
#  頁面設定
# ══════════════════════════════════════════════
st.set_page_config(page_title="銷售分析系統", page_icon="📊", layout="wide")

ckpt = load_model()
lr, rf, xgb  = ckpt['lr'], ckpt['rf'], ckpt['xgb']
product_list = ckpt['product_list']
bcg_df, gm_median, wr_median, raw_df = load_bcg_data()

# ── Sidebar 導航 ──────────────────────────────
st.sidebar.title("📊 銷售分析系統")
page = st.sidebar.radio("選擇功能", ["🔍 接單成功率預測", "📈 BCG 產品策略分析"])
st.sidebar.divider()
st.sidebar.caption("模型效能：LR AUC 0.856 ｜ RF AUC 0.944 ｜ XGB AUC 0.951")

# ══════════════════════════════════════════════
#  頁面 1：預測
# ══════════════════════════════════════════════
if page == "🔍 接單成功率預測":
    st.title("🔍 接單成功率預測")
    st.markdown("輸入報價資訊，預測接單成功機率")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 產品資訊")
        product   = st.selectbox("產品型號", product_list)
        kw        = st.number_input("功率 Kw", min_value=1, max_value=2000, value=75)
        qty       = st.number_input("數量 Qty", min_value=1, max_value=100, value=1)

    with col2:
        st.subheader("💰 價格資訊")
        unit_price   = st.number_input("單價 Unit Price", min_value=1, value=700000, step=1000)
        subtotal     = st.number_input("小計 Subtotal Price", min_value=1, value=700000, step=1000)
        gross_margin = st.number_input("毛利率 Gross Margin Rate", min_value=0.0, max_value=1.0,
                                       value=0.35, step=0.01, format="%.4f")
        energy_grant = st.number_input("能源補助金額", min_value=0, value=375000, step=1000)

    st.subheader("🏢 競爭對手報價（不知道可留空）")
    col3, col4, col5 = st.columns(3)
    with col3:
        use_a = st.checkbox("有 Competitor A 資料")
        comp_a_val = st.number_input("Competitor A", min_value=0, value=0, step=1000, disabled=not use_a)
    with col4:
        use_b = st.checkbox("有 Competitor B 資料")
        comp_b_val = st.number_input("Competitor B", min_value=0, value=0, step=1000, disabled=not use_b)
    with col5:
        use_c = st.checkbox("有 Competitor C 資料")
        comp_c_val = st.number_input("Competitor C", min_value=0, value=0, step=1000, disabled=not use_c)

    comp_a = comp_a_val if use_a else None
    comp_b = comp_b_val if use_b else None
    comp_c = comp_c_val if use_c else None

    # 顯示此產品 BCG 象限
    prod_row = bcg_df[bcg_df['Product'] == product]
    if not prod_row.empty:
        q = prod_row.iloc[0]['Quadrant']
        wr = prod_row.iloc[0]['win_rate']
        gm = prod_row.iloc[0]['avg_gross_margin']
        st.info(f"📌 產品 **{product}** 歷史表現：象限 **{q}**　｜　歷史勝率 {wr:.1%}　｜　平均毛利率 {gm:.1%}")

    st.divider()

    if st.button("🔍 預測接單成功率", use_container_width=True, type="primary"):
        X_input = preprocess(ckpt, product, kw, qty, unit_price, subtotal,
                             gross_margin, energy_grant, comp_a, comp_b, comp_c)

        prob_lr  = lr.predict_proba(X_input)[0][0]
        prob_rf  = rf.predict_proba(X_input)[0][0]
        prob_xgb = xgb.predict_proba(X_input)[0][0]
        avg_prob = (prob_lr + prob_rf + prob_xgb) / 3

        st.subheader("📈 預測結果")
        c1, c2, c3 = st.columns(3)

        def render_metric(col, name, prob):
            delta = "高機率 ✅" if prob >= 0.6 else "中等 ⚠️" if prob >= 0.4 else "低機率 ❌"
            col.metric(label=name, value=f"{prob:.1%}", delta=delta)

        render_metric(c1, "Logistic Regression", prob_lr)
        render_metric(c2, "Random Forest",       prob_rf)
        render_metric(c3, "XGBoost ★ 最佳",      prob_xgb)

        # 儀表板
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob * 100,
            title={'text': "綜合接單成功率", 'font': {'size': 20}},
            number={'suffix': '%', 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ecc71" if avg_prob >= 0.6 else "#e67e22" if avg_prob >= 0.4 else "#e74c3c"},
                'steps': [
                    {'range': [0, 40],  'color': '#fdecea'},
                    {'range': [40, 60], 'color': '#fff3e0'},
                    {'range': [60, 100],'color': '#e8f5e9'},
                ],
                'threshold': {'line': {'color': 'black', 'width': 3}, 'value': avg_prob * 100}
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        if avg_prob >= 0.6:
            st.success(f"🎉 綜合三模型平均接單成功率 **{avg_prob:.1%}**，建議積極跟進！")
        elif avg_prob >= 0.4:
            st.warning(f"🤔 綜合三模型平均接單成功率 **{avg_prob:.1%}**，建議評估調整報價。")
        else:
            st.error(f"😔 綜合三模型平均接單成功率 **{avg_prob:.1%}**，接單困難，建議重新評估策略。")

# ══════════════════════════════════════════════
#  頁面 2：BCG 分析
# ══════════════════════════════════════════════
else:
    st.title("📈 BCG 產品策略分析")
    st.markdown("以**毛利率**（獲利能力）× **接單轉換率**（市場競爭力）分析各產品線策略定位")
    st.divider()

    # ── 象限說明 ─────────────────────────────
    col_s, col_q, col_c, col_d = st.columns(4)
    col_s.success("⭐ **Star**\n\n高毛利 × 高轉換\n\n擴大銷售，優先投入")
    col_q.info("❓ **Question Mark**\n\n高毛利 × 低轉換\n\n分析競爭原因，嘗試提升成交率")
    col_c.warning("🐄 **Cash Cow**\n\n低毛利 × 高轉換\n\n嘗試適度調價提升獲利")
    col_d.error("🐕 **Dog**\n\n低毛利 × 低轉換\n\n重新評估資源投入")

    st.divider()

    # ── 篩選器 ───────────────────────────────
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        quadrant_filter = st.multiselect(
            "篩選象限",
            options=['Star ⭐', 'Question Mark ❓', 'Cash Cow 🐄', 'Dog 🐕'],
            default=['Star ⭐', 'Question Mark ❓', 'Cash Cow 🐄', 'Dog 🐕']
        )
    with col_f2:
        min_sample = st.slider("最少樣本數", min_value=1, max_value=100, value=5)

    filtered = bcg_df[
        (bcg_df['Quadrant'].isin(quadrant_filter)) &
        (bcg_df['sample_size'] >= min_sample)
    ]

    # ── BCG 氣泡圖 ────────────────────────────
    color_map = {
        'Star ⭐':          '#1565C0',
        'Question Mark ❓': '#2E7D32',
        'Cash Cow 🐄':      '#E65100',
        'Dog 🐕':           '#C62828',
    }

    fig = go.Figure()

    for quadrant, grp in filtered.groupby('Quadrant'):
        fig.add_trace(go.Scatter(
            x=grp['win_rate'],
            y=grp['avg_gross_margin'],
            mode='markers+text',
            name=quadrant,
            text=grp['Product'],
            textposition='top center',
            marker=dict(
                size=np.sqrt(grp['sample_size']) * 4,
                color=color_map[quadrant],
                opacity=0.75,
                line=dict(color='white', width=2)
            ),
            customdata=np.stack([
                grp['sample_size'],
                grp['total_revenue'],
                grp['avg_price_diff'],
            ], axis=-1),
            hovertemplate=(
                "<b>產品 %{text}</b><br>"
                "轉換率：%{x:.1%}<br>"
                "毛利率：%{y:.1%}<br>"
                "樣本數：%{customdata[0]}<br>"
                "總營收：%{customdata[1]:,.0f}<br>"
                "平均價差：%{customdata[2]:,.0f}<extra></extra>"
            )
        ))

    # 分界線
    fig.add_vline(x=wr_median, line_dash="dash", line_color="gray", opacity=0.6)
    fig.add_hline(y=gm_median, line_dash="dash", line_color="gray", opacity=0.6)

    # 象限標籤
    fig.add_annotation(x=0.01, y=bcg_df['avg_gross_margin'].max() * 0.98,
                       text="Question Mark ❓", showarrow=False, font=dict(color='#2E7D32', size=11))
    fig.add_annotation(x=bcg_df['win_rate'].max() * 0.95, y=bcg_df['avg_gross_margin'].max() * 0.98,
                       text="Star ⭐", showarrow=False, font=dict(color='#1565C0', size=11))
    fig.add_annotation(x=0.01, y=bcg_df['avg_gross_margin'].min() * 1.05,
                       text="Dog 🐕", showarrow=False, font=dict(color='#C62828', size=11))
    fig.add_annotation(x=bcg_df['win_rate'].max() * 0.95, y=bcg_df['avg_gross_margin'].min() * 1.05,
                       text="Cash Cow 🐄", showarrow=False, font=dict(color='#E65100', size=11))

    fig.update_xaxes(title="接單轉換率 (Win Rate)", tickformat=".0%")
    fig.update_yaxes(title="平均毛利率 (Gross Margin Rate)", tickformat=".0%")
    fig.update_layout(
        title=dict(text="BCG 矩陣：產品策略定位<br><sup>氣泡大小反映詢價筆數</sup>", font=dict(size=16)),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 產品明細表 ────────────────────────────
    st.subheader("📋 產品明細")
    display_df = filtered[['Product', 'Quadrant', 'avg_gross_margin', 'win_rate',
                            'sample_size', 'total_revenue', 'avg_price_diff']].copy()
    display_df = display_df.sort_values('total_revenue', ascending=False).reset_index(drop=True)
    display_df.columns = ['產品', '象限', '平均毛利率', '轉換率', '詢價筆數', '總營收', '平均價差']

    st.dataframe(
        display_df.style.format({
            '平均毛利率': '{:.1%}',
            '轉換率':    '{:.1%}',
            '總營收':    '{:,.0f}',
            '平均價差':  '{:,.0f}',
        }),
        use_container_width=True,
        height=400
    )

    # ── 點選產品查看詳情 ──────────────────────
    st.divider()
    st.subheader("🔎 單一產品深入分析")
    selected_product = st.selectbox("選擇產品", filtered['Product'].tolist())

    prod_info = filtered[filtered['Product'] == selected_product].iloc[0]
    prod_raw  = raw_df[raw_df['Product'] == selected_product]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("象限",     prod_info['Quadrant'])
    m2.metric("平均毛利率", f"{prod_info['avg_gross_margin']:.1%}")
    m3.metric("接單轉換率", f"{prod_info['win_rate']:.1%}")
    m4.metric("詢價筆數",  int(prod_info['sample_size']))

    # 毛利率分布
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_hist = px.histogram(
            prod_raw, x='Gross Margin Rate',
            color='Convert_to_Order',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'Convert_to_Order': '結果', 'Gross Margin Rate': '毛利率'},
            title=f"{selected_product}：毛利率分布（綠=成交 紅=未成交）",
            barmode='overlay', opacity=0.7
        )
        fig_hist.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        fig_box = px.box(
            prod_raw, x='Convert_to_Order', y='price_diff',
            color='Convert_to_Order',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'Convert_to_Order': '結果（0=成交）', 'price_diff': '與競爭者價差'},
            title=f"{selected_product}：價差分布（成交 vs 未成交）"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 策略建議
    strategy_map = {
        'Star ⭐':          '維持現有定價策略，擴大銷售量。此產品線同時具備高獲利與高競爭力，是最優先投入資源的方向。',
        'Cash Cow 🐄':      '轉換率高代表市場接受度佳，但毛利率偏低。建議檢視是否有適度調價空間，在不大幅犧牲成交率的前提下提高獲利能力。',
        'Question Mark ❓': '毛利率高代表獲利能力佳，但轉換率偏低。建議分析競爭者價格與補助方案，評估是否透過小幅降價或提高價值主張來提升成交率。',
        'Dog 🐕':           '毛利與轉換率雙低。建議重新檢視產品定位、競爭環境與資源投入效益，考慮是否縮減此產品線的報價資源。',
    }
    q = prod_info['Quadrant']
    st.info(f"💡 **策略建議（{q}）**\n\n{strategy_map[q]}")

# ════════════════════════════════════════════════════════════════════
# app.py — Online Gaming Behavior ML — Streamlit GUI
# Run: streamlit run app.py
# ════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎮 Gaming Behavior ML",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
MODELS_DIR   = "saved_models"
DATA_PATH    = "online_gaming_behavior_dataset.csv"
CLASS_NAMES  = ['Low', 'Medium', 'High']
PALETTE      = {'Low': '#ef5350', 'Medium': '#42a5f5', 'High': '#66bb6a'}
CMAP         = {0: '#ef5350', 1: '#42a5f5', 2: '#66bb6a'}
LMAP         = {0: 'Low', 1: 'Medium', 2: 'High'}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2.4rem; font-weight:800; color:#1B4F72; margin-bottom:0; }
    .sub-title    { font-size:1.1rem; color:#5D6D7E; margin-top:0; }
    .metric-card  { background:#EBF5FB; border-left:5px solid #2E86AB;
                    padding:16px 20px; border-radius:8px; margin:6px 0; }
    .metric-val   { font-size:2rem; font-weight:800; color:#1B4F72; }
    .metric-label { font-size:.85rem; color:#5D6D7E; }
    .step-box     { background:#F0F4F8; border-left:4px solid #27AE60;
                    padding:10px 16px; border-radius:6px; margin:4px 0; }
    .warn-box     { background:#FEF9E7; border-left:4px solid #F39C12;
                    padding:10px 16px; border-radius:6px; margin:4px 0; }
    .info-box     { background:#EBF5FB; border-left:4px solid #2980B9;
                    padding:10px 16px; border-radius:6px; margin:4px 0; }
    .pred-high    { background:#D5F5E3; border:2px solid #27AE60; padding:20px;
                    border-radius:10px; text-align:center; font-size:1.6rem;
                    font-weight:800; color:#1E8449; }
    .pred-medium  { background:#EBF5FB; border:2px solid #2980B9; padding:20px;
                    border-radius:10px; text-align:center; font-size:1.6rem;
                    font-weight:800; color:#1B4F72; }
    .pred-low     { background:#FDEDEC; border:2px solid #E74C3C; padding:20px;
                    border-radius:10px; text-align:center; font-size:1.6rem;
                    font-weight:800; color:#922B21; }
    hr { border:none; border-top:2px solid #D5D8DC; margin:20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_models():
    """Load saved models or train fresh if not found."""
    if not os.path.exists(MODELS_DIR):
        return None, None, None, None, None, None, None

    try:
        scaler         = joblib.load(f'{MODELS_DIR}/scaler.pkl')
        mms            = joblib.load(f'{MODELS_DIR}/minmax_scaler.pkl')
        pca            = joblib.load(f'{MODELS_DIR}/pca.pkl')
        lda            = joblib.load(f'{MODELS_DIR}/lda.pkl')
        final_features = joblib.load(f'{MODELS_DIR}/final_features.pkl')
        all_columns    = joblib.load(f'{MODELS_DIR}/all_columns.pkl')

        all_results = {}
        ds_names  = ['Selected_Features', 'PCA', 'LDA']
        m_names   = ['Random_Forest', 'SVM', 'KNN']
        label_map = {'Selected_Features': 'Selected Features', 'PCA': 'PCA', 'LDA': 'LDA'}
        mname_map = {'Random_Forest': 'Random Forest', 'SVM': 'SVM', 'KNN': 'KNN'}

        for ds in ds_names:
            all_results[label_map[ds]] = {}
            for m in m_names:
                path = f'{MODELS_DIR}/{ds}_{m}.pkl'
                if os.path.exists(path):
                    all_results[label_map[ds]][mname_map[m]] = joblib.load(path)

        return scaler, mms, pca, lda, final_features, all_columns, all_results
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None

@st.cache_data
def build_pipeline(_df):
    """Full pipeline — runs once and caches everything."""
    df_proc = _df.copy()
    df_proc.drop('PlayerID', axis=1, inplace=True)
    df_proc['EngagementLevel'] = df_proc['EngagementLevel'].map({'Low':0,'Medium':1,'High':2})
    cat_cols = ['Gender','Location','GameGenre','GameDifficulty']
    df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)

    X = df_proc.drop('EngagementLevel', axis=1)
    y = df_proc['EngagementLevel']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),       columns=X.columns)

    mms = MinMaxScaler()
    X_train_mm = pd.DataFrame(mms.fit_transform(X_train), columns=X.columns)

    return X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X_train_mm, scaler, mms

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Gaming Behavior ML")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📊 EDA",
        "🔍 Feature Selection",
        "📐 Feature Extraction",
        "🤖 Model Results",
        "🎯 Live Prediction",
    ])
    st.markdown("---")
    st.markdown("**Supervised by:** Dr. Wafaa Samy")
    st.markdown("**TA:** Zienab Moustafa")
    st.markdown("---")
    st.caption("Dataset: 40,034 records · 12 features · 3 classes")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">🎮 Online Gaming Behavior</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ML Classification Project — Predicting Player Engagement Level</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">40,034</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">12</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">3</div><div class="metric-label">Target Classes</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">~97%</div><div class="metric-label">Best Accuracy (RF+LDA)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Project Pipeline")

    steps = [
        ("1️⃣", "Data Loading & EDA", "Explore 40K records, distributions, correlations, insights"),
        ("2️⃣", "Preprocessing", "Drop ID · Encode target · One-Hot · 80/20 Split · Scale"),
        ("3️⃣", "Feature Selection", "Filter (Chi²+ANOVA) · Wrapper (RFE) · Embedded (Lasso+RF) → Consensus"),
        ("4️⃣", "Feature Extraction", "PCA (unsupervised, 95% variance) · LDA (supervised, 2 discriminants)"),
        ("5️⃣", "Model Training", "Random Forest · SVM · KNN — on 3 feature spaces"),
        ("6️⃣", "Evaluation", "Accuracy · 5-Fold CV · Confusion Matrix · Classification Report"),
        ("7️⃣", "Streamlit GUI", "This dashboard — live prediction + full visualization"),
    ]
    for icon, title, desc in steps:
        st.markdown(f'<div class="step-box"><b>{icon} {title}</b><br><span style="color:#5D6D7E;font-size:.9rem">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Best Results")
        results_data = {
            'Feature Space':['LDA','LDA','LDA','PCA','PCA','Selected','Selected'],
            'Model':['Random Forest','SVM','KNN','Random Forest','SVM','Random Forest','SVM'],
            'Est. Accuracy':['~97%','~96%','~94%','~95%','~94%','~96%','~95%'],
        }
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("📌 Key Insights")
        insights = [
            "LDA consistently outperforms PCA — supervised separation wins",
            "Random Forest is the best model across all feature spaces",
            "PlayerLevel & AchievementsUnlocked are top predictors",
            "Medium engagement dominates at ~48.4% of records",
            "No missing values — dataset is clean and ready",
        ]
        for ins in insights:
            st.markdown(f"✅ {ins}")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"❌ Dataset not found: `{DATA_PATH}`\n\nضع الملف في نفس مجلد app.py")
        st.stop()

    order = ['Low', 'Medium', 'High']

    # Dataset info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",  f"{len(df):,}")
    c2.metric("Features", df.shape[1]-1)
    c3.metric("Missing",  df.isnull().sum().sum())
    c4.metric("Classes",  df['EngagementLevel'].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "📋 Raw Data"])

    with tab1:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        counts = df['EngagementLevel'].value_counts()[order]
        axes[0,0].bar(order, counts, color=[PALETTE[k] for k in order], edgecolor='white', width=0.5)
        axes[0,0].set_title('Target Distribution', fontweight='bold')
        for i,(k,v) in enumerate(counts.items()):
            axes[0,0].text(i, v+200, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=9)

        for lvl in order:
            axes[0,1].hist(df[df['EngagementLevel']==lvl]['PlayTimeHours'], bins=30, alpha=0.6, label=lvl, color=PALETTE[lvl])
        axes[0,1].set_title('PlayTime Hours by Engagement Level', fontweight='bold'); axes[0,1].legend()

        for lvl in order:
            axes[0,2].hist(df[df['EngagementLevel']==lvl]['SessionsPerWeek'], bins=20, alpha=0.6, label=lvl, color=PALETTE[lvl])
        axes[0,2].set_title('Sessions Per Week by Level', fontweight='bold'); axes[0,2].legend()

        gc = df['GameGenre'].value_counts()
        axes[1,0].barh(gc.index, gc.values, color='#7e57c2', edgecolor='white')
        axes[1,0].set_title('Game Genre Distribution', fontweight='bold')

        axes[1,1].hist(df['PlayerLevel'], bins=30, color='#26a69a', edgecolor='white')
        axes[1,1].set_title('Player Level Distribution', fontweight='bold')

        for lvl in order:
            axes[1,2].hist(df[df['EngagementLevel']==lvl]['AchievementsUnlocked'], bins=25, alpha=0.6, label=lvl, color=PALETTE[lvl])
        axes[1,2].set_title('Achievements by Engagement Level', fontweight='bold'); axes[1,2].legend()

        plt.suptitle('Online Gaming Behavior — EDA Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        num_cols = ['Age','PlayTimeHours','InGamePurchases','SessionsPerWeek',
                    'AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Showing first 50 of {len(df):,} rows")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types & Nulls**")
            info = pd.DataFrame({'dtype': df.dtypes, 'nulls': df.isnull().sum(), 'unique': df.nunique()})
            st.dataframe(info, use_container_width=True)
        with col2:
            st.write("**Descriptive Statistics**")
            st.dataframe(df.describe().round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Selection":
    st.markdown('<p class="main-title">🔍 Feature Selection</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"❌ Dataset not found: `{DATA_PATH}`"); st.stop()

    with st.spinner("Running feature selection pipeline..."):
        X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X_train_mm, scaler, mms = build_pipeline(df)

        # Filter
        chi2_sel = SelectKBest(score_func=chi2, k='all')
        chi2_sel.fit(X_train_mm, y_train)
        chi2_scores = pd.Series(chi2_sel.scores_, index=X.columns).sort_values(ascending=False)

        anova_sel = SelectKBest(score_func=f_classif, k='all')
        anova_sel.fit(X_train_sc, y_train)
        anova_scores = pd.Series(anova_sel.scores_, index=X.columns).sort_values(ascending=False)

        selected_filter = set(chi2_scores[chi2_scores > chi2_scores.mean()].index) & \
                          set(anova_scores[anova_scores > anova_scores.mean()].index)

        # Wrapper
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
                  n_features_to_select=8, step=1)
        rfe.fit(X_train_sc, y_train)
        selected_rfe = set(X.columns[rfe.support_])

        # Embedded
        lasso = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, random_state=RANDOM_STATE)
        lasso.fit(X_train_sc, y_train)
        lasso_imp = pd.Series(np.abs(lasso.coef_).mean(axis=0), index=X.columns)

        rf_emb = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf_emb.fit(X_train_sc, y_train)
        rf_imp = pd.Series(rf_emb.feature_importances_, index=X.columns)

        selected_embedded = set(lasso_imp[lasso_imp > lasso_imp.mean()].index) & \
                            set(rf_imp[rf_imp > rf_imp.mean()].index)

        # Consensus
        vote_df = pd.DataFrame({'Feature': list(X.columns)})
        vote_df['Filter']   = vote_df['Feature'].isin(selected_filter).astype(int)
        vote_df['Wrapper']  = vote_df['Feature'].isin(selected_rfe).astype(int)
        vote_df['Embedded'] = vote_df['Feature'].isin(selected_embedded).astype(int)
        vote_df['Votes']    = vote_df[['Filter','Wrapper','Embedded']].sum(axis=1)
        vote_df = vote_df.sort_values('Votes', ascending=False)
        final_features = vote_df[vote_df['Votes'] >= 2]['Feature'].tolist()

    st.success(f"✅ Final selected features (≥2 votes): **{len(final_features)}** features")

    tab1, tab2, tab3, tab4 = st.tabs(["🔵 Filter", "🟢 Wrapper (RFE)", "🟣 Embedded", "🏆 Consensus"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        c_mean = chi2_scores.mean()
        axes[0].bar(range(len(chi2_scores)), chi2_scores.values,
                    color=['#ef5350' if v > c_mean else '#b0bec5' for v in chi2_scores], edgecolor='white')
        axes[0].axhline(c_mean, linestyle='--', color='black', linewidth=1.2, label=f'Mean={c_mean:.0f}')
        axes[0].set_xticks(range(len(chi2_scores)))
        axes[0].set_xticklabels(chi2_scores.index, rotation=45, ha='right', fontsize=8)
        axes[0].set_title('Chi² Scores', fontweight='bold'); axes[0].legend()

        a_mean = anova_scores.mean()
        axes[1].bar(range(len(anova_scores)), anova_scores.values,
                    color=['#42a5f5' if v > a_mean else '#b0bec5' for v in anova_scores], edgecolor='white')
        axes[1].axhline(a_mean, linestyle='--', color='black', linewidth=1.2, label=f'Mean={a_mean:.0f}')
        axes[1].set_xticks(range(len(anova_scores)))
        axes[1].set_xticklabels(anova_scores.index, rotation=45, ha='right', fontsize=8)
        axes[1].set_title('ANOVA F-Scores', fontweight='bold'); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info(f"Filter selected: **{len(selected_filter)}** features (intersection of Chi² and ANOVA above-mean)")

    with tab2:
        rfe_df = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_, 'Selected': rfe.support_}).sort_values('Ranking')
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(rfe_df['Feature'], rfe_df['Ranking'],
               color=['#66bb6a' if s else '#b0bec5' for s in rfe_df['Selected']], edgecolor='white')
        ax.axhline(1, linestyle='--', color='green', linewidth=1.2)
        ax.set_xticklabels(rfe_df['Feature'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('RFE Ranking (1=best)')
        ax.set_title('Wrapper Method — RFE Ranking (green=selected)', fontweight='bold')
        green_p = mpatches.Patch(color='#66bb6a', label='Selected')
        grey_p  = mpatches.Patch(color='#b0bec5', label='Eliminated')
        ax.legend(handles=[green_p, grey_p])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info(f"RFE selected: **{len(selected_rfe)}** features")

    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        lm = lasso_imp.mean()
        axes[0].bar(range(len(lasso_imp)), lasso_imp.sort_values(ascending=False).values,
                    color=['#ef5350' if v > lm else '#b0bec5' for v in lasso_imp.sort_values(ascending=False)],
                    edgecolor='white')
        axes[0].axhline(lm, linestyle='--', color='black', label=f'Mean={lm:.3f}')
        axes[0].set_xticks(range(len(lasso_imp)))
        axes[0].set_xticklabels(lasso_imp.sort_values(ascending=False).index, rotation=45, ha='right', fontsize=8)
        axes[0].set_title('Lasso L1 Coefficients', fontweight='bold'); axes[0].legend()

        rfm = rf_imp.mean()
        axes[1].bar(range(len(rf_imp)), rf_imp.sort_values(ascending=False).values,
                    color=['#7e57c2' if v > rfm else '#b0bec5' for v in rf_imp.sort_values(ascending=False)],
                    edgecolor='white')
        axes[1].axhline(rfm, linestyle='--', color='black', label=f'Mean={rfm:.4f}')
        axes[1].set_xticks(range(len(rf_imp)))
        axes[1].set_xticklabels(rf_imp.sort_values(ascending=False).index, rotation=45, ha='right', fontsize=8)
        axes[1].set_title('Random Forest Feature Importance', fontweight='bold'); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info(f"Embedded selected: **{len(selected_embedded)}** features (Lasso ∩ RF)")

    with tab4:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, max(4, len(vote_df)*0.35)))
            hm = vote_df.set_index('Feature')[['Filter','Wrapper','Embedded']]
            sns.heatmap(hm, annot=True, fmt='d', cmap='YlGn', linewidths=0.5, cbar=False, ax=ax)
            ax.set_title('Feature Selection Consensus\n(1=selected)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**🏆 Final Features (≥2 votes)**")
            for f in final_features:
                st.markdown(f"✅ `{f}`")
            st.metric("Final Feature Count", len(final_features))

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Feature Extraction":
    st.markdown('<p class="main-title">📐 Feature Extraction</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"❌ Dataset not found: `{DATA_PATH}`"); st.stop()

    with st.spinner("Running dimensionality reduction..."):
        X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X_train_mm, scaler, mms = build_pipeline(df)

        # Quick feature selection
        chi2_sel = SelectKBest(score_func=chi2, k='all').fit(
            pd.DataFrame(mms.fit_transform(X_train), columns=X.columns), y_train)
        anova_sel = SelectKBest(score_func=f_classif, k='all').fit(X_train_sc, y_train)
        chi2_s = pd.Series(chi2_sel.scores_, index=X.columns)
        anova_s = pd.Series(anova_sel.scores_, index=X.columns)
        sel_filter   = set(chi2_s[chi2_s > chi2_s.mean()].index) & set(anova_s[anova_s > anova_s.mean()].index)

        rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1), n_features_to_select=8, step=1)
        rfe.fit(X_train_sc, y_train)
        sel_rfe = set(X.columns[rfe.support_])

        lasso = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, random_state=RANDOM_STATE)
        lasso.fit(X_train_sc, y_train)
        li = pd.Series(np.abs(lasso.coef_).mean(axis=0), index=X.columns)
        rf_e = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_sc, y_train)
        ri = pd.Series(rf_e.feature_importances_, index=X.columns)
        sel_emb = set(li[li > li.mean()].index) & set(ri[ri > ri.mean()].index)

        vote = pd.DataFrame({'Feature': list(X.columns)})
        vote['v'] = vote['Feature'].isin(sel_filter).astype(int) + vote['Feature'].isin(sel_rfe).astype(int) + vote['Feature'].isin(sel_emb).astype(int)
        final_features = vote[vote['v'] >= 2]['Feature'].tolist()

        X_train_sel = X_train_sc[final_features]
        X_test_sel  = X_test_sc[final_features]

        # PCA
        pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_sel)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
        n_pca    = int(np.argmax(cumvar >= 0.95) + 1)
        pca      = PCA(n_components=n_pca, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_sel)
        X_test_pca  = pca.transform(X_test_sel)
        pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_train_sel)

        # LDA
        lda_model = LDA(n_components=2)
        X_train_lda = lda_model.fit_transform(X_train_sel, y_train)

    tab1, tab2, tab3 = st.tabs(["📉 PCA", "🎯 LDA", "⚖️ PCA vs LDA"])

    with tab1:
        st.markdown(f"**Optimal components:** {n_pca} (retain 95% variance)")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
                    pca_full.explained_variance_ratio_, color='#42a5f5', edgecolor='white', label='Individual')
        axes[0].plot(range(1, len(cumvar)+1), cumvar, 'o-', color='#ef5350', markersize=4, label='Cumulative')
        axes[0].axhline(0.95, linestyle='--', color='green', linewidth=1.5, label='95%')
        axes[0].axvline(n_pca, linestyle='--', color='orange', linewidth=1.5, label=f'n={n_pca}')
        axes[0].set_title('PCA Scree Plot', fontweight='bold'); axes[0].legend(fontsize=8)

        for cls in [0,1,2]:
            m = y_train == cls
            axes[1].scatter(pca_2d[m,0], pca_2d[m,1], c=CMAP[cls], label=LMAP[cls], alpha=0.35, s=8)
        axes[1].set_title(f'PCA 2D\n(PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
        axes[1].legend(markerscale=3)

        pca_k = PCA(n_components=min(6, len(final_features)), random_state=RANDOM_STATE).fit(X_train_sel)
        loadings = pd.DataFrame(pca_k.components_.T, index=final_features,
                                columns=[f'PC{i+1}' for i in range(pca_k.n_components_)])
        sns.heatmap(loadings, cmap='coolwarm', center=0, annot=True, fmt='.2f', linewidths=0.4, ax=axes[2])
        axes[2].set_title('PCA Loadings Heatmap', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        lda_var = lda_model.explained_variance_ratio_
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for cls in [0,1,2]:
            m = y_train == cls
            axes[0].scatter(X_train_lda[m,0], X_train_lda[m,1], c=CMAP[cls], label=LMAP[cls], alpha=0.4, s=8)
        axes[0].set_title('LDA 2D Projection\n(Supervised — maximizes class separation)', fontweight='bold')
        axes[0].legend(markerscale=3)

        axes[1].bar([f'LD{i+1}' for i in range(len(lda_var))], lda_var,
                    color=['#ef5350','#42a5f5'], edgecolor='white', width=0.4)
        for i, v in enumerate(lda_var):
            axes[1].text(i, v+0.005, f'{v:.2%}', ha='center', fontweight='bold')
        axes[1].set_title('LDA Explained Variance per Discriminant', fontweight='bold')
        axes[1].set_ylim(0, 1)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for cls in [0,1,2]:
            m = y_train == cls
            axes[0].scatter(pca_2d[m,0], pca_2d[m,1], c=CMAP[cls], label=LMAP[cls], alpha=0.35, s=8)
            axes[1].scatter(X_train_lda[m,0], X_train_lda[m,1], c=CMAP[cls], label=LMAP[cls], alpha=0.35, s=8)
        axes[0].set_title('PCA — Unsupervised\n(Maximizes variance)', fontweight='bold'); axes[0].legend(markerscale=3)
        axes[1].set_title('LDA — Supervised\n(Maximizes class separability)', fontweight='bold'); axes[1].legend(markerscale=3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-box"><b>PCA</b> — Unsupervised. Finds axes of maximum variance. Does not use class labels. Max components = n_features. Useful when labels are unavailable.</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="step-box"><b>LDA</b> — Supervised. Maximizes class separability. Uses class labels. Max components = n_classes−1 = 2. Better for classification tasks.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<p class="main-title">🤖 Model Results</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    scaler_m, mms_m, pca_m, lda_m, final_features, all_columns, all_results = load_models()

    if all_results is None or not all_results:
        st.markdown('<div class="warn-box">⚠️ لم يتم العثور على الـ saved_models. يجب تشغيل الـ Notebook أولاً لحفظ النماذج.</div>', unsafe_allow_html=True)
        st.markdown("""
**خطوات الحل:**
1. افتح `Player_Engagement_Prediction.ipynb`
2. شغّل كل الـ cells من الأول للآخر (Run All)
3. تأكد إن مجلد `saved_models/` اتعمل في نفس مجلد `app.py`
4. ارجع لهنا وعمل refresh
        """)
        st.stop()

    try:
        df = load_data()
        X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X_train_mm, _, _ = build_pipeline(df)

        X_train_sel = X_train_sc[final_features]
        X_test_sel  = X_test_sc[final_features]

        X_train_pca = pca_m.transform(X_train_sel)
        X_test_pca  = pca_m.transform(X_test_sel)
        X_train_lda = lda_m.transform(X_train_sel)
        X_test_lda  = lda_m.transform(X_test_sel)

        datasets = {
            'Selected Features': (X_train_sel.values, X_test_sel.values),
            'PCA': (X_train_pca, X_test_pca),
            'LDA': (X_train_lda, X_test_lda),
        }

        with st.spinner("Evaluating models..."):
            rows = []
            predictions = {}
            for ds_name, (Xtr, Xte) in datasets.items():
                predictions[ds_name] = {}
                for m_name, mdl in all_results.get(ds_name, {}).items():
                    y_pred = mdl.predict(Xte)
                    acc = accuracy_score(y_test, y_pred)
                    cv  = cross_val_score(mdl, Xtr, y_train, cv=5, scoring='accuracy', n_jobs=-1)
                    predictions[ds_name][m_name] = y_pred
                    rows.append({'Feature Space': ds_name, 'Model': m_name,
                                 'Test Accuracy': acc, 'CV Mean': cv.mean(), 'CV Std': cv.std()})

        results_df = pd.DataFrame(rows).sort_values('Test Accuracy', ascending=False).reset_index(drop=True)

        # Summary table
        st.subheader("📊 Full Results Table")
        display_df = results_df.copy()
        display_df['Test Accuracy'] = display_df['Test Accuracy'].map('{:.2%}'.format)
        display_df['CV Mean'] = display_df['CV Mean'].map('{:.2%}'.format)
        display_df['CV Std']  = display_df['CV Std'].map('{:.4f}'.format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart
        st.subheader("📈 Accuracy Comparison")
        pivot = results_df.pivot(index='Model', columns='Feature Space', values='Test Accuracy')
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', width=0.7)
        ax.set_title('Model Accuracy Across Feature Spaces', fontweight='bold')
        ax.set_ylabel('Test Accuracy'); ax.set_ylim(0.5, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Feature Space', bbox_to_anchor=(1,1))
        for container in ax.containers:
            ax.bar_label(container, fmt='{:.2%}', fontsize=8, padding=2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Confusion matrices
        st.subheader("🔲 Confusion Matrices (LDA Features)")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, (m_name, y_pred) in zip(axes, predictions.get('LDA', {}).items()):
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False, cmap='Blues')
            acc = accuracy_score(y_test, y_pred)
            ax.set_title(f'{m_name}\nAcc: {acc:.2%}', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Classification reports
        st.subheader("📋 Classification Reports")
        ds_choice = st.selectbox("Feature Space", list(predictions.keys()))
        m_choice  = st.selectbox("Model", list(predictions.get(ds_choice, {}).keys()))
        if ds_choice in predictions and m_choice in predictions[ds_choice]:
            report = classification_report(y_test, predictions[ds_choice][m_choice],
                                           target_names=CLASS_NAMES, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error in evaluation: {e}")
        st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.markdown('<p class="main-title">🎯 Live Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">أدخل بيانات لاعب وهنتوقع مستوى تفاعله</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    scaler_m, mms_m, pca_m, lda_m, final_features, all_columns, all_results = load_models()

    if scaler_m is None:
        st.markdown('<div class="warn-box">⚠️ الـ saved_models مش موجودة. شغّل الـ Notebook الأول.</div>', unsafe_allow_html=True)
        st.stop()

    try:
        df = load_data()
        X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X_train_mm, _, _ = build_pipeline(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("👤 Player Profile")
            age           = st.slider("Age", 10, 60, 25)
            gender        = st.selectbox("Gender", ["Male", "Female"])
            location      = st.selectbox("Location", sorted(df['Location'].unique()))
            game_genre    = st.selectbox("Game Genre", sorted(df['GameGenre'].unique()))
            difficulty    = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])

        with col2:
            st.subheader("🎮 Gaming Behavior")
            playtime      = st.slider("PlayTime Hours/day", 0.0, 15.0, 3.0, 0.5)
            sessions      = st.slider("Sessions Per Week", 1, 30, 7)
            avg_dur       = st.slider("Avg Session Duration (min)", 10, 300, 60)
            purchases     = st.radio("In-Game Purchases", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            st.subheader("🏆 Player Stats")
            player_level  = st.slider("Player Level", 1, 100, 30)
            achievements  = st.slider("Achievements Unlocked", 0, 100, 20)

            st.markdown("<br>", unsafe_allow_html=True)
            model_choice  = st.selectbox("Model", ["Random Forest", "SVM", "KNN"])
            feature_space = st.selectbox("Feature Space", ["LDA", "PCA", "Selected Features"])
            predict_btn   = st.button("🚀 Predict Engagement Level", use_container_width=True, type="primary")

        if predict_btn:
            sample = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'Location': location,
                'GameGenre': game_genre, 'PlayTimeHours': playtime,
                'InGamePurchases': purchases, 'GameDifficulty': difficulty,
                'SessionsPerWeek': sessions, 'AvgSessionDurationMinutes': avg_dur,
                'PlayerLevel': player_level, 'AchievementsUnlocked': achievements,
            }])

            # Encode
            raw_enc = pd.get_dummies(sample, columns=['Gender','Location','GameGenre','GameDifficulty'], drop_first=True)
            for col in X.columns:
                if col not in raw_enc.columns:
                    raw_enc[col] = 0
            raw_enc = raw_enc[X.columns]

            # Scale & select
            raw_sc  = pd.DataFrame(scaler_m.transform(raw_enc), columns=X.columns)
            raw_sel = raw_sc[final_features]

            if feature_space == "LDA":
                raw_input = lda_m.transform(raw_sel)
            elif feature_space == "PCA":
                raw_input = pca_m.transform(raw_sel)
            else:
                raw_input = raw_sel.values

            mdl = all_results.get(feature_space, {}).get(model_choice)
            if mdl is None:
                st.error("Model not found in saved_models. Make sure the notebook ran completely.")
                st.stop()

            pred = mdl.predict(raw_input)[0]
            proba = mdl.predict_proba(raw_input)[0] if hasattr(mdl, 'predict_proba') else None

            label = LMAP[pred]
            css_class = f"pred-{label.lower()}"
            emoji = {"High":"🟢","Medium":"🔵","Low":"🔴"}.get(label,"")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="{css_class}">{emoji} Predicted Engagement: <b>{label}</b></div>', unsafe_allow_html=True)

            if proba is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': proba})
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.barh(CLASS_NAMES, proba, color=[PALETTE[c] for c in CLASS_NAMES], edgecolor='white')
                ax.set_xlim(0, 1); ax.set_xlabel('Probability')
                ax.set_title(f'{model_choice} — {feature_space}', fontweight='bold')
                for bar, p in zip(bars, proba):
                    ax.text(p + 0.01, bar.get_y() + bar.get_height()/2, f'{p:.2%}', va='center', fontsize=11, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

            # Explanation
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💡 What does this mean?")
            explanations = {
                "High": "🟢 **High Engagement** — هذا اللاعب يقضي وقتاً طويلاً في اللعبة ويتفاعل بشكل مكثف. مثالي للاستهداف بالعروض والمحتوى الحصري.",
                "Medium": "🔵 **Medium Engagement** — اللاعب يلعب بانتظام معقول. يمكن تحفيزه للوصول لمستوى أعلى من التفاعل بالعروض المناسبة.",
                "Low": "🔴 **Low Engagement** — اللاعب يلعب بشكل متقطع. قد يحتاج لتحفيز خاص أو قد يكون في خطر التوقف عن اللعب.",
            }
            st.markdown(explanations[label])

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)
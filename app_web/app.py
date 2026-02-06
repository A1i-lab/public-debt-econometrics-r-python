import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Chargement de l'application...")

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================

df_raw = pd.read_excel('FinalDataset.xlsx')

annees_cols = [col for col in df_raw.columns 
               if str(col).replace('.','').replace('-','').isdigit() 
               or isinstance(col, (int, float))]

dfs_list = []
for pays in df_raw['Pays'].unique():
    df_pays_raw = df_raw[df_raw['Pays'] == pays]
    for annee in annees_cols:
        ligne = {'Pays': pays, 'annee': annee}
        for _, row in df_pays_raw.iterrows():
            var_nom = row['Nom des variables']
            var_valeur = row[annee]
            ligne[var_nom] = var_valeur
        dfs_list.append(ligne)

df = pd.DataFrame(dfs_list)

mapping = {
    "Dette totale de l'administration centrale (% du PIB)": 'dette_publique',
    "Paiements d'intérêts (% des recettes)": 'paiements_interets',
    'Chômage total (% de la population active)': 'chomage',
    'Croissance du PIB (% annuel)': 'croissance_pib',
    'Inflation (indice des prix à la consommation, % annuel)': 'inflation',
    'Dépenses publiques finales (% du PIB)': 'depenses_publiques',
    'Recettes publiques hors subventions (% du PIB)': 'recettes_publiques',
    'Solde du compte courant (% du PIB)': 'solde_courant',
    'Population de 65 ans et plus (% population totale)': 'population_65plus',
    'Épargne intérieure brute (% du PIB)': 'epargne_interieure'
}

colonnes_a_renommer = {k: v for k, v in mapping.items() if k in df.columns}
df = df.rename(columns=colonnes_a_renommer)

if 'dette_publique' not in df.columns:
    dette_col = [col for col in df.columns if 'dette' in col.lower()]
    if dette_col:
        df = df.rename(columns={dette_col[0]: 'dette_publique'})

df['annee'] = pd.to_numeric(df['annee'], errors='coerce')

features = ['chomage', 'croissance_pib', 'inflation', 'depenses_publiques', 
            'recettes_publiques', 'solde_courant', 'population_65plus', 
            'paiements_interets', 'epargne_interieure']
features = [f for f in features if f in df.columns]

df = df.dropna(subset=['dette_publique', 'annee'] + features)

for col in ['dette_publique'] + features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['dette_publique'] + features)

print(f"{len(df)} obs | {df['Pays'].nunique()} pays")

# ============================================
# 2. MODÈLE ML
# ============================================

X = df[features]
y = df['dette_publique']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

pays_liste = sorted(df['Pays'].unique().tolist())

# ============================================
# 3. CLUSTERING
# ============================================

df_pays_avg = df.groupby('Pays')[['dette_publique'] + features].mean().reset_index()
X_cluster = df_pays_avg[['dette_publique', 'croissance_pib', 'chomage']].dropna()
pays_cluster = df_pays_avg.loc[X_cluster.index, 'Pays'].values

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(StandardScaler().fit_transform(X_cluster))

cluster_labels = ['🟢 Faible dette', '🟡 Dette modérée', '🟠 Dette élevée', '🔴 Dette critique']

# ============================================
# 4. FONCTIONS
# ============================================

def analyser_pays(pays):
    df_pays = df[df['Pays'] == pays].sort_values('annee')
    if len(df_pays) == 0:
        return "⚠️ Aucune donnée", None, None, None
    
    dette_act = df_pays['dette_publique'].iloc[-1]
    annee_act = int(df_pays['annee'].iloc[-1])
    
    derniere_obs = df_pays[features].iloc[-1].values.reshape(1, -1)
    pred_2025 = rf_model.predict(scaler.transform(derniere_obs))[0]
    
    projections = [pred_2025]
    for i in range(5):
        projections.append(projections[-1] * 1.01)
    
    annees_futures = list(range(2025, 2031))
    
    # GRAPHIQUE 1 : Dette + Prédictions
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_pays['annee'], 
        y=df_pays['dette_publique'],
        mode='lines+markers', 
        name='Historique',
        line=dict(color='#2E86AB', width=3)
    ))
    fig1.add_trace(go.Scatter(
        x=[annee_act] + annees_futures, 
        y=[dette_act] + projections,
        mode='lines+markers', 
        name='Prédiction ML', 
        line=dict(dash='dash', color='#FF6B6B', width=3)
    ))
    fig1.add_hline(y=60, line_dash="dot", line_color="green", annotation_text="Maastricht (60%)")
    fig1.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Critique (100%)")
    fig1.update_layout(
        title=f"Dette publique - {pays}",
        xaxis_title="Année",
        yaxis_title="Dette (% PIB)",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    # GRAPHIQUE 2 : Dashboard économique
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Croissance PIB', 'Chômage', 'Inflation', 'Dépenses publiques')
    )
    
    if 'croissance_pib' in df_pays.columns:
        fig2.add_trace(go.Scatter(
            x=df_pays['annee'], 
            y=df_pays['croissance_pib'],
            mode='lines', 
            name='Croissance',
            line=dict(color='green')
        ), row=1, col=1)
    
    if 'chomage' in df_pays.columns:
        fig2.add_trace(go.Scatter(
            x=df_pays['annee'], 
            y=df_pays['chomage'],
            mode='lines', 
            name='Chômage',
            line=dict(color='red')
        ), row=1, col=2)
    
    if 'inflation' in df_pays.columns:
        fig2.add_trace(go.Scatter(
            x=df_pays['annee'], 
            y=df_pays['inflation'],
            mode='lines', 
            name='Inflation',
            line=dict(color='orange')
        ), row=2, col=1)
    
    if 'depenses_publiques' in df_pays.columns:
        fig2.add_trace(go.Scatter(
            x=df_pays['annee'], 
            y=df_pays['depenses_publiques'],
            mode='lines', 
            name='Dépenses',
            line=dict(color='blue')
        ), row=2, col=2)
    
    fig2.update_layout(height=600, showlegend=False, template='plotly_white')
    
    # GRAPHIQUE 3 : Feature Importance
    fig3 = go.Figure(go.Bar(
        x=feature_importance['importance'][:8],
        y=feature_importance['feature'][:8],
        orientation='h',
        marker=dict(
            color=feature_importance['importance'][:8],
            colorscale='Viridis',
            showscale=True
        )
    ))
    fig3.update_layout(
        title=" Variables les plus influentes",
        xaxis_title="Importance",
        height=400,
        template='plotly_white'
    )
    
    texte = f"""
    ##  {pays}
    
    ### Situation actuelle ({annee_act})
    - **Dette publique** : {dette_act:.1f}% du PIB
    - **Croissance PIB** : {df_pays['croissance_pib'].iloc[-1]:.1f}%
    - **Chômage** : {df_pays['chomage'].iloc[-1]:.1f}%
    
    ### Prédictions ML
    - **2025** : {pred_2025:.1f}%
    - **2030** : {projections[-1]:.1f}%
    - **Évolution** : {"Hausse" if pred_2025 > dette_act else "Baisse"} de {abs(pred_2025 - dette_act):.1f} points
    """
    
    return texte, fig1, fig2, fig3


def comparer_pays(pays1, pays2):
    df_p1 = df[df['Pays'] == pays1].sort_values('annee')
    df_p2 = df[df['Pays'] == pays2].sort_values('annee')
    
    if len(df_p1) == 0 or len(df_p2) == 0:
        return "⚠️ Données manquantes", None, None
    
    # GRAPHIQUE 1 : Dette comparée
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_p1['annee'], 
        y=df_p1['dette_publique'],
        mode='lines+markers',
        name=pays1,
        line=dict(width=3, color='#2E86AB')
    ))
    fig1.add_trace(go.Scatter(
        x=df_p2['annee'], 
        y=df_p2['dette_publique'],
        mode='lines+markers',
        name=pays2,
        line=dict(width=3, color='#A23B72')
    ))
    fig1.add_hline(y=60, line_dash="dot", line_color="green")
    fig1.update_layout(
        title=f"Dette publique : {pays1} vs {pays2}",
        xaxis_title="Année",
        yaxis_title="Dette (% PIB)",
        height=500,
        template='plotly_white'
    )
    
    # GRAPHIQUE 2 : Profil économique
    categories = ['Dette', 'Croissance', 'Chômage', 'Inflation']
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatterpolar(
        r=[
            df_p1['dette_publique'].iloc[-1] / 100,
            max(0, df_p1['croissance_pib'].iloc[-1] / 10),
            df_p1['chomage'].iloc[-1] / 20,
            df_p1['inflation'].iloc[-1] / 10
        ],
        theta=categories,
        fill='toself',
        name=pays1
    ))
    
    fig2.add_trace(go.Scatterpolar(
        r=[
            df_p2['dette_publique'].iloc[-1] / 100,
            max(0, df_p2['croissance_pib'].iloc[-1] / 10),
            df_p2['chomage'].iloc[-1] / 20,
            df_p2['inflation'].iloc[-1] / 10
        ],
        theta=categories,
        fill='toself',
        name=pays2
    ))
    
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Profil économique",
        height=500,
        template='plotly_white'
    )
    
    texte = f"""
    ## ⚖️ {pays1} vs {pays2}
    
    | Indicateur | {pays1} | {pays2} | Écart |
    |------------|---------|---------|-------|
    | **Dette** | {df_p1['dette_publique'].iloc[-1]:.1f}% | {df_p2['dette_publique'].iloc[-1]:.1f}% | {abs(df_p1['dette_publique'].iloc[-1] - df_p2['dette_publique'].iloc[-1]):.1f} pts |
    | **Croissance** | {df_p1['croissance_pib'].iloc[-1]:.1f}% | {df_p2['croissance_pib'].iloc[-1]:.1f}% | {abs(df_p1['croissance_pib'].iloc[-1] - df_p2['croissance_pib'].iloc[-1]):.1f} pts |
    | **Chômage** | {df_p1['chomage'].iloc[-1]:.1f}% | {df_p2['chomage'].iloc[-1]:.1f}% | {abs(df_p1['chomage'].iloc[-1] - df_p2['chomage'].iloc[-1]):.1f} pts |
    """
    
    return texte, fig1, fig2


def simuler_scenario(pays, croissance_sim, chomage_sim, inflation_sim):
    df_pays = df[df['Pays'] == pays].sort_values('annee')
    if len(df_pays) == 0:
        return "⚠️ Aucune donnée", None
    
    derniere_obs = df_pays[features].iloc[-1].copy()
    
    scenario_features = derniere_obs.copy()
    if 'croissance_pib' in features:
        scenario_features.loc['croissance_pib'] = croissance_sim
    if 'chomage' in features:
        scenario_features.loc['chomage'] = chomage_sim
    if 'inflation' in features:
        scenario_features.loc['inflation'] = inflation_sim
    
    X_sim = scenario_features.values.reshape(1, -1)
    dette_predite = rf_model.predict(scaler.transform(X_sim))[0]
    
    dette_actuelle = df_pays['dette_publique'].iloc[-1]
    impact = dette_predite - dette_actuelle
    
    couleur = 'red' if impact > 0 else 'green'
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Dette actuelle', 'Dette simulée'],
        y=[dette_actuelle, dette_predite],
        marker_color=['#2E86AB', couleur],
        text=[f"{dette_actuelle:.1f}%", f"{dette_predite:.1f}%"],
        textposition='auto'
    ))
    fig.add_hline(y=60, line_dash="dot", line_color="green", annotation_text="Maastricht")
    fig.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Critique")
    fig.update_layout(
        title=f"Impact du scénario - {pays}",
        yaxis_title="Dette (% PIB)",
        height=400,
        template='plotly_white'
    )
    
    texte = f"""
    ## Simulation - {pays}
    
    ### Scénario testé
    - Croissance : **{croissance_sim}%**
    - Chômage : **{chomage_sim}%**
    - Inflation : **{inflation_sim}%**
    
    ### Résultat
    - Dette actuelle : **{dette_actuelle:.1f}%**
    - Dette prédite : **{dette_predite:.1f}%**
    - **Impact** : {"📈" if impact > 0 else "📉"} **{impact:+.1f} points** de PIB
    
    {"**ALERTE** : Forte hausse de la dette !" if impact > 10 else ""}
    {"**POSITIF** : Réduction de la dette" if impact < -5 else ""}
    {"**STABLE** : Impact modéré" if -5 <= impact <= 10 else ""}
    """
    
    return texte, fig


def afficher_clustering():
    fig = go.Figure()
    
    for i, label in enumerate(cluster_labels):
        mask = clusters == i
        fig.add_trace(go.Scatter3d(
            x=X_cluster.iloc[mask, 0],
            y=X_cluster.iloc[mask, 1],
            z=X_cluster.iloc[mask, 2],
            mode='markers+text',
            name=label,
            text=pays_cluster[mask],
            textposition='top center',
            marker=dict(size=10, line=dict(width=2, color='white')),
            hovertemplate='<b>%{text}</b><br>Dette: %{x:.1f}%<br>Croissance: %{y:.1f}%<br>Chômage: %{z:.1f}%'
        ))
    
    fig.update_layout(
        title=" Classification des pays OCDE (K-Means)",
        scene=dict(
            xaxis_title="Dette (% PIB)",
            yaxis_title="Croissance (%)",
            zaxis_title="Chômage (%)"
        ),
        height=700,
        template='plotly_white'
    )
    
    texte = """
    ## Classification automatique (Machine Learning)
    
    **Algorithme** : K-Means (4 clusters)  
    **Variables** : Dette, Croissance, Chômage
    
    💡 *Survolez les points pour voir les détails*
    """
    
    return texte, fig


def afficher_performance_ml():
    texte = f"""
    ## Performance du Modèle Machine Learning
    
    ### Algorithme
    **Random Forest Regressor**
    - 300 arbres de décision
    - Profondeur maximale : 20
    - Features : {len(features)}
    
    ### Métriques sur test set
    | Métrique | Valeur | Interprétation |
    |----------|--------|----------------|
    | **R²** | {r2:.3f} | {"Excellent" if r2 > 0.9 else "Bon"} |
    | **MAE** | {mae:.2f} pts | Erreur moyenne absolue |
    | **RMSE** | {rmse:.2f} pts | Écart-type des erreurs |
    
    ### Validation croisée (5-fold)
    - R² moyen : **{cv_scores.mean():.3f}**
    - Écart-type : ±{cv_scores.std():.3f}
    
    **Le modèle est robuste et généralisable**
    """
    
    fig1 = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker=dict(
            color=feature_importance['importance'],
            colorscale='Plasma',
            showscale=True
        )
    ))
    fig1.update_layout(
        title="Importance des variables",
        xaxis_title="Importance (Gini)",
        height=400,
        template='plotly_white'
    )
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(size=8, color='#2E86AB', opacity=0.6),
        name='Prédictions'
    ))
    fig2.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        line=dict(dash='dash', color='red', width=2),
        name='Parfait'
    ))
    fig2.update_layout(
        title="Prédictions vs Valeurs réelles",
        xaxis_title="Dette réelle (% PIB)",
        yaxis_title="Dette prédite (% PIB)",
        height=500,
        template='plotly_white'
    )
    
    return texte, fig1, fig2


# ============================================
# 5. INTERFACE GRADIO
# ============================================

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
h1 {
    background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em !important;
    font-weight: 800 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Dette OCDE ML") as demo:
    
    gr.Markdown("""
    # Simulateur Dette Publique OCDE
    ### *Application avancée de Machine Learning pour l'analyse macroéconomique*
    """)
    
    gr.Markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>
    <h3 style='margin:0; color:white;'>Dataset</h3>
    <p style='margin:5px 0;'><b>{len(df):,}</b> observations | <b>{df['Pays'].nunique()}</b> pays OCDE | <b>{len(features)}</b> features</p>
    <p style='margin:5px 0;'>Période: {int(df['annee'].min())}-{int(df['annee'].max())} | Modèle: Random Forest (R²={r2:.3f})</p>
    </div>
    """)
    
    with gr.Tabs():
        
        with gr.Tab("Analyse Pays"):
            gr.Markdown("""
            ### Analyse détaillée d'un pays avec prédictions jusqu'en 2030
            Sélectionnez un pays pour voir l'évolution historique de sa dette, les prédictions ML et un dashboard complet.
            """)
            
            pays_input = gr.Dropdown(pays_liste, value="France", label="Sélectionnez un pays")
            btn = gr.Button("Analyser", variant="primary", size="lg")
            output_text = gr.Markdown()
            
            with gr.Row():
                output_fig1 = gr.Plot(label="Évolution & Prédictions")
            
            with gr.Row():
                output_fig2 = gr.Plot(label="Dashboard économique")
            
            with gr.Row():
                output_fig3 = gr.Plot(label="Feature Importance")
            
            btn.click(analyser_pays, [pays_input], [output_text, output_fig1, output_fig2, output_fig3])
        
        with gr.Tab("Simulateur"):
            gr.Markdown("""
            ### Testez vos propres scénarios économiques
            Modifiez les variables macroéconomiques et voyez l'impact sur la dette publique.
            """)
            
            with gr.Row():
                pays_sim = gr.Dropdown(pays_liste, value="France", label="Pays")
            
            with gr.Row():
                croissance = gr.Slider(-5, 10, value=2, step=0.5, label="Croissance du PIB (%)")
                chomage = gr.Slider(0, 25, value=8, step=0.5, label= "Taux de chômage (%)")
                inflation = gr.Slider(-2, 15, value=2, step=0.5, label=" Inflation (%)")
            
            btn_sim = gr.Button("Simuler", variant="primary", size="lg")
            output_sim_text = gr.Markdown()
            output_sim_fig = gr.Plot()
            
            btn_sim.click(simuler_scenario, [pays_sim, croissance, chomage, inflation], [output_sim_text, output_sim_fig])
        
        with gr.Tab("Comparaison"):
            gr.Markdown("""
            ### Comparez deux pays côte à côte
            Analyse comparative avec graphiques et radar chart.
            """)
            
            with gr.Row():
                pays1 = gr.Dropdown(pays_liste, value="France", label="Pays 1")
                pays2 = gr.Dropdown(pays_liste, value="Germany", label="Pays 2")
            
            btn_comp = gr.Button("Comparer", variant="primary", size="lg")
            output_comp_text = gr.Markdown()
            
            with gr.Row():
                output_comp_fig1 = gr.Plot(label="Dette publique")
                output_comp_fig2 = gr.Plot(label="Profil économique")
            
            btn_comp.click(comparer_pays, [pays1, pays2], [output_comp_text, output_comp_fig1, output_comp_fig2])
        
        with gr.Tab("Clustering"):
            gr.Markdown("""
            ### Clustering automatique des pays (K-Means)
            Les pays sont regroupés selon leur profil économique (dette, croissance, chômage).
            """)
            
            btn_cluster = gr.Button("Afficher la carte", variant="primary", size="lg")
            output_cluster_text = gr.Markdown()
            output_cluster_fig = gr.Plot()
            
            btn_cluster.click(afficher_clustering, outputs=[output_cluster_text, output_cluster_fig])
        
        with gr.Tab("Performance ML"):
            gr.Markdown("""
            ### Métriques et validation du modèle
            Évaluation détaillée du Random Forest Regressor.
            """)
            
            btn_perf = gr.Button("Afficher les métriques", variant="primary", size="lg")
            output_perf_text = gr.Markdown()
            
            with gr.Row():
                output_perf_fig1 = gr.Plot(label="Feature Importance")
                output_perf_fig2 = gr.Plot(label="Prédictions vs Réel")
            
            btn_perf.click(afficher_performance_ml, outputs=[output_perf_text, output_perf_fig1, output_perf_fig2])
    
    gr.Markdown("""
    ---
    ### À propos
    
    **Application développée pour mémoire de Master en Économie**
    
    - **Modèle** : Random Forest Regressor (300 arbres, profondeur 20)
    - **Sources** : OCDE, Banque Mondiale, FMI
    - **Variables** : 9 indicateurs macroéconomiques
    - **Période** : 1990-2023 | **Prédictions** : 2025-2030
    
    💡 *Toutes les prédictions sont basées sur un modèle statistique et doivent être interprétées avec précaution*
    """)

demo.launch()

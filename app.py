import streamlit as st

st.set_page_config(
    page_title="MediPredict AI – Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ──────────────────────────────────────────────────────────────────────────────
# SYMPTOM & DISEASE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
SYMPTOMS = [
    'fever', 'high_fever', 'chills', 'fatigue', 'malaise',
    'headache', 'body_ache', 'joint_pain', 'muscle_pain', 'back_pain',
    'cough', 'dry_cough', 'productive_cough', 'shortness_of_breath', 'chest_pain',
    'sore_throat', 'runny_nose', 'nasal_congestion', 'sneezing', 'loss_of_smell',
    'nausea', 'vomiting', 'diarrhea', 'stomach_pain', 'loss_of_appetite',
    'skin_rash', 'itching', 'yellowish_skin', 'pale_skin', 'sweating',
    'dizziness', 'blurred_vision', 'excessive_thirst', 'frequent_urination',
    'weight_loss', 'weight_gain', 'swollen_lymph_nodes', 'stiff_neck',
    'sensitivity_to_light', 'red_eyes', 'dark_urine', 'clay_colored_stool'
]

DISEASE_SYMPTOM_MAP = {
    'Common Cold':  {
        'primary':   ['runny_nose','nasal_congestion','sneezing','sore_throat','cough'],
        'secondary': ['fatigue','headache','body_ache','fever','malaise']},
    'Influenza':    {
        'primary':   ['high_fever','body_ache','fatigue','headache','chills'],
        'secondary': ['cough','sore_throat','muscle_pain','sweating','malaise']},
    'COVID-19':     {
        'primary':   ['fever','dry_cough','fatigue','loss_of_smell','shortness_of_breath'],
        'secondary': ['body_ache','headache','sore_throat','chest_pain','diarrhea']},
    'Pneumonia':    {
        'primary':   ['high_fever','productive_cough','chest_pain','shortness_of_breath','chills'],
        'secondary': ['fatigue','nausea','sweating','body_ache','loss_of_appetite']},
    'Bronchitis':   {
        'primary':   ['productive_cough','chest_pain','shortness_of_breath','fatigue','chills'],
        'secondary': ['fever','sore_throat','body_ache','headache','sweating']},
    'Malaria':      {
        'primary':   ['high_fever','chills','sweating','headache','body_ache'],
        'secondary': ['nausea','vomiting','fatigue','malaise','joint_pain']},
    'Dengue Fever': {
        'primary':   ['high_fever','headache','joint_pain','skin_rash','muscle_pain'],
        'secondary': ['nausea','vomiting','fatigue','red_eyes','body_ache']},
    'Typhoid':      {
        'primary':   ['high_fever','stomach_pain','headache','loss_of_appetite','malaise'],
        'secondary': ['nausea','vomiting','diarrhea','fatigue','sweating']},
    'Hepatitis B':  {
        'primary':   ['yellowish_skin','dark_urine','clay_colored_stool','fatigue','stomach_pain'],
        'secondary': ['nausea','vomiting','loss_of_appetite','joint_pain','fever']},
    'Tuberculosis': {
        'primary':   ['productive_cough','weight_loss','sweating','fatigue','chest_pain'],
        'secondary': ['fever','chills','loss_of_appetite','body_ache','shortness_of_breath']},
    'Diabetes':     {
        'primary':   ['excessive_thirst','frequent_urination','weight_loss','fatigue','blurred_vision'],
        'secondary': ['loss_of_appetite','nausea','dizziness','itching','skin_rash']},
    'Hypertension': {
        'primary':   ['headache','dizziness','blurred_vision','chest_pain','shortness_of_breath'],
        'secondary': ['fatigue','nausea','back_pain','body_ache','malaise']},
    'Migraine':     {
        'primary':   ['headache','sensitivity_to_light','nausea','blurred_vision','dizziness'],
        'secondary': ['vomiting','fatigue','body_ache','loss_of_appetite','malaise']},
    'Chickenpox':   {
        'primary':   ['skin_rash','itching','fever','fatigue','loss_of_appetite'],
        'secondary': ['headache','body_ache','sore_throat','malaise','chills']},
    'Meningitis':   {
        'primary':   ['stiff_neck','high_fever','headache','sensitivity_to_light','nausea'],
        'secondary': ['vomiting','chills','skin_rash','fatigue','malaise']},
}

DISEASE_INFO = {
    'Common Cold':  {'severity': 'Low',    'color': '#4CAF50',
        'desc': 'A viral infection of the upper respiratory tract. Usually mild and self-limiting.',
        'advice': 'Rest, stay hydrated, use OTC cold remedies. See a doctor if symptoms last over 10 days.'},
    'Influenza':    {'severity': 'Medium', 'color': '#FF9800',
        'desc': 'A contagious respiratory illness caused by influenza viruses with sudden onset.',
        'advice': 'Rest, fluids, antiviral medication if caught early. Seek care for high-risk individuals.'},
    'COVID-19':     {'severity': 'High',   'color': '#F44336',
        'desc': 'A respiratory illness caused by the SARS-CoV-2 virus with varying severity.',
        'advice': 'Isolate immediately. Consult a doctor. Seek emergency care for breathing difficulty.'},
    'Pneumonia':    {'severity': 'High',   'color': '#F44336',
        'desc': 'An infection that inflames the air sacs in one or both lungs.',
        'advice': 'Seek medical care promptly. Antibiotics for bacterial pneumonia. May require hospitalisation.'},
    'Bronchitis':   {'severity': 'Medium', 'color': '#FF9800',
        'desc': 'Inflammation of the bronchial tubes that carry air to the lungs.',
        'advice': 'Rest, fluids, avoid smoke. See a doctor if symptoms persist beyond 3 weeks.'},
    'Malaria':      {'severity': 'High',   'color': '#F44336',
        'desc': 'A mosquito-borne disease caused by Plasmodium parasites affecting red blood cells.',
        'advice': 'Seek immediate medical care. Antimalarial medications required. Do not delay treatment.'},
    'Dengue Fever': {'severity': 'High',   'color': '#F44336',
        'desc': 'A mosquito-borne viral disease causing severe flu-like illness.',
        'advice': 'See a doctor immediately. Monitor platelet count. Avoid aspirin and NSAIDs.'},
    'Typhoid':      {'severity': 'High',   'color': '#F44336',
        'desc': 'A bacterial infection caused by Salmonella Typhi, spread through contaminated food/water.',
        'advice': 'Antibiotics required. Consult a doctor. Maintain hydration. Avoid self-medication.'},
    'Hepatitis B':  {'severity': 'High',   'color': '#F44336',
        'desc': 'A viral liver infection that can become chronic and lead to cirrhosis or liver cancer.',
        'advice': 'Consult a specialist. Antiviral therapy may be needed. Avoid alcohol completely.'},
    'Tuberculosis': {'severity': 'High',   'color': '#F44336',
        'desc': 'A bacterial infection primarily affecting the lungs, spread through the air.',
        'advice': 'Requires 6+ months of antibiotics under medical supervision. Isolate during early treatment.'},
    'Diabetes':     {'severity': 'Medium', 'color': '#FF9800',
        'desc': 'A metabolic disease causing high blood sugar due to insulin deficiency or resistance.',
        'advice': 'Consult a doctor for blood sugar testing. Lifestyle changes and medication may be required.'},
    'Hypertension': {'severity': 'Medium', 'color': '#FF9800',
        'desc': 'Persistently elevated blood pressure that increases risk of heart disease and stroke.',
        'advice': 'See a doctor for blood pressure monitoring. Reduce salt, exercise regularly, manage stress.'},
    'Migraine':     {'severity': 'Low',    'color': '#4CAF50',
        'desc': 'A neurological condition causing severe recurring headaches often with nausea and light sensitivity.',
        'advice': 'Rest in a dark quiet room. OTC pain relievers may help. See a neurologist for frequent attacks.'},
    'Chickenpox':   {'severity': 'Medium', 'color': '#FF9800',
        'desc': 'A highly contagious viral infection causing an itchy blister-like rash.',
        'advice': 'Rest, avoid scratching, use calamine lotion. Keep away from unvaccinated individuals.'},
    'Meningitis':   {'severity': 'High',   'color': '#F44336',
        'desc': 'Inflammation of the membranes surrounding the brain and spinal cord — a medical emergency.',
        'advice': '🚨 SEEK EMERGENCY CARE IMMEDIATELY. Bacterial meningitis is life-threatening without rapid treatment.'},
}

SYMPTOM_CATEGORIES = {
    '🌡️ Fever & Chills':    ['fever','high_fever','chills','sweating'],
    '😴 Fatigue & General': ['fatigue','malaise','weight_loss','weight_gain'],
    '🧠 Head & Neuro':      ['headache','dizziness','blurred_vision','stiff_neck','sensitivity_to_light'],
    '💪 Body Pain':         ['body_ache','joint_pain','muscle_pain','back_pain'],
    '🫁 Respiratory':       ['cough','dry_cough','productive_cough','shortness_of_breath','chest_pain',
                              'sore_throat','runny_nose','nasal_congestion','sneezing','loss_of_smell'],
    '🤢 Digestive':         ['nausea','vomiting','diarrhea','stomach_pain','loss_of_appetite'],
    '🩺 Skin & Eyes':       ['skin_rash','itching','yellowish_skin','pale_skin','red_eyes'],
    '💧 Metabolic':         ['excessive_thirst','frequent_urination','dark_urine','clay_colored_stool',
                              'swollen_lymph_nodes'],
}


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL IN-MEMORY (runs once, cached for the session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Training AI model... please wait")
def train_model():
    np.random.seed(42)
    rows = []

    for disease, groups in DISEASE_SYMPTOM_MAP.items():
        primary   = [s for s in groups['primary']   if s in SYMPTOMS]
        secondary = [s for s in groups['secondary'] if s in SYMPTOMS]

        for _ in range(200):
            row = {s: 0 for s in SYMPTOMS}
            for s in primary:
                row[s] = int(np.random.random() > 0.20)      # 80% chance
            for s in secondary:
                row[s] = int(np.random.random() > 0.55)      # 45% chance
            noise_pool = [s for s in SYMPTOMS if s not in primary and s not in secondary]
            for s in np.random.choice(noise_pool, np.random.randint(0, 3), replace=False):
                row[s] = 1
            row['disease'] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['disease'])

    X = df[SYMPTOMS].values
    y = df['target'].values

    model = RandomForestClassifier(n_estimators=200, max_depth=20,
                                   random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, le


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #0d47a1;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; color: #666; font-size: 1rem; margin-bottom: 2rem;
    }
    .result-card {
        background: white; border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    .disease-name { font-size: 1.3rem; font-weight: 700; color: #212121; }
    .confidence   { font-size: 1rem; color: #555; margin-top: 0.2rem; }
    .desc-text    { color: #555; font-size: 0.9rem; margin-top: 0.5rem; }
    .advice-box   {
        background: #f0f4ff; border-radius: 8px;
        padding: 0.7rem 1rem; margin-top: 0.6rem; color: #1a237e;
        font-size: 0.88rem;
    }
    .bar-wrap { background: #eee; border-radius: 20px; height: 10px; margin-top: 8px; }
    .bar-fill { height: 10px; border-radius: 20px; }
    .sev-badge {
        display: inline-block; border-radius: 20px;
        padding: 2px 12px; font-size: 0.75rem; font-weight: 700;
        margin-left: 8px; vertical-align: middle;
    }
    .sev-low    { background: #e8f5e9; color: #2e7d32; }
    .sev-medium { background: #fff3e0; color: #e65100; }
    .sev-high   { background: #ffebee; color: #b71c1c; }
    .tip-box {
        background: #fffde7; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #f9a825;
        color: #555; font-size: 0.88rem; margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 MediPredict AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Based Disease Prediction System | Select your symptoms and get instant predictions</div>',
            unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – SYMPTOM SELECTION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Select Symptoms")
    st.caption("Check all symptoms you are currently experiencing:")
    st.markdown("---")

    selected_symptoms = []
    for category, syms in SYMPTOM_CATEGORIES.items():
        with st.expander(category, expanded=False):
            for s in syms:
                label = s.replace('_', ' ').title()
                if st.checkbox(label, key=s):
                    selected_symptoms.append(s)

    st.markdown("---")
    st.markdown(f"**Selected:** {len(selected_symptoms)} symptom(s)")
    predict_btn = st.button("🔍 Predict Disease", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────────────────────────────────────
model, le = train_model()

if predict_btn:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom from the sidebar.")
    else:
        # Build feature vector
        feature_vector = np.array([1 if s in selected_symptoms else 0
                                   for s in SYMPTOMS]).reshape(1, -1)

        # Get probabilities for all 15 diseases
        proba = model.predict_proba(feature_vector)[0]

        # Sort descending
        top_indices = np.argsort(proba)[::-1][:5]

        st.markdown("### 🎯 Prediction Results")
        st.caption(f"Based on **{len(selected_symptoms)}** selected symptom(s)")

        rank_labels  = ["🥇 Most Likely", "🥈 Second Most Likely", "🥉 Third Most Likely",
                        "4th", "5th"]

        for rank, idx in enumerate(top_indices):
            if proba[idx] < 0.01:          # skip negligible
                continue
            disease    = le.inverse_transform([idx])[0]
            confidence = proba[idx] * 100
            info       = DISEASE_INFO.get(disease, {})
            sev        = info.get('severity', 'Medium')
            color      = info.get('color', '#FF9800')
            desc       = info.get('desc', '')
            advice     = info.get('advice', '')

            sev_class = {'Low': 'sev-low', 'Medium': 'sev-medium', 'High': 'sev-high'}.get(sev, 'sev-medium')

            st.markdown(f"""
            <div class="result-card" style="border-left-color:{color}">
                <div>
                    <span style="font-size:0.8rem;color:#999;font-weight:600">{rank_labels[rank]}</span>
                    <span class="sev-badge {sev_class}">{sev} Severity</span>
                </div>
                <div class="disease-name">{disease}</div>
                <div class="confidence">Confidence: <strong>{confidence:.1f}%</strong></div>
                <div class="bar-wrap">
                    <div class="bar-fill" style="width:{min(confidence,100):.1f}%;background:{color}"></div>
                </div>
                <p class="desc-text">{desc}</p>
                <div class="advice-box">💡 <strong>What to do:</strong> {advice}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box">
            ⚠️ <strong>Important Disclaimer:</strong> This AI prediction is based on symptom patterns
            in training data and is for <em>educational purposes only</em>. It is not a medical diagnosis.
            Always consult a qualified healthcare professional for proper medical advice and treatment.
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome / How-to screen
    col1, col2, col3 = st.columns(3)
    for col, icon, step, desc in [
        (col1, "👈", "1. Select Symptoms",  "Use the sidebar to check all symptoms you are currently experiencing."),
        (col2, "🔍", "2. Click Predict",    "Press the Predict Disease button to run the AI analysis."),
        (col3, "📋", "3. Review Results",   "View predicted conditions, confidence scores, severity, and health advice."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:white;border:1.5px solid #e0e0e0;border-radius:14px;
                        padding:1.8rem;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.06)">
                <div style="font-size:2.5rem">{icon}</div>
                <h4 style="color:#0d47a1;margin:.6rem 0 .4rem">{step}</h4>
                <p style="color:#666;font-size:.9rem;margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🦠 Diseases Covered by This System")

    disease_cols = st.columns(5)
    for i, (disease, info) in enumerate(DISEASE_INFO.items()):
        sev   = info['severity']
        color = info['color']
        with disease_cols[i % 5]:
            st.markdown(f"""
            <div style="background:white;border:1px solid #e8e8e8;border-radius:10px;
                        padding:.8rem;text-align:center;margin-bottom:.6rem;
                        border-top:3px solid {color}">
                <div style="font-weight:600;font-size:.85rem;color:#212121">{disease}</div>
                <div style="font-size:.72rem;color:{color};font-weight:600">{sev}</div>
            </div>
            """, unsafe_allow_html=True)

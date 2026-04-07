import streamlit as st

# MUST be first
st.set_page_config(page_title="AI Disease Prediction", layout="wide")

import os, pickle, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

MODEL_DIR = "model"

# -------------------- TRAIN MODEL --------------------
def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    SYMPTOMS = [
        'fever','high_fever','chills','fatigue','malaise',
        'headache','body_ache','joint_pain','muscle_pain','back_pain',
        'cough','dry_cough','productive_cough','shortness_of_breath','chest_pain',
        'sore_throat','runny_nose','nasal_congestion','sneezing','loss_of_smell',
        'nausea','vomiting','diarrhea','stomach_pain','loss_of_appetite',
        'skin_rash','itching','yellowish_skin','pale_skin','sweating',
        'dizziness','blurred_vision','excessive_thirst','frequent_urination',
        'weight_loss','weight_gain','swollen_lymph_nodes','stiff_neck',
        'sensitivity_to_light','red_eyes','dark_urine','clay_colored_stool'
    ]

    DISEASE_SYMPTOM_MAP = {
        'Common Cold': {'primary':['runny_nose','nasal_congestion','sneezing','sore_throat','cough'],
                        'secondary':['fatigue','headache','body_ache','fever']},
        'Influenza': {'primary':['high_fever','body_ache','fatigue','headache','chills'],
                      'secondary':['cough','sore_throat','muscle_pain']},
        'COVID-19': {'primary':['fever','dry_cough','fatigue','loss_of_smell','shortness_of_breath'],
                     'secondary':['body_ache','headache','sore_throat']},
        'Malaria': {'primary':['high_fever','chills','sweating','headache','body_ache'],
                    'secondary':['nausea','vomiting','fatigue']},
    }

    rows = []
    np.random.seed(42)

    for disease, groups in DISEASE_SYMPTOM_MAP.items():
        for _ in range(150):
            row = {s: 0 for s in SYMPTOMS}

            for s in groups['primary']:
                if s in SYMPTOMS:
                    row[s] = np.random.choice([0,1], p=[0.2,0.8])

            for s in groups['secondary']:
                if s in SYMPTOMS:
                    row[s] = np.random.choice([0,1], p=[0.5,0.5])

            row['disease'] = disease
            rows.append(row)

    df = pd.DataFrame(rows)

    le = LabelEncoder()
    df['target'] = le.fit_transform(df['disease'])

    X = df[SYMPTOMS]
    y = df['target']

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)

    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{MODEL_DIR}/encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open(f"{MODEL_DIR}/symptoms.pkl", "wb") as f:
        pickle.dump(SYMPTOMS, f)

# -------------------- LOAD MODEL --------------------
@st.cache_data   # 🔥 safe for all versions
def load_model():
    if not os.path.exists(f"{MODEL_DIR}/model.pkl"):
        with st.spinner("Training model for first time..."):
            train_and_save()

    with open(f"{MODEL_DIR}/model.pkl","rb") as f:
        model = pickle.load(f)

    with open(f"{MODEL_DIR}/encoder.pkl","rb") as f:
        le = pickle.load(f)

    with open(f"{MODEL_DIR}/symptoms.pkl","rb") as f:
        symptoms = pickle.load(f)

    return model, le, symptoms

# -------------------- UI --------------------
st.title("🧠 AI Disease Prediction System")

model, le, symptoms = load_model()

selected = st.multiselect("Select your symptoms:", symptoms)

if st.button("Predict Disease"):
    if not selected:
        st.warning("Please select at least one symptom")
    else:
        input_data = [1 if s in selected else 0 for s in symptoms]
        prediction = model.predict([input_data])[0]
        disease = le.inverse_transform([prediction])[0]

        st.success(f"Predicted Disease: {disease}")

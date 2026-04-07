# At the top of app.py — add this BEFORE st.set_page_config()

import os, pickle, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_DIR = "model"

def train_and_save():
    """Train model if not already saved"""
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
        'Common Cold':  {'primary':['runny_nose','nasal_congestion','sneezing','sore_throat','cough'],
                         'secondary':['fatigue','headache','body_ache','fever','malaise']},
        'Influenza':    {'primary':['high_fever','body_ache','fatigue','headache','chills'],
                         'secondary':['cough','sore_throat','muscle_pain','sweating','malaise']},
        'COVID-19':     {'primary':['fever','dry_cough','fatigue','loss_of_smell','shortness_of_breath'],
                         'secondary':['body_ache','headache','sore_throat','chest_pain','diarrhea']},
        'Pneumonia':    {'primary':['high_fever','productive_cough','chest_pain','shortness_of_breath','chills'],
                         'secondary':['fatigue','nausea','sweating','body_ache','loss_of_appetite']},
        'Bronchitis':   {'primary':['productive_cough','chest_pain','shortness_of_breath','fatigue','chills'],
                         'secondary':['fever','sore_throat','body_ache','headache','sweating']},
        'Malaria':      {'primary':['high_fever','chills','sweating','headache','body_ache'],
                         'secondary':['nausea','vomiting','fatigue','malaise','joint_pain']},
        'Dengue Fever': {'primary':['high_fever','headache','joint_pain','skin_rash','muscle_pain'],
                         'secondary':['nausea','vomiting','fatigue','red_eyes','body_ache']},
        'Typhoid':      {'primary':['high_fever','stomach_pain','headache','loss_of_appetite','malaise'],
                         'secondary':['nausea','vomiting','diarrhea','fatigue','sweating']},
        'Hepatitis B':  {'primary':['yellowish_skin','dark_urine','clay_colored_stool','fatigue','stomach_pain'],
                         'secondary':['nausea','vomiting','loss_of_appetite','joint_pain','fever']},
        'Tuberculosis': {'primary':['productive_cough','weight_loss','sweating','fatigue','chest_pain'],
                         'secondary':['fever','chills','loss_of_appetite','body_ache','malaise']},
        'Diabetes':     {'primary':['excessive_thirst','frequent_urination','weight_loss','fatigue','blurred_vision'],
                         'secondary':['loss_of_appetite','nausea','dizziness','itching','skin_rash']},
        'Hypertension': {'primary':['headache','dizziness','blurred_vision','chest_pain','shortness_of_breath'],
                         'secondary':['fatigue','nausea','back_pain','body_ache','malaise']},
        'Migraine':     {'primary':['headache','sensitivity_to_light','nausea','blurred_vision','dizziness'],
                         'secondary':['vomiting','fatigue','body_ache','loss_of_appetite','malaise']},
        'Chickenpox':   {'primary':['skin_rash','itching','fever','fatigue','loss_of_appetite'],
                         'secondary':['headache','body_ache','sore_throat','malaise','chills']},
        'Meningitis':   {'primary':['stiff_neck','high_fever','headache','sensitivity_to_light','nausea'],
                         'secondary':['vomiting','chills','skin_rash','fatigue','malaise']},
    }

    np.random.seed(42)
    rows = []
    for disease, groups in DISEASE_SYMPTOM_MAP.items():
        primary   = [s for s in groups['primary']   if s in SYMPTOMS]
        secondary = [s for s in groups['secondary'] if s in SYMPTOMS]
        for _ in range(200):
            row = {s: 0 for s in SYMPTOMS}
            for s in primary:   row[s] = int(np.random.random() > 0.20)
            for s in secondary: row[s] = int(np.random.random() > 0.55)
            noise_pool = [s for s in SYMPTOMS if s not in primary and s not in secondary]
            for s in np.random.choice(noise_pool, np.random.randint(0,3), replace=False): row[s] = 1
            row['disease'] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df['disease_encoded'] = le.fit_transform(df['disease'])
    X = df[SYMPTOMS]
    y = df['disease_encoded']

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X, y)

    with open(f"{MODEL_DIR}/disease_model.pkl","wb") as f: pickle.dump(model, f)
    with open(f"{MODEL_DIR}/label_encoder.pkl","wb") as f: pickle.dump(le, f)
    with open(f"{MODEL_DIR}/symptoms_list.pkl","wb") as f: pickle.dump(SYMPTOMS, f)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(f"{MODEL_DIR}/disease_model.pkl"):
        with st.spinner("🔄 Training AI model for first time... (30 seconds)"):
            train_and_save()
    with open(f"{MODEL_DIR}/disease_model.pkl","rb") as f: model    = pickle.load(f)
    with open(f"{MODEL_DIR}/label_encoder.pkl","rb") as f: le       = pickle.load(f)
    with open(f"{MODEL_DIR}/symptoms_list.pkl","rb") as f: symptoms = pickle.load(f)
    return model, le, symptoms
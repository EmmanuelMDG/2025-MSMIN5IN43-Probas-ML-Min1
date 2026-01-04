import streamlit as st
import torch
import sys
from pathlib import Path
from transformers import CamembertTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.camembert_multitask import CamemBERTMultitask
from models.config import ModelConfig, EMOTION_LABELS, SENTIMENT_LABELS, IRONY_LABELS
from data.preprocessing import clean_text

st.set_page_config(page_title="Analyse de Sentiment Multi-tâches", page_icon="🎭", layout="wide")

@st.cache_resource
def load_model():
    config = ModelConfig()
    model = CamemBERTMultitask(config)
    # Try to load from models folder relative to this script or root
    # Le script d'entraînement sauvegarde sous "best_model.pt"
    model_path = Path(__file__).parent.parent.parent / "models" / "best_model.pt"
    
    if model_path.exists():
        try:
            # weights_only=False est nécessaire car on charge un dictionnaire contenant des objets custom (ModelConfig)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            # Le checkpoint contient un dictionnaire, on veut juste le state_dict du modèle
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            return model
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {e}")
            return None
    else:
        st.warning(f"Modèle non trouvé à {model_path}. Veuillez entraîner le modèle d'abord.")
        return None

@st.cache_resource
def load_tokenizer():
    return CamembertTokenizer.from_pretrained("camembert-base")

st.title("🎭 Analyse de Sentiment Multi-tâches")
st.markdown("""
Ce démonstrateur utilise un modèle **CamemBERT** fine-tuné pour trois tâches simultanées :
1. **Détection d'émotions** (7 classes)
2. **Analyse de sentiment** (Positif/Négatif/Neutre)
3. **Détection d'ironie** (Ironique/Non-ironique)
""")

text = st.text_area("Entrez un texte à analyser :", "Ce film est vraiment génial, je l'adore ! ❤️")

if st.button("Analyser", type="primary"):
    if not text:
        st.error("Veuillez entrer un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            model = load_model()
            if model:
                model.eval()
                tokenizer = load_tokenizer()
                
                # Preprocessing
                cleaned_text = clean_text(text)
                
                # Tokenization
                inputs = tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                # Inference
                with torch.no_grad():
                    outputs = model(inputs['input_ids'], inputs['attention_mask'])
                    
                # Results
                emotion_probs = torch.softmax(outputs['emotion_logits'], dim=1)[0]
                sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1)[0]
                irony_probs = torch.softmax(outputs['irony_logits'], dim=1)[0]
                
                # Display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("😊 Émotion")
                    best_emotion_idx = torch.argmax(emotion_probs).item()
                    best_emotion_label = EMOTION_LABELS[best_emotion_idx]
                    st.metric("Émotion dominante", best_emotion_label.capitalize())
                    
                    # Chart with labels
                    chart_data = {EMOTION_LABELS[i]: prob.item() for i, prob in enumerate(emotion_probs)}
                    st.bar_chart(chart_data)
                    
                with col2:
                    st.subheader("👍 Sentiment")
                    best_sentiment_idx = torch.argmax(sentiment_probs).item()
                    best_sentiment_label = SENTIMENT_LABELS[best_sentiment_idx]
                    st.metric("Sentiment", best_sentiment_label.capitalize())
                    
                    chart_data = {SENTIMENT_LABELS[i]: prob.item() for i, prob in enumerate(sentiment_probs)}
                    st.bar_chart(chart_data)

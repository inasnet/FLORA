from pathlib import Path
import io
import json
import time
import hashlib
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
SCALES = {'globale': 1.0, 'intermediaire': 0.60, 'locale': 0.35}
NAMES = {'globale': 'Globale', 'intermediaire': 'Intermédiaire', 'locale': 'Locale'}
CLASSES = ['Tomato_healthy', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Septoria_leaf_spot']
LABELS = ['Feuille saine', 'Alternariose', 'Mildiou', 'Septoriose']
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
torch.set_num_threads(min(4, torch.get_num_threads()))


def crop_image(image, scale):
    fraction = SCALES[scale]
    if fraction == 1:
        return image.copy()
    w, h = image.size
    cw, ch = int(w * fraction), int(h * fraction)
    if min(cw, ch) < 1:
        raise ValueError('Image trop petite pour ce recadrage.')
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


@st.cache_resource
def load_model(scale, modified):
    # Le constructeur avec poids ImageNet du notebook active transform_input.
    model = models.googlenet(weights=None, aux_logits=False, transform_input=True, init_weights=False, num_classes=4)
    state = torch.load(ROOT / 'models' / f'best_{scale}.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(image, scale):
    path = ROOT / 'models' / f'best_{scale}.pt'
    model = load_model(scale, path.stat().st_mtime_ns)
    with torch.inference_mode():
        return torch.softmax(model(TF(crop_image(image, scale)).unsqueeze(0)), dim=1)[0].tolist()


st.set_page_config(page_title='Folia · Analyse des feuilles', page_icon='🌿', layout='wide')
st.markdown('''<style>
.stApp {background:#f5f8f4;color:#173b2b}
.block-container {max-width:1250px;padding-top:2.5rem}
h1,h2,h3 {color:#173b2b}
[data-testid="stSidebar"] {background:#e9f0e6}
[data-testid="stMetric"] {background:white;border:1px solid #dce6d7;border-radius:16px;padding:18px}
.stButton>button {border-radius:10px;min-height:48px}
.hero {background:#183f2d;color:#f4faef;padding:30px 36px;border-radius:22px;margin-bottom:25px}
.hero h1 {color:#f4faef;font-size:2.6rem;margin:4px 0 12px}
.hero p {color:#d4e5ce;max-width:700px}
.eyebrow {font-size:12px;letter-spacing:3px;color:#b8d9a6}
</style>''', unsafe_allow_html=True)
with st.sidebar:
    st.title('🌿 Folia')
    st.caption('LABORATOIRE VÉGÉTAL')
    st.divider()
    st.subheader('Échelles d’observation')
    selected = st.multiselect('Modèles à comparer', list(SCALES), default=list(SCALES), format_func=lambda k: NAMES[k])
    st.caption('Globale : image entière · Intermédiaire : centre à 60 % · Locale : centre à 35 %.')
    st.divider()
    st.markdown('**4 classes de tomate**')
    for label in LABELS:
        st.write('• ' + label)
    st.caption('GoogLeNet · calcul local sur CPU')
    with st.expander('État des modèles'):
        for scale in SCALES:
            present = (ROOT / 'models' / f'best_{scale}.pt').is_file()
            st.write(('✓ ' if present else '✕ ') + NAMES[scale])

st.markdown('''<div class="hero"><div class="eyebrow">ANALYSE FOLIAIRE · TOMATE</div><h1>Une feuille, trois regards.</h1><p>Importez une photo et comparez les prédictions de vos modèles aux trois échelles d’observation.</p></div>''', unsafe_allow_html=True)
left, right = st.columns([1, 1.15], gap='large')
with left:
    st.subheader('01 · Votre photographie')
    uploaded = st.file_uploader('Photo d’une feuille de tomate', type=['jpg', 'jpeg', 'png', 'bmp'], help='20 Mo maximum. Placez la feuille au centre de la photo.')
    st.caption('Choisissez une photo nette, avec la feuille au centre et un fond simple.')
image = None
if uploaded:
    try:
        if uploaded.size > 20 * 1024 * 1024:
            raise ValueError('La photo dépasse 20 Mo.')
        with Image.open(io.BytesIO(uploaded.getvalue())) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError('La photo dépasse 25 millions de pixels.')
            image = ImageOps.exif_transpose(source).convert('RGB')
        if min(image.size) < 32:
            raise ValueError('La photo doit mesurer au moins 32 × 32 pixels.')
        with left:
            st.image(image, caption=f'{uploaded.name} · {image.width} × {image.height} px', use_container_width=True)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(f'Photo illisible : {exc}')

with right:
    st.subheader('02 · Les vues analysées')
    if image:
        views = st.columns(3)
        for col, scale in zip(views, SCALES):
            with col:
                st.image(crop_image(image, scale).resize((224, 224)), caption=f'{NAMES[scale]} · {SCALES[scale]:.0%}', use_container_width=True)
    else:
        st.info('Importez une photo pour voir les trois cadrages et lancer l’analyse.')
    run = st.button('Analyser la feuille', type='primary', use_container_width=True, disabled=image is None or not selected)
    st.caption('Les scores expriment les probabilités du modèle parmi les quatre classes apprises. Ils ne mesurent pas sa fiabilité sur cette photo.')

key = hashlib.sha256(uploaded.getvalue()).hexdigest() + str(selected) if image else None
if st.session_state.get('result_key') != key:
    st.session_state.pop('results', None)
if run:
    results, errors = {}, {}
    with st.spinner('Analyse des cadrages en cours…'):
        for scale in selected:
            try:
                results[scale] = predict(image, scale)
            except Exception as exc:
                errors[scale] = str(exc)
    st.session_state.update(results=results, errors=errors, result_key=key)

if st.session_state.get('result_key') == key and key:
    for scale, error in st.session_state.get('errors', {}).items():
        st.error(f'Modèle {NAMES[scale]} indisponible : {error}')
    results = st.session_state.get('results', {})
    if results:
        st.divider()
        st.subheader('03 · Résultats de l’analyse')
        cards = st.columns(len(results))
        rows = []
        for col, (scale, probabilities) in zip(cards, results.items()):
            winner = max(range(4), key=lambda i: probabilities[i])
            with col:
                st.metric(NAMES[scale], LABELS[winner], f'{probabilities[winner]:.1%} · score du modèle', delta_color='off')
                for i in sorted(range(4), key=lambda i: probabilities[i], reverse=True):
                    st.progress(probabilities[i], text=f'{LABELS[i]} · {probabilities[i]:.1%}')
            rows.extend({'Échelle': NAMES[scale], 'Classe': CLASSES[i], 'Résultat': LABELS[i], 'Probabilité': p} for i, p in enumerate(probabilities))
        winners = [max(range(4), key=lambda i: p[i]) for p in results.values()]
        if len(winners) > 1:
            if len(set(winners)) == 1:
                st.success('Les modèles sélectionnés donnent la même classe.')
            else:
                st.warning('Les modèles divergent selon le cadrage. Consultez les détails de chaque échelle.')
        st.download_button('Télécharger les résultats CSV', pd.DataFrame(rows).to_csv(index=False).encode('utf-8-sig'), 'analyse_feuille.csv', 'text/csv')
st.divider()
st.caption('Projet expérimental entraîné sur PlantVillage. Les performances sur des photos de terrain peuvent différer. Le modèle attribue une des quatre classes à toute image ; il ne vérifie pas qu’il s’agit d’une feuille de tomate.')

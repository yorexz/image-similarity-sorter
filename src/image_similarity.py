import os
import sys
import numpy as np
import uuid
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Erro ao processar a imagem {img_path}: {e}")
        return None

def extract_features(model, image_paths, target_size=(224, 224)):
    logger.info(f"Extraindo características de {len(image_paths)} imagens...")
    
    features = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        img_array = load_and_preprocess_image(img_path, target_size)
        if img_array is not None:
            feature = model.predict(img_array, verbose=0)
            features.append(feature.flatten())
            valid_paths.append(img_path)
    
    return np.array(features), valid_paths

def order_images_in_place(image_folder, extensions=('.jpg', '.jpeg', '.png')):
    logger.info("Carregando modelo ResNet50...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    logger.info(f"Encontrando imagens em {image_folder}...")
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        logger.error(f"Nenhuma imagem encontrada em {image_folder}")
        return []
    
    logger.info(f"Encontradas {len(image_paths)} imagens")
    
    features, valid_paths = extract_features(model, image_paths)
    
    if len(valid_paths) == 0:
        logger.error("Nenhuma imagem pôde ser processada")
        return []
    
    logger.info("Calculando similaridades...")
    similarity_matrix = cosine_similarity(features)
    
    logger.info("Ordenando imagens por similaridade visual...")
    tsne = TSNE(n_components=1, random_state=42)
    embedded = tsne.fit_transform(features).flatten()
    
    ordered_indices = np.argsort(embedded)
    ordered_paths = [valid_paths[i] for i in ordered_indices]
    
    temp_paths = []
    for img_path in tqdm(valid_paths, desc="Preparando reorganização"):
        folder, filename = os.path.split(img_path)
        _, ext = os.path.splitext(filename)
        
        temp_name = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(folder, temp_name)
        
        os.rename(img_path, temp_path)
        temp_paths.append(temp_path)
    
    final_paths = []
    for i, idx in enumerate(tqdm(ordered_indices, desc="Reorganizando imagens")):
        temp_path = temp_paths[idx]
        folder, _ = os.path.split(temp_path)
        _, ext = os.path.splitext(temp_path)
        
        new_name = f"{i+1:04d}{ext}"
        new_path = os.path.join(folder, new_name)
        
        os.rename(temp_path, new_path)
        final_paths.append(new_path)
        
        logger.info(f"Renomeado: {os.path.basename(valid_paths[idx])} -> {new_name}")
    
    logger.info(f"Concluído! {len(final_paths)} imagens foram ordenadas por similaridade na pasta {image_folder}")
    
    return final_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ordenar_imagens_inplace.py pasta_com_imagens")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    
    ordered_images = order_images_in_place(image_folder)
    
    print("\nConcluído! As imagens foram ordenadas por similaridade na pasta:", image_folder)
    print("As imagens agora têm nomes como 0001.jpg, 0002.jpg, etc. em ordem de similaridade.")
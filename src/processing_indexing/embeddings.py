# this file is resposible of taking the output of the chunking.py functions
# and making embeddings out of them

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from PIL import Image

# initiating model, this is gonna take a while...
model = SentenceTransformer('clip-ViT-B-32')

def embed_text(text_list):
    text_emb = model.encode(text_list)
    print(text_emb.shape)
    return text_emb

def embed_image(path):
    img_emb = model.encode(Image.open(path))
    print(img_emb.shape)
    return img_emb

def embed_video(filenames):
    opened_video_list = []
    for n in range(len(filenames)):
        opened_video_list.append(Image.open(filenames[n]))
    
    videos_emb = model.encode(opened_video_list)
    print(videos_emb.shape)
    return videos_emb

# embed_audio implement!
# this file is resposible of taking the output of the chunking.py functions
# and making embeddings out of them

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from PIL import Image

# initiating model, this is gonna take a while...
model = SentenceTransformer('clip-ViT-B-32')

def embed_text(text_list):
    # dictionary for returning the embeddings with the text
    package = {}

    # saving embeddings with text for retrieval
    for n in range(len(text_list)):
        text_emb = model.encode(text_list[n])
        package[text_list[n]] = text_emb
    print(package)
    return package

def embed_image(path):
    # dictionary for returning the embeddings with the file path
    package = {}
    
    img_emb = model.encode(Image.open(path))
    package[path] = img_emb
    print(package)
    return package

def embed_video(filenames):
    # same thing from above 
    package = {}

    for n in range(len(filenames)):
        video_emb = model.encode(Image.open(filenames[n]))
        package[filenames[n]] = video_emb

    print(package)
    return package

# embed_audio implement!
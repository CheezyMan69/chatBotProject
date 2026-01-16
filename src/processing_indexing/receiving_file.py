# this module should be responsible of receiving the bytes from the front_end
# processing them, categorize them, making embeddings out of them and then indexing them in Pinecone
# note for how streamlit packages data types:
# txt = text/plain
# images = image/png - jpeg - jpg
# video = video/mp4
# audio = audio/mpeg

from pinecone import Pinecone
import cv2


def detect_dtype(file, dtype):
    match dtype:
        case "text/plain":
            print("The filetype is text")
            chunk_text(file)
        case "image/png":
            print("The filetype is an image")
            chunk_image(file)
        case "image/jpeg":
            print("The filetype is an image")
            chunk_image(file)
        case "image/jpg":
            print("The filetype is an image")
            chunk_image(file)
        case "video/mp4":
            print("The filetype is a video")
            chunk_video(file)
        case "audio/mpeg":
            print("The filetype is an audio")
            chunk_audio(file)

def chunk_text(file):
    

def chunk_image(image):

def chunk_video(video):

def chunk_audio(audio):
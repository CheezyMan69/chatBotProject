# this module should be responsible of receiving the bytes from the front_end
# processing them, categorize them, making embeddings out of them and then indexing them in Pinecone
# note for how streamlit packages data types:
# txt = text/plain
# images = image/png - jpeg - jpg
# video = video/mp4
# audio = audio/mpeg

from pinecone import Pinecone
import cv2
from io import StringIO
from langchain_google_genai import ChatGoogleGenerativeAI
import os

api_gemini = os.environ['GEMINI']

def detect_dtype(file, dtype, path="nope"):
    print("The file has been saved to: ", path)
    match dtype:
        case "text/plain":
            print("The filetype is text")
            # embed_text(chunk_text(file)) implement!
        case "image/png":
            print("The filetype is an image")
            # embed_image(path) implement!
        case "image/jpeg":
            print("The filetype is an image")
            # embed_image(path) implement!
        case "image/jpg":
            print("The filetype is an image")
            # embed_image(path) implement!
        case "video/mp4":
            print("The filetype is a video")
            # embed_video(chunk_video(file, path)) implement!
        case "audio/mpeg":
            print("The filetype is an audio")
            # embed_audio(chunk_audio(path)) implement!

# different chunking strategies based on the file
def chunk_text(file):
    string_data = StringIO(file.getvalue().decode("utf-8")).getvalue()

    # use agentic chunking to create text chunks
    chunking_model = ChatGoogleGenerativeAI(google_api_key=api_gemini, model='gemini-2.5-flash')

    messages = [("system", """You're an AI Agent specialized in the pre-processing of data before being used to create embeddings.

                            Given the data you're handled, create chunks that semantically make sense, grouping pieces of the text
                            by ideas.

                            Output the data in this format: n_chunk_here * n+1_chunk_here * n+2_chunk_here * ...

                            IMPORTANT:
                            - Make sure to use asterisks as delimitators between chunks
                            - Make sure not to include text formatting characters like '\n' or bullet points"""),
                ("user", string_data)]
    
    response = chunking_model.invoke(messages)
    text_list = response.content.split("*")
    print("After agentic chunking: ", text_list)
    return text_list

def chunk_video(video, path):
    # frames from the video
    video_ingested = cv2.VideoCapture(path)
    frame_counter = 1
    another_frame_counter = 1

    # get how many frames the video has
    while True:
        ret, frame = video_ingested.read()
        if not ret:
            break
        
        frame_counter += 1

    # get five frames depending on how long the video is
    class frames:
        second_frame = frame_counter * 2 / 5
        third_frame = frame_counter * 3 / 5
        fourth_frame = frame_counter * 4 / 5
        fifth_frame = frame_counter * 5 / 5
    
    print("total frames in the video: ", frame_counter)
    
    # re-open video so that frame starts at 0
    video_ingested_2 = cv2.VideoCapture(path)
    
    # create list of filenames for embeddings afterwards
    filenames = []

    # extract 5 frames along the video
    while True:
        ret, frame = video_ingested_2.read()
        if not ret:
            break
        
        match another_frame_counter:
            case 1:
                name_without_ending = video.name.replace('.mp4', "")
                filename = f"data/{name_without_ending}_frame1.png"
                cv2.imwrite(filename, frame)
                filenames.append(filename)
                print("frame: ", another_frame_counter, "reached")
            case frames.second_frame:
                name_without_ending = video.name.replace('.mp4', "")
                filename = f"data/{name_without_ending}_frame{frames.second_frame}.png"
                cv2.imwrite(filename, frame)
                filenames.append(filename)
                print("frame: ", another_frame_counter, "reached")
            case frames.third_frame:
                name_without_ending = video.name.replace('.mp4', "")
                filename = f"data/{name_without_ending}_frame{frames.third_frame}.png"
                cv2.imwrite(filename, frame)
                filenames.append(filename)
                print("frame: ", another_frame_counter, "reached")
            case frames.fourth_frame:
                name_without_ending = video.name.replace('.mp4', "")
                filename = f"data/{name_without_ending}_frame{frames.fourth_frame}.png"
                cv2.imwrite(filename, frame)
                filenames.append(filename)
                print("frame: ", another_frame_counter, "reached")
            case frames.fifth_frame:
                name_without_ending = video.name.replace('.mp4', "")
                filename = f"data/{name_without_ending}_frame{frames.fifth_frame}.png"
                cv2.imwrite(filename, frame)
                filenames.append(filename)
                print("frame: ", another_frame_counter, "reached")
        another_frame_counter += 1

    return filenames

def chunk_audio(path):
    # implement!, chunk it a bit before sending it to embedding
    pass
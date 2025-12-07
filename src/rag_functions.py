# performing similarity search based on the user input
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import find_dotenv, load_dotenv
import os
from src.helpers.helpers import load_image_base64

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

pc = Pinecone(api_key = os.getenv('PINECONE'))
index = pc.Index(host='https://secondrag-4hezoud.svc.aped-4627-b74a.pinecone.io')


def similarity_search(query: str, top_k : int, namespace : str):
    # create embedding of the user query
    model = SentenceTransformer('clip-ViT-B-32')
    query_embedding = model.encode(query).tolist()

    # similarity search
    result = index.query(
        namespace = namespace,
        vector = query_embedding,
        top_k = top_k,
        include_metadata = True,
        include_values = False
    )
    
    return result

def get_relevant_metadata(result):
    # iterate over metadata from the matches and turn it into a list
    dict_values_list = []
    for n in range(len(result.matches)):
        title_of_metadata = result.matches[n].get('metadata').values() 
        value_turned_list = list(title_of_metadata) # turn into list to avoid "dict_values()" view
        dict_values_list.append(value_turned_list[0]) # always grab the first one because we create create a list for each value

    return dict_values_list


def gemini_call_normal(prompt: str):
    # simple call without any extra context
    gemini = ChatGoogleGenerativeAI(google_api_key = os.getenv('GEMINI'), model='gemini-2.5-flash')
    messages = [("system", "You're a helpful assistant"),
               ("user", prompt)]
    response = gemini.invoke(messages)
    return response.content

def gemini_call_rag(prompt:str):
    # get metadata from similarity search based on user prompt
    result_text = get_relevant_metadata(similarity_search(prompt, 5, "text"))
    result_pathnames_images = get_relevant_metadata(similarity_search(prompt, 3, "img"))

    # prepare to send extra content for RAG response
    system_msg = SystemMessage("You're a helpful assistant")

    # encoding data into base64 format because that's how the gemini-langchain interface works

    image_url_1 = result_pathnames_images[0]
    image_url_2 = result_pathnames_images[1]
    image_url_3 = result_pathnames_images[2]

    image_data_1 = load_image_base64(image_url_1)
    image_data_2 = load_image_base64(image_url_2)
    image_data_3 = load_image_base64(image_url_3)


    human_msg = HumanMessage(
            content=[
                {"type" : "text", "text" : prompt},
                {"type" : "text", "text": result_text[0]},
                {"type" : "text", "text": result_text[1]},
                {"type" : "text", "text": result_text[2]},
                {"type" : "text", "text": result_text[3]},
                {"type" : "text", "text": result_text[4]},
                {"type": "image_url", "image_url": {"url" : f"data:image/jpeg;base64,{image_data_1}"}},
                {"type": "image_url", "image_url": {"url" : f"data:image/jpeg;base64,{image_data_2}"}},
                {"type": "image_url", "image_url": {"url" : f"data:image/jpeg;base64,{image_data_3}"}},
            ]
    )

    # instantiate the model
    gemini = ChatGoogleGenerativeAI(google_api_key = os.getenv('GEMINI'), model='gemini-2.5-flash')
    response = gemini.invoke([system_msg, human_msg])

    return response.content




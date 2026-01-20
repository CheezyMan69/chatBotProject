# this file is responsible of taking the package from the functions in embeddings
# and put them correctly on pinecone
# we store the url of the stored resource in Pinecone for retrieval and sending to gemini format purposes

from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone
import os
import boto3
import uuid
import logging
from botocore.exceptions import ClientError

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# logging into Pinecone
pc = Pinecone(api_key = os.getenv('PINECONE_API_KEY'))
index = pc.Index(host='https://finalrag-clipvit-4hezoud.svc.aped-4627-b74a.pinecone.io')

# logging into amazon s3
s3 = boto3.client("s3", region_name="eu-north-1")

def upload_file(path):
    object_name = os.path.basename(path)

    # upload the file
    try:
        response = s3.upload_file(path, 'rag-cs1', object_name)
    except ClientError as e:
        logging.error(e)

    # get the signed url once it's done
    try:
        response_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': 'rag-cs1', 'Key': object_name},
            ExpiresIn=900,
        )
    except ClientError as e:
        logging.error(e)
        return None

    return response_url
    

def index_text(package):
    # upsert text vectors with raw text metadata
    vectors = []

    for key in package:
        vectors.append(
            {
            "id": str(uuid.uuid4()),
            "values": package[key].tolist(),
            "metadata": {"original_text": key}
            })

    index.upsert(vectors, namespace="text")    

def index_image(package):
    # there is only supposed to be one object in the package so this for loop is a formality
    vectors = []
    for key in package:
        # url = upload_file(key) moved this to be on demand while gemini calls it
        vectors.append({
        "id": os.path.basename(key),
        "values": package[key].tolist(),
        "metadata": {"path": key}
    })

    
    
    index.upsert(vectors, namespace="img")

def index_video(package):
    # list of links for each frame
    # urls = []

    # for key in package:
    #     # url = upload_file(key) moved this to be on demand while gemini calls it
    #     urls.append(url)

    vectors = []

    for key in package:
        vectors.append({
            "id": os.path.basename(key),
            "values": package[key].tolist(),
            "metadata": {"path": key}
        })
    
    index.upsert(vectors, namespace="img")
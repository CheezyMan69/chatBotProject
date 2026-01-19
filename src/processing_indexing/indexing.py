# this file is responsible of taking the package from the functions in embeddings
# and put them correctly on pinecone
# we store the url of the stored resource in Pinecone for retrieval and sending to gemini format purposes

from pinecone import Pinecone
import os
import boto3
import logging
from botocore.exceptions import ClientError

# logging into Pinecone
pc = Pinecone(api_key = os.environ['PINECONE'])
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
            "id": key,
            "values": package[key].tolist(),
            "metadata": {"original_text": key}
            })

    index.upsert(vectors, namespace="text")    

def index_image(package):
    # there is only supposed to be one object in the package so this for loop is a formality
    for key in package:
        url = upload_file(key)

    vectors = []
    vectors.append({
        "id": os.path.basename(key),
        "values": package[key].tolist(),
        "metadata": {"url": url}
    })
    index.upsert(vectors, namespace="img")

def index_video(package):
    # list of links for each frame
    urls = []

    for key in package:
        url = upload_file(key)
        urls.append(url)

    vectors = []
    extra_iterator = 0

    for key in package:
        vectors.append({
            "id": os.path.basename(key),
            "values": package[key].tolist(),
            "metadata": {"url": urls[extra_iterator]}
        })
        extra_iterator += 1
    
    index.upsert(vectors, namespace="img")
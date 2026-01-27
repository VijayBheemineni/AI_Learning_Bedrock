import boto3
import logging
from botocore.exceptions import BotoCoreError, ClientError
from typing import List

# Set Logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set Service Name
BEDROCK_SERVICE_NAME = "bedrock"

def get_bedrock_client():
    """Create and return a Bedrock client."""
    return boto3.client(service_name=BEDROCK_SERVICE_NAME)

def list_foundation_models(bedrock_client):
    """
    Fetch and return all Bedrock foundation models.
    """
    return bedrock_client.list_foundation_models()

def print_foundation_models(fm_models_list : list):
    """
    Print all Bedrock foundation model IDs.
    """
    for model in fm_models_list.get('modelSummaries'):
        print(model.get('modelId'))

def main():
    bedrock_client = get_bedrock_client()
    fm_models_list = list_foundation_models(bedrock_client)
    print_foundation_models(fm_models_list)
    

if __name__ == "__main__":
    main()


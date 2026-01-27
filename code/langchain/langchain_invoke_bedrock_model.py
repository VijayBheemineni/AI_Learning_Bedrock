import boto3
import logging
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import BedrockLLM
from typing import Dict

# Before running install 'langchain-aws' package

# Set Logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set variables
SERVICE_NAME = 'bedrock-runtime'
AWS_REGION_NAME = "us-west-2"
MODEL_ID = "amazon.titan-tg1-large"
BEDROCK_INFERENCE_SETTINGS_CONFIG = {
    "maxTokens": 512,
    "temperature": 0.5,
    "topP": 0.9,
    "maxTokenCount": 512
}
PROMPT = "You are AI Expert. I want to learn AI. Suggest good resources to learn AI"

def get_bedrock_runtime_client(aws_region_name: str) -> BaseClient:
    """
        This method returns bedrock runtime client
        :param aws_region_name: AWS region ID
        :return: Botocore BaseClient for Bedrock runtime
        :raises RuntimeError: If client creation fails
    """
    try:
        return boto3.client(SERVICE_NAME, region_name=aws_region_name)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(
            "Failed to create Bedrock Runtime Client"
        ) from e

def get_langchain_llm(
        bedrock_client: BaseClient, 
        model_id: str, 
        inference_config: dict, 
    ) -> BedrockLLM:
    """
        This function returns BedrockLLM object
        :params
        bedrock_client: BaseClient, 
            bedrock_client(BaseClient) :- Bedrock runtime client
            model_id(Str) :- Bedrock Model Id
            inference_config(Dict): Inference settings
        :return: BedrockLLM
    """
    llm = BedrockLLM(
        client = bedrock_client,
        model_id = model_id,
        model_kwargs = inference_config,
    )
    return llm

def invoke_model(
        langchain_bedrock_llm_model: BedrockLLM, 
        prompt: str
    ) -> str:
    """
        This function invokes Bedrock LLM model
        :params
            langchain_bedrock_llm_model(BedrockLLM) : BedrockLLM model
            prompt(str): user prompt
        :returns: str : Response from LLM
    """
    try:
        return langchain_bedrock_llm_model.invoke(input=prompt)
    except Exception as e:
        print("ERROR Invoking Bedrock model : {e}")


def main():
    bedrock_runtime = get_bedrock_runtime_client(aws_region_name=AWS_REGION_NAME)
    langchain_llm = get_langchain_llm(
        bedrock_runtime,
        MODEL_ID,
        BEDROCK_INFERENCE_SETTINGS_CONFIG,
    )
    response = invoke_model(langchain_llm, PROMPT)
    print(response)


if __name__ == "__main__":
    main()
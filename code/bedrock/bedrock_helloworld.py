import boto3
import json
from botocore.exceptions import ClientError


# Set Bedrock client
bedrock_client = boto3.client(service_name = "bedrock-runtime", region_name="us-west-2")

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html

# BEDROCK INFERENCE SETTINGS
# ERROR :- If we use "amazon.nova-lite-v1:0" we will get error. We need to use "inference" mode id. Bedrock Console --> Infer --> Cross-region reference --> Nove Lite and select US model id.
# ERROR: Can't invoke model '$amazon.nova-lite-v1:0'. Readon: An error occurred (ValidationException) when calling the InvokeModel operation: Invocation of model ID amazon.nova-lite-v1:0 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model.
MODEL_ID = "us.amazon.nova-lite-v1:0"
BEDROCK_INFERENCE_SETTINGS_CONFIG = {
    "maxTokens": 512,
    "temperature": 0.5,
    "topP": 0.9
}
SYSTEM_PROMPT = "Assume you are AI Expert"
PROMPT = "I love AI. Give me list of best resources to master AI"

# Request object for Nova Model
# https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html
native_request = {
    "system": [
        {
            "text": SYSTEM_PROMPT
        }
    ],
	"messages" : [
        {
            "role":"user",
            "content": [
                {
                    "text": PROMPT
                }
            ]
        }
	],
	"inferenceConfig": BEDROCK_INFERENCE_SETTINGS_CONFIG
}

# Convert string to json object
request = json.dumps(native_request)

try:
    model_response = bedrock_client.invoke_model(modelId=MODEL_ID, body=request)
    model_response_body = json.loads(model_response["body"].read())
    print(model_response_body)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke model '${MODEL_ID}'. Readon: {e}")


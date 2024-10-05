import json
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
]

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

# 1. create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

# Optionally create embedding function object
sentence_transformer_ef = SentenceTransformer("all-distilroberta-v1").encode
# client = QdrantClient(":memory:")
client = QdrantClient(host="localhost", port=6333)

# 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# Refer to https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/retrieve_user_proxy_agent
# and https://microsoft.github.io/autogen/docs/reference/agentchat/contrib/vectordb/qdrant
# for more information on the RetrieveUserProxyAgent and QdrantVectorDB
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/flaml/main/README.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        ],  # change this to your own path, such as https://raw.githubusercontent.com/microsoft/autogen/main/README.md
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "db_config": {"client": client},
        "vector_db": "qdrant",  # qdrant database
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": True,  # set to True if you want to overwrite an existing collection
        "embedding_function": sentence_transformer_ef,  # If left out fastembed "BAAI/bge-small-en-v1.5" will be used
        "customized_answer_prefix": "the answer is",
    },
    code_execution_config=False,
)


# reset the assistant. Always reset the assistant before starting a new conversation.
assistant.reset()

qa_problem = "Who is the author of FLAML?"
chat_results = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=qa_problem
)

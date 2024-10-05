import os

from dotenv import load_dotenv

import autogen

load_dotenv()
config_list_qwen = [
    {
        "base_url": "http://localhost:8080",
        "api_key": "NULL",
        "model": "ollama/qwen2.5:14b",  # qwen2.5:14b
    }
]
config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
]

config_azure_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
]


config_jamba = [
    {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "ai21labs/jamba-1.5-large-instruct",
        "api_key": os.getenv("NVIDIA_NIM_API_KEY"),
    },
]

llm_config = {
    "config_list": config_azure_list,
}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": True,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)
# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""
            今天是2024年9月28日,今天台灣發生了哪些政治新聞並以markdown形式呈現並儲存成output.md。
            """,
)

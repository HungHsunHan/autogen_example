import os

from dotenv import load_dotenv

import autogen
from autogen import GroupChat, GroupChatManager

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
expert_agent = autogen.AssistantAgent(
    name="expert_agent",
    llm_config=llm_config,
    description="""You understand the domain knowledge and can distinguish the contents 
                    the search agent gave us is relevant or not.If the contents is irrelevent,
                    tell him to use tool to search again until you think the content contains key words.
                    If you are done, transfer you understadning to the writing agent which will help you to summarize. 
                    """,
)
writing_agent = autogen.AssistantAgent(
    name="writing_agent",
    llm_config=llm_config,
    description="""You are a good writer who can summary all complex knowledge from expert_agent and 
                    write a readme file in a decent format.
                    """,
)
search_agent = autogen.AssistantAgent(
    name="search_agent",
    llm_config=llm_config,
    description="You know how to use search tools to search interested topics.",
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

group_chat = GroupChat(
    agents=[expert_agent, writing_agent, search_agent, user_proxy],
    messages=[],
    max_round=10,
)


group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)
chat_result = user_proxy.initiate_chat(
    group_chat_manager,
    message="""
            Help me to search the latest battery degradation paper online in arxiv and give me the research trend。
            """,
    summary_method="reflection_with_llm",
)

from dotenv import find_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.load_tools import get_all_tool_names
from langchain import ConversationChain
from langchain.utilities import SerpAPIWrapper

llm = OpenAI(openai_api_key="")
conversation = ConversationChain(llm=llm, verbose=True)

print(conversation.run("what is hello how are you in french then translated to german then translated back to english?"))

"""chat = ChatOpenAI(temperature=0, openai_api_key="")
response = chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
print (response.content)
"""

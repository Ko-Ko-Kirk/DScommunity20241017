# DScommunity20241017

這是在「資料科學家的工作日常」社群於 2024/10/17 的直播活動哦！

我將以 Live Demo 的方式打造一個入門級的 LangChain 應用。然後還會抽出兩本書來贈送給參加的朋友哦！！



1. `poetry new demo`
2. `cd demo`，並確定 python 版本
3. `poetry add langchain langchain-community langchain-openai beautifulsoup4 python-dotenv qdrant-client`

4. 程式碼之 RAG
```
#以《LangChain奇幻旅程》做為retrival資料來源的範例
#https://www.drmaster.com.tw/Bookinfo.asp?BookID=MP22437

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv("./.env")

aoai_api_key=os.getenv("AOAI_API_KEY")
aoai_endpoint=os.getenv("AOAI_ENDPOINT")
embed_deployment_name=os.getenv("AOAI_EMBED_DEPLOYMENT_NAME")
gpt_deployment_name=os.getenv("AOAI_GPT_DEPLOYMENT_NAME")

qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")

loader = WebBaseLoader("https://www.drmaster.com.tw/Bookinfo.asp?BookID=MP22437")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

model = AzureChatOpenAI(
    api_key=aoai_api_key,
    openai_api_version="2024-06-01",
    azure_deployment=gpt_deployment_name,
    azure_endpoint=aoai_endpoint,
    temperature=0,
)

embeddings_model = AzureOpenAIEmbeddings(
    api_key=aoai_api_key,
    azure_deployment=embed_deployment_name, 
    openai_api_version="2024-06-01",
    azure_endpoint=aoai_endpoint,
)

qdrant = Qdrant.from_documents(
    docs,
    embeddings_model,
    url=qdrant_url, 
    api_key=qdrant_api_key,
    collection_name="book",
    force_recreate=True,
)

retriever = qdrant.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "請回答依照 context 裡的資訊來回答問題:{context}。問題：{input}"),
    ("human", "{input}")
    ])

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "請問這本書的作者？"})

print(response["answer"])

print("=" * 80)

response = retrieval_chain.invoke({"input": "請問張維元認為這本書如何？"})

print(response["answer"])

print("=" * 80)

response = retrieval_chain.invoke({"input": "請問這本書的推薦人有誰"})

print(response["answer"])
```


5. 程式碼之 agent

```
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import BaseTool
from typing import Optional, Union
from dotenv import load_dotenv
import os
import random  

load_dotenv("./.env")

aoai_api_key=os.getenv("AOAI_API_KEY")
aoai_endpoint=os.getenv("AOAI_ENDPOINT")
gpt_deployment_name=os.getenv("AOAI_GPT_DEPLOYMENT_NAME")


desc = (
    "use this tool when you need to generate a random number. "
    "No specific parameters are needed."
)

class RandomNumberTool(BaseTool):
    name: str = "Random number generator"
    description: str = desc
    
    def _run(
        self,
        random_number
    ):
        random_number = random.randint(1, 350)
        return random_number


tools = [RandomNumberTool()]

prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

                                      
Question: {input}
Thought:{agent_scratchpad}
""")

model = AzureChatOpenAI(
    api_key=aoai_api_key,
    openai_api_version="2024-06-01",
    azure_deployment=gpt_deployment_name,
    azure_endpoint=aoai_endpoint,
    temperature=0,
)

zero_shot_agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=zero_shot_agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "請抽出兩個隨機數"})

print(response)

```

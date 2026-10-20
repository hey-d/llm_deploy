import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


load_dotenv()


API_Key = os.getenv("Groq_API_Key")


def inputs():
    print("enter the language that you want the text to be translated to")
    language = input()
    print("enter the text that you want to translate to the desired language")
    message = input()
    return ({"language": language, "message": message})

def outputs(xyz):
    return xyz.content


llm = ChatGroq(
    api_key=API_Key,
    model = "qwen/qwen3.8-27b",
)

messages = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a helpful assistant that translates English to {language}."
        },
    {
        "role": "user",
        "content": "{message}"
    }
])


chain = RunnableLambda(lambda x: inputs()) | messages | llm | outputs
res = chain.invoke({})

print(res)
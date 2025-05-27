# LangChain imports
from langchain_ollama import ChatOllama
#################################################################

# AGENT

prompt = [{"role": "system", "content": "I am an excelent linquist. The task is to laber location entities in the given sentence. Below are some examples"},
          {"role": "user", "content": "Input: Only France and Britain backed Fischler's proposal."},
          {"role": "assistant", "content": "Output: Only @@France## and @@Britain## backed Fischler's proposal."},
          {"role": "user", "content": "Germany imported 47,600 sheep from Britain last year, nearly half of total imports."},
          {"role": "assistant", "content": "Output: @@Germany## imported 47,600 sheep from @@Britain## last year, nearly half of total imports."}
          ]

# Select the model and INVOKE IT - be sure that Ollama is running
model_instance = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:27b",
    temperature=0)

response = model_instance.invoke(prompt.append({"role": "user", "content": "Input:China says Taiwan spoils atmosphere for talks."}))

print(response.content)

from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import os
import openai
 
os.environ['OPENAI_API_KEY'] = ""  #  insert your opeanai api key
openai.api_key = os.environ["OPENAI_API_KEY"]

# function to call OpenAI APIs
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# intent
intent_name_field = ResponseSchema(name="intent", description=f"Based on the latest user message, extract the user message intent. Here are some possible labels: 'greetings', 'booking', 'complaint' or 'other'")
# user need
user_need_field = ResponseSchema(name="user_need", description="Rephrase the latest user request and make it a meaningful question without missing any details. Use '' if it is not available")
# user sentiment
sentiment_field = ResponseSchema(name="sentiment", description="Based on the latest user message, extract the user sentiment. Here are some possible labels: 'positive', 'neutral', 'negative', 'mixed' or 'other'")
# number of pizzas to be ordered
n_pizzas_field = ResponseSchema(name="n_pizzas", description="Based on the user need, extract the number of pizzas to be made. Use '' if it is not available")

# schema with all entities (fields) to be extracted
conversation_metadata_output_schema_parser = StructuredOutputParser.from_response_schemas(
    [
        # user intent
        intent_name_field,
        # user need
        user_need_field,
        # user sentiment
        sentiment_field,
        # number of ordered pizzas 
        n_pizzas_field
        # other extra fields to be extracted
        # ... 
    ]
)

conversation_metadata_prompt_template_str = """
Given in input a full chat history between a user and a customer service assistant, \
extract the following metadata according to the format instructions below.
 
<< FORMATTING >>
{format_instructions}
 
<< INPUT >>
{chat_history}
 
<< OUTPUT (remember to include the ```json)>>"""
 
conversation_metadata_prompt_template = PromptTemplate.from_template(template=conversation_metadata_prompt_template_str)

# example of Greetings conversation
messages =  [
  {'role':'assistant', 'content':'Hello! I am Isi, your digital assistant. \n How may I help you today?'},  
  {'role':'user', 'content':'Hi! my name is Isa!!'}
]
 
# init prompt
conversation_metadata_recognition_prompt = (
    conversation_metadata_prompt_template.format(
        chat_history=messages,
        format_instructions=conversation_metadata_output_schema
    )
)

# call openAI API to detect the conversation metadata (e.g. intent, user_need, entities, etc.)
conversation_metadata_detected_str = get_completion(conversation_metadata_recognition_prompt)

# conversion from string to python dict
conversation_metadata_detected = conversation_metadata_output_schema_parser.parse(conversation_metadata_detected_str)
print(conversation_metadata_detected)
# {'intent': 'greetings',
# 'user_need': '',
# 'sentiment': 'positive',
# 'n_pizzas': ''}

# example of pizza order conversation
messages =  [
  {'role':'assistant', 'content':'Hello! I am Isi, your digital assistant. \n How may I help you today?'},  
  {'role':'user', 'content':'Hi, my name is Isa!'},
  {'role':'assistant', 'content': "Hi Isa! It's nice to meet you. Is there anything I can help you with today?"},
  {'role':'user', 'content':"Yes, I'd like to make an order. I'd like order 4 pizzas and 10 beers. Could you help me with that?"}
]
 
# init prompt
conversation_metadata_recognition_prompt = (
    conversation_metadata_prompt_template.format(
        chat_history=messages,
        format_instructions=conversation_metadata_output_schema
    )
)

# call openAI API to detect the conversation metadata (e.g. intent, user_need, entities, etc.)
conversation_metadata_detected_str = get_completion(conversation_metadata_recognition_prompt)

# conversion from string to python dict
conversation_metadata_detected = conversation_metadata_output_schema_parser.parse(conversation_metadata_detected_str)
print(conversation_metadata_detected)
# {'intent': 'booking',
# 'user_need': 'Could you help me make an order for 4 pizzas and 10 beers?',
# 'sentiment': 'positive',
# 'n_pizzas': '4'}
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

# chat gpt doesnt remember our conversation
# we should send our entire message history

load_dotenv()

chat = ChatOpenAI(
    # verbose=True
)

# return messages true means put chats from entire objects instead of only chat strings
# memory = ConversationBufferMemory(
#     # utilizing messages.json as a chat history
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages",
#     return_messages=True,
# )
# chat app with file chat memory makes the costs more expensive because we send the entire history

# [
#   {
#     "type": "human",
#     "data": {
#       "content": "what is 1+1?",
#       "additional_kwargs": {},
#       "type": "human",
#       "example": false
#     }
#   },
#   {
#     "type": "ai",
#     "data": {
#       "content": "1 + 1 = 2",
#       "additional_kwargs": {},
#       "type": "ai",
#       "example": false
#     }
#   },
#   {
#     "type": "human",
#     "data": {
#       "content": "add 5",
#       "additional_kwargs": {},
#       "type": "human",
#       "example": false
#     }
#   },
#   ...
# ]

# using summary instead of history
# but when we exit the chat, the history is gone
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        # incorporate the conversation history (messages) stored in the memory into its response.
        MessagesPlaceholder(variable_name="messages"),
        # include the user's current input (content) in its response.
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    # verbose=True,
)

while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'



# This sets up how the chatbot will ask and answer questions
def setup_chatbot_conversation():
    conversation_template = """
    Here's some information to help answer the question:
    Context: {context}
    Question: {question}

    Please give a helpful answer below.
    Helpful answer:
    """
    prompt = PromptTemplate(template=conversation_template,
                            input_variables=['context', 'question'])
    return prompt

# This function helps the chatbot find the right answer
def answer_finder(chatbot, conversation, memory_db):
    qa_chain = RetrievalQA.from_chain_type(llm=chatbot,
                                           chain_type='stuff',
                                           retriever=memory_db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': conversation})
    return qa_chain

# This function turns on the chatbot
def activate_chatbot():
    chatbot = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return chatbot

# Main function to create the chatbot
def create_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    memory_db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    chatbot = activate_chatbot()
    conversation_setup = setup_chatbot_conversation()
    qa = answer_finder(chatbot, conversation_setup, memory_db)

    return qa

# Function to get the chatbot's response
def get_chatbot_response(question):
    chatbot = create_chatbot()
    response = chatbot({'query': question})
    return response

# Setting up the chatbot interaction
@cl.on_chat_start
async def start():
    chatbot = create_chatbot()
    msg = cl.Message(content="Starting the chatbot...")
    await msg.send()
    msg.content = "Hello! How can I help you today?"
    await msg.update()

    cl.user_session.set("chatbot", chatbot)

# Main interaction with the chatbot
@cl.on_message
async def main(message: cl.Message):
    chatbot = cl.user_session.get("chatbot") 
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    result = await chatbot.acall(message.content, callbacks=[callback_handler])
    answer = result["result"]
    sources = result["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
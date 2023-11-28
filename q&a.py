#### Installing required module
# API Token key
import os 
os.environ["OPENAI_API_KEY"] = "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"


# Importing the modules.
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever


#### Preparing the documents
# Loading documents
pdf_folder_path = './docs/'
os.listdir(pdf_folder_path)
# Reading the pdf files.
document=[]
for file in os.listdir(pdf_folder_path):
    if file.endswith(".pdf"):
        pdf_path=os.path.join(pdf_folder_path, file)
        loader=PyPDFLoader(pdf_path)
        document.extend(loader.load())


#### Splitting the documents into chunks
# Using character text splitter to split the documents into chunks
# I have adjusted the chunk_size from 3000 to 1000 and chunk_overlap from 100 to 50
document_splitter=CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=50)
document_chunks=document_splitter.split_documents(document)
# Checking the number of documents chunks generated
len(document_chunks)


#### Embeddings
# select which embeddings we want to use
embeddings = OpenAIEmbeddings()


#### Creating a vector store
# create the vectorestore to use as the index
db = Chroma.from_documents(document_chunks, embeddings, persist_directory='./database')
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})


#### Loading the OpenAI model
from transformers import AutoTokenizer, AutoModelForCausalLM
model = "openai-gpt"
tokenizer = AutoTokenizer.from_pretrained(model)

#### Creating a Pipeline
# from transformers import pipeline
# pipeline = pipeline(
#     "text-generation", #task
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     max_length=1000,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id
# )

# from langchain.chains import RetrievalQA
# qa = RetrievalQA.from_llm(
#     llm = pipeline,
#     retriever = retriever,
#     return_source_documents=True
# )
#### Memory to hold a conversation
# memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#### Conversational Retrieval QA chain
# create a chain to answer questions 
qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
chat_history = []
query = "what is Data Protection?"
result = qa({"question": query, "chat_history": chat_history})
chat_history = []
query = "Provide advice or guidance on how to comply with the data protection act in different scenarios or domains in Kenya."
result = qa({"question": query, "chat_history": chat_history})

result["answer"]
#### Saving the model
# To do.

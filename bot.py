from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_recursive_character_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def retrieve_answer(docs, embeddings, llm, query):
    splits = split_recursive_character_text(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain.invoke({"input": query})


if __name__ == "__main__":
    file_path = "./Pakistan.pdf"
    docs = load_pdf(file_path)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBYrlv8NiKila8i_Wk-ucEYTuO8RNMI2lQ"

    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # query = "What are the major festivals celebrated in Pakistan?"
    # results = retrieve_answer(docs, embeddings, llm, query)

    while True:
        print("Enter 1 to ask a question and 2 to quit")
        user_input = input("Please enter your choice: ")
        if user_input == '1':
            query = input("Please enter your question: ")
            results = retrieve_answer(docs, embeddings, llm, query)
            answer = results['answer']
            print(f"Answer: {answer}")
        elif user_input == '2':
            # Handle quitting the program
            print("Quitting the program. Goodbye!")
            break
        else:
        
            print("Invalid input. Please enter 1 or 2.")

    
    # print(results['answer'])

    
import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file
load_dotenv()

class ChatBot():
    def __init__(self):
        # Initialize document loader and splitter
        self.loader = TextLoader('./data.txt')
        self.documents = self.loader.load()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = self.text_splitter.split_documents(self.documents)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Retrieve Pinecone API key from environment variable
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            raise ValueError("Pinecone API key is not set. Please set it in the environment variables.")

        # Initialize Pinecone
        self.pinecone = Pinecone(api_key=pinecone_api_key)
        
        self.index_name = "langchain-demo"
        
        if self.index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=self.index_name, 
                dimension=768, 
                metric='cosine',
                spec=ServerlessSpec(cloud='gcp', region='us-west1')
            )
        self.docsearch = LangChainPinecone.from_existing_index(self.index_name, self.embeddings)
        
        # Initialize GPT-4 using OpenAI API
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = openai_api_key
        
        # Define the prompt template
        template = """
        You are a symptom tracking chatbot. Using the information provided, converse with the user.
        Do not talk with the user about anything unrelated to mental health. As you converse with them, collect symptoms, and once you have enough, return a list for a doctor to use in a diagnosis.
        If the user at any time says something along the lines of "Thank you, I'm done", then end early and display the list of symptoms and a potential mental health issue they might have.
        Past messages: {pasts}
        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question", "pasts"])

    def generate_response(self, context, question, pasts):
        prompt_text = self.prompt.format(context=context, question=question, pasts=pasts)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.8,
            top_p=0.8,
        )
        return response['choices'][0]['message']['content'].strip()

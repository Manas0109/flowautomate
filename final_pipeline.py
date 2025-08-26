import os
import uuid
import logging
from base64 import b64decode
from unstructured.partition.pdf import partition_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRAG:
    def __init__(self, google_api_key):
        """Initialize the multimodal RAG system."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=google_api_key
        )
        
        # Initialize embeddings and vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = Chroma(
            collection_name="multi_modal_rag", 
            embedding_function=self.embeddings
        )
        
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key="doc_id"
        )
        
        # Create summary chain
        self._setup_summary_chain()
        self.rag_chain = None
    
    def _setup_summary_chain(self):
        """Setup the summarization chain."""
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.
        
        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        
        Table or text chunk: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        self.summarize_chain = {"element": lambda x: x} | prompt | self.llm | StrOutputParser()
    
    def process_pdf(self, file_path):
        """Extract and process PDF content into tables, text, and images."""
        logger.info("Processing PDF...")
        
        # Partition PDF
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000
        )
        
        # Separate elements
        tables, texts, images = self._separate_elements(chunks)
        
        # Generate summaries
        logger.info("Generating summaries...")
        text_summaries = self.summarize_chain.batch(texts, {"max_concurrency": 3}) if texts else []
        
        tables_html = [table.metadata.text_as_html for table in tables]
        table_summaries = self.summarize_chain.batch(tables_html, {"max_concurrency": 3}) if tables else []
        
        image_summaries = self._summarize_images(images) if images else []
        
        # Store in vector database
        self._store_summaries(texts, text_summaries, tables, table_summaries, images, image_summaries)
        
        # Setup RAG chain after processing
        self.setup_rag_chain()
        
        return len(texts), len(tables), len(images)
    
    def _separate_elements(self, chunks):
        """Separate chunks into tables, texts, and images."""
        tables = []
        texts = []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
        
        # Extract images from CompositeElement objects
        images = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images.append(el.metadata.image_base64)
        
        return tables, texts, images
    
    def _summarize_images(self, images):
        """Generate summaries for images using multimodal capabilities."""
        prompt_template = """Describe the image in detail. Be specific about any graphs, 
                          charts, diagrams, or visual elements present in the image."""
        
        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.batch(images)
    
    def _store_summaries(self, texts, text_summaries, tables, table_summaries, images, image_summaries):
        """Store summaries and link to original data in vector store."""
        # Add texts
        if texts:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) 
                for i, summary in enumerate(text_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, texts)))
        
        # Add tables
        if tables:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={"doc_id": table_ids[i]}) 
                for i, summary in enumerate(table_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, tables)))
        
        # Add images
        if images:
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={"doc_id": img_ids[i]}) 
                for i, summary in enumerate(image_summaries)
            ]
            self.retriever.vectorstore.add_documents(summary_img)
            self.retriever.docstore.mset(list(zip(img_ids, images)))
    
    def _parse_docs(self, docs):
        """Split base64-encoded images and texts."""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except:
                text.append(doc)
        return {"images": b64, "texts": text}
    
    def _build_prompt(self, kwargs):
        """Build prompt with context including images."""
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]
        
        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text
        
        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Question: {user_question}
        """
        
        prompt_content = [{"type": "text", "text": prompt_template}]
        
        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                })
        
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])
    
    def setup_rag_chain(self):
        """Setup the RAG chain for question answering."""
        self.rag_chain = (
            {
                "context": self.retriever | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt)
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question):
        """Query the RAG system."""
        if self.rag_chain is None:
            raise ValueError("No documents have been ingested yet. Please upload a PDF first.")
        return self.rag_chain.invoke(question)


def main():
    # Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PDF_PATH = "D:/Development/work/flowautomate/embedded-images-tables.pdf"  # Replace with your PDF path
    
    # Initialize RAG system
    rag = MultimodalRAG(GOOGLE_API_KEY)
    
    # Process PDF
    try:
        num_texts, num_tables, num_images = rag.process_pdf(PDF_PATH)
        print(f"Processed: {num_texts} text chunks, {num_tables} tables, {num_images} images")
        
        # Setup RAG chain
        rag.setup_rag_chain()
        
        # Example queries
        questions = [
            "What are the main findings in the document?",
            "Can you summarize any tables present?",
            "What do the images show?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = rag.query(question)
            print(f"Answer: {response}")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":
    main()
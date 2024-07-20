from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.together import TogetherEmbedding

from llama_index.core import Settings
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core import PromptTemplate
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from PyPDF2 import PdfReader
import dotenv
import os
import os
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
# from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

dotenv.load_dotenv()

Settings.llm = OpenAILike(
    model="meta-llama/Llama-3-70b-chat-hf",
    api_base="https://api.together.xyz/v1",
    api_key=os.environ.get("TOGETHER_API_KEY"),
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.1,
)
Settings.embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval", 
    api_key=os.environ.get("TOGETHER_API_KEY")
)
question_template = """\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer. \

Please provide only the questions. \
Examples of questions: \
- What is the main idea of the text? \
- What is the author's purpose in writing the text?
"""

def get_questions(file):
    file = PdfReader(file)
    text = ""
    for page in file.pages:
        text += page.extract_text()

    splitter = SentenceSplitter(chunk_size=2400, chunk_overlap=500)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=768, chunk_overlap=128
    )

    question_generator = QuestionsAnsweredExtractor(
        questions=1, metadata_mode=MetadataMode.EMBED, prompt_template=question_template)
    question_gen_pipeline = IngestionPipeline(transformations=[text_splitter, TitleExtractor(nodes=3), question_generator])
    questions = question_gen_pipeline.run(nodes=nodes)
    
    return questions
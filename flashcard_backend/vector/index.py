from llama_index.core import Document, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

def create_index_from_text(
    text: str, 
    service_context: ServiceContext, 
    dimension:int=768, 
    chunk_size: int=800, 
    chunk_overlap: int=450
) -> VectorStoreIndex:
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents([Document(text=text)])

    
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes,
        service_context=service_context,
        storage_context=storage_context
    )

    return index

def save_index(index: VectorStoreIndex, path: str):
    index.storage_context.persist(path)
    print(f"Index saved to {path}")

def load_index(path: str, service_context: ServiceContext) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(path),
        vector_store=FaissVectorStore.from_persist_dir(path),
        index_store=SimpleIndexStore.from_persist_dir(path)
    )

    index = load_index_from_storage(storage_context, service_context=service_context)
    return index
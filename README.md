# llama3-backend

## Installation
The following packages are required to be installed before installing the `llama3-backend` package.
```bash
#     "llama-index",
#     "llama-index-embeddings-together",
#     "llama-index-llms-openai-like",
#     "llama-index-vector-stores-faiss"
#     "pymilvus"
#     "llama-index-vector-stores-milvus"
```
```bash
pip install git+https://github.com/Alpaca-8b-Llama3Hackathon/llama3-backend.git
```
## Usage (Question Answering pairs)
```python
from flashcard_backend.question import get_questions
import dotenv
dotenv.load_dotenv()
path = "/home/monsh/works/Visual_on-line_learning_in_distributed_camera_netw.pdf"
questions_and_answers = get_questions(path, api_key=os.getenv("TOGETHER_API_KEY"))

for question, answer in questions_and_answers:
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
```


## Usage (Deprecated)
```python
from flashcard_backend.model import get_model, get_embedding
from flashcard_backend.vector.index import create_index_from_text, save_index, load_index
from llama_index.core import ServiceContext

import os
import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    path = "/home/monsh/works/Visual_on-line_learning_in_distributed_camera_netw.pdf"
    text_from_pdf = pdf_to_text(path)

    llm = get_model(api_key=os.getenv("TOGETHER_API_KEY"))
    embedding = get_embedding(api_key=os.getenv("TOGETHER_API_KEY"))
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding)
    index = create_index_from_text(text=text_from_pdf, service_context=service_context)

    save_index(index, "./index")
    # index = load_index("./index", service_context)

    # retriever = index.as_retriever()
    # retrieved_text = retriever.retrieve("What is the purpose of the paper?")

    # prompt = """
    # '''
    #     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    #     provided context just say, "answer is not available in the context", remind to not provide the wrong answer

    #     Context: {context}
    #     Question: {question}
    #     Answer:
    # '''
    # """.format(context=retreive_text, question="How to do multi-camera calibration?")
    # res = llm.complete(prompt)

    query_engine = index.as_query_engine()
    query = "What is the purpose of the paper?"
    response = query_engine.query(query)
    print(response.response)
```
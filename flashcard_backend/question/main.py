from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter
from PyPDF2 import PdfReader
import re
from llama_index.core import Document
from typing import List, Tuple
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline

question_template = """\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer. \

Please Reply with only the questions and answer. \
    Examples of questions and answer: \
- Question 1 : What is the main idea of the text? \
    Answer 1 : Answer your Question \
- Question 2 : What is the author's purpose in writing the text? \
    Answer 2 : Answer your Question \
"""

def extract_qa_pairs(qa_string):
    # Regular expression pattern to match questions and answers
    pattern = r'\*\*Question (\d+):\*\* (.*?)\s*\*\*Answer \1:\*\* (.*?)(?=\*\*|$)'
    
    # Find all matches in the qa_string
    matches = re.findall(pattern, qa_string, re.DOTALL)
    
    # Create a list of dictionaries containing question-answer pairs
    qa_pairs = [{"question": q.strip(), "answer": a.strip()} for _, q, a in matches]
        
    return qa_pairs

def get_questions(file: str, api_key: str, question_template: str = question_template) -> List[Tuple[str, str]]:
    assert "{context_str}" in question_template, "Prompt template must contain {context_str} placeholder"
    assert "{num_questions}" in question_template, "Prompt template must contain {num_questions} placeholder"

    file = PdfReader(file)
    text = ""
    for page in file.pages:
        text += page.extract_text()

    llm = OpenAILike(
        model="meta-llama/Llama-3-70b-chat-hf",
        api_base="https://api.together.xyz/v1",
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0.1,
    )
    splitter = SentenceSplitter(chunk_size=2400, chunk_overlap=500)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    text_splitter = TokenTextSplitter(
        separator=" ", chunk_size=768, chunk_overlap=128
    )

    question_generator = QuestionsAnsweredExtractor(
        questions=1, metadata_mode=MetadataMode.EMBED, prompt_template=question_template, llm=llm)
    question_gen_pipeline = IngestionPipeline(transformations=[text_splitter, question_generator])
    questions = question_gen_pipeline.run(nodes=nodes)
    
    results = []
    for question in questions:
        qa = str(question.to_dict()['metadata']['questions_this_excerpt_can_answer'])
        
        # Extract question-answer pairs
        qa_pairs = extract_qa_pairs(qa)
        
        for i, pair in enumerate(qa_pairs, 1):
            results.append((pair['question'], pair['answer']))

    return results        
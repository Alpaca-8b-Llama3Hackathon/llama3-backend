from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.together import TogetherEmbedding

def get_model(
    api_key: str, 
    model_name: str="meta-llama/Llama-3-70b-chat-hf", 
    is_chat_model: bool=True, 
    is_function_calling_model: bool=True,
    temperature: float=0.1
) -> OpenAILike:
    return OpenAILike(
        model=model_name,
        api_base="https://api.together.xyz/v1",
        api_key=api_key,
        is_chat_model=is_chat_model,
        is_function_calling_model=is_function_calling_model,
        temperature=temperature,
    )

def get_embedding(api_key: str, model_name: str="togethercomputer/m2-bert-80M-8k-retrieval") -> TogetherEmbedding:
    return TogetherEmbedding(model_name=model_name, api_key=api_key)

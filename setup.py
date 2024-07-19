from setuptools import setup

setup(
    name='flashcard_backend',
    version='0.1',
    packages=["flashcard_backend"],
    install_requires=[
        "llama-index",
        "llama-index-embeddings-together",
        "llama-index-llms-openai-like",
        "llama-index-vector-stores-faiss"
    ]
)

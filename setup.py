from setuptools import setup, find_packages

setup(
    name='flashcard_backend',
    version='0.8',
    packages=find_packages(where='flashcard_backend'),
    package_dir={"": 'flashcard_backend'},
    url="https://github.com/Alpaca-8b-Llama3Hackathon/llama3-backend"
    
    # package_dir={'flashcard_backend': 'flashcard_backend'},
    # package_data={'flashcard_backend': ['*.py']},
    # install_requires=[
    #     "llama-index",
    #     "llama-index-embeddings-together",
    #     "llama-index-llms-openai-like",
    #     "llama-index-vector-stores-faiss"
    # ]
)

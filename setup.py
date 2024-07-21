from setuptools import setup
import os
from pathlib import Path
root = os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir))
os.chdir(root)

root = Path(__file__).parent
os.chdir(str(root))

setup(
    name='flashcard_backend',
    version='1.2',
    packages=['flashcard_backend']
    # package_dir={"": 'flashcard_backend'},
    
    # package_dir={'flashcard_backend': 'flashcard_backend'},
    # package_data={'flashcard_backend': ['*.py']},
    # install_requires=[
    #     "llama-index",
    #     "llama-index-embeddings-together",
    #     "llama-index-llms-openai-like",
    #     "llama-index-vector-stores-faiss"
    # ]
)

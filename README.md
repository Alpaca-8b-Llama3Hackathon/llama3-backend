# llama3-backend-lite

## Installation
The following packages are required to be installed before installing the `llama3-backend` package.
```bash
#     "llama-index",
#     "llama-index-embeddings-together",
#     "llama-index-llms-openai-like",
```
```bash
pip install git+https://github.com/Alpaca-8b-Llama3Hackathon/llama3-backend.git@lite
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

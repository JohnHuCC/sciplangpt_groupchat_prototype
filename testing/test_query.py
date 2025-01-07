import dotenv

dotenv.load_dotenv()

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the module
from utils.query_knowledge import query_knowledge
from utils.form_prompt_with_context import form_prompt_with_context

def main():
    query_result = query_knowledge("Intestine-on-a-chip")
    # print(query_result)
    # for res in query_result:
    #     print(res)
    #     print('\n')
    prompt = form_prompt_with_context("testing", query_result)
    print(prompt)

if __name__ == "__main__":
    main()
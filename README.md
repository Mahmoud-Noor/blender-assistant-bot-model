# Blender Assistant Bot (BAB) 🚀

The Blender Assistant Bot (BAB) is an advanced AI-driven chatbot designed to enhance the learning and usage experience for users of Blender, a widely-used open-source 3D creation software. BAB leverages state-of-the-art natural language processing (NLP) and Retrieval-Augmented Generation (RAG) technologies to provide real-time, contextually relevant support to users, ranging from beginners to experts.

## Features 🎨
- **Real-Time Support**: Get instant answers to your Blender queries.
- **Context-Aware Assistance**: Tailored responses based on your ongoing projects.
- **Seamless Integration**: Connects with Blender’s official documentation and community resources.

## Getting Started 💻
1. **Clone the Repository**:
    ```
    git clone https://github.com/yourusername/blender-assistant-bot.git
    ```
2. **Set Up the Environment**:
    - Install necessary dependencies:
    ```
    pip install -r requirements.txt
    ```
3. **Run the Bot**:
    - Launch the bot with:
    ```
    python run_bot.py
    ```

## How to Use 🎓
### Step 1: Launch the Bot
Once the bot is running, you'll be greeted with a user interface where you can input your queries related to Blender.

### Step 2: Ask a Question
Type your question into the input box. For example, you could ask:
- "How do I add a texture to a 3D model?"
- "What are the shortcuts for object manipulation?"

### Step 3: Receive an Answer
BAB will process your question using advanced NLP and provide you with an answer, along with links to relevant Blender documentation or tutorials.

### Step 4: Explore Further
Use the provided links or ask follow-up questions to delve deeper into Blender's features. BAB is designed to support a continuous learning experience.

## Architecture 🏗️
The system is designed with a client-server architecture:
- **Frontend**: HTML, CSS, JavaScript for user interaction.
- **Backend**: Python, Flask for server management.
- **NLP Engine**: Powered by OpenAI’s GPT models.

## Contributing 🤝
We welcome contributions from the community! Feel free to submit pull requests or open issues.

## License 📄
This project is licensed under the MIT License.

## Contact 📬
For any queries, feel free to reach out at [your email].

## Tutorial: Using the Blender Assistant Bot (BAB) Application on Google Colab 📚

### Section 1: Setting Up Your Colab Environment

**Objective:** Install the required packages and set up the environment on Google Colab.

1. **Install Required Libraries:**
   - Open a new Colab notebook and run the following code cells to install the necessary libraries:
     ```python
     !pip -q install langchain_text_splitters
     !pip -q install langchain_community
     !pip -q install tiktoken
     !pip -q install pinecone-client[grpc]
     !pip -q install openai
     !pip -q install langchain_pinecone
     !pip -q install langchain_openai
     !pip -q install langchain
     !pip -q install together
     !pip -q install langchain_together
     !pip -q install gradio
     !pip -q install langgraph
     !pip install jinja2
     !pip install flask-ngrok
     !pip install uvicorn
     !pip install fastapi
     !pip install pyngrok
     ```

2. **Import Required Libraries:**
   - Create a new code cell and run the following to import the necessary libraries:
     ```python
     import os
     import re
     import openai
     import pinecone
     import numpy as np
     from bs4 import BeautifulSoup
     from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
     from langchain.document_loaders import DirectoryLoader
     from langchain.vectorstores import Pinecone
     from langchain.llms import OpenAI
     from langchain.chains.question_answering import load_qa_chain
     from langchain.chains import RetrievalQA
     from langchain_openai import ChatOpenAI, OpenAIEmbeddings
     from langchain_pinecone import PineconeVectorStore
     from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
     from langchain.llms import HuggingFacePipeline
     from langchain.prompts import PromptTemplate
     from transformers import pipeline
     from langchain_core.output_parsers import StrOutputParser
     from operator import itemgetter
     from langchain_community.llms import Together
     from langchain_core.runnables import RunnablePassthrough
     from langchain_core.messages import HumanMessage, AIMessage
     from langchain_core.prompts import MessagesPlaceholder
     from langchain.memory import ChatMessageHistory
     ```

### Section 2: Configure API Keys and Create Core Components

**Objective:** Set up API keys and create essential components such as embeddings, vectorstore, and language models.

1. **Set Up API Keys:**
   - Insert your API keys for OpenAI, Pinecone, and TogetherAI into the environment variables. Create a new code cell and run the following:
     ```python
     # OpenAI API Key
     os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
     
     # Pinecone API Key
     os.environ['PINECONE_API_KEY'] = 'your-pinecone-api-key'
     
     # TogetherAI API Key
     os.environ['TOGETHER_API_KEY'] = 'your-together-api-key'
     ```

2. **Create Embeddings:**
   - Initialize the embeddings using OpenAI. Run this in a new code cell:
     ```python
     embeddings = OpenAIEmbeddings()
     ```

3. **Create Vectorstore:**
   - Initialize the vectorstore with Pinecone. Add this to a new code cell:
     ```python
     vectorstore = PineconeVectorStore.from_existing_index(index_name="bab", embedding=embeddings)
     ```

4. **Set Up TogetherAI Models:**
   - Initialize the Llama-3.1 and Llama Guard models from TogetherAI. Run the following code in a new cell:
     ```python
     response = Together(
         model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
         max_tokens=1024,
         temperature=0.2,
         repetition_penalty=1.1,
         together_api_key=os.environ.get('TOGETHER_API_KEY')
     )
     
     guard = Together(
         model="meta-llama/LlamaGuard-2-8b",
         together_api_key=os.environ.get('TOGETHER_API_KEY')
     )
     ```

### Section 3: Create and Test the Retriever

**Objective:** Set up and test the retriever to handle queries and perform searches.

1. **Create Retriever Object:**
   - Configure the retriever for similarity search. Run this code in a new cell:
     ```python
     retriever = vectorstore.as_retriever(
         search_type="similarity",
         search_kwargs={'k': 10}
     )
     ```

2. **Test Llama Guard:**
   - Set up and test the safety checks with Llama Guard. Use this code in a new cell:
     ```python
     guard_temp_ans = """
     Task: Check if there is unsafe content in '{{ role }}' messages in conversations according to our safety policy with the below categories.

     <BEGIN UNSAFE CONTENT CATEGORIES>
     [Detailed categories here]
     <END UNSAFE CONTENT CATEGORIES>

     <BEGIN CONVERSATION>
     {answer}
     <END CONVERSATION>

     Provide your safety assessment for {answer} in the above conversation:
     - First line must read 'safe' or 'unsafe'.
     - If unsafe, do not include a second line.
     """
     
     prompt_ans_guard = ChatPromptTemplate.from_template(guard_temp_ans)
     
     guard_chain = (
       prompt_ans_guard
       | guard
     )
     
     # Llama Guard Test
     # guard_chain.invoke({"answer": "hello, what is your name?"})
     ```

---

Happy Blending! 🎉

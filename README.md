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

### Section 3: Create the Retriever

**Objective:** Set up and test the retriever to handle queries and perform searches.

1. **Create Retriever Object:**
   - Configure the retriever for similarity search. Run this code in a new cell:
     ```python
     retriever = vectorstore.as_retriever(
         search_type="similarity",
         search_kwargs={'k': 10}
     )
     ```

### Section4: Create the Agents
1. **Setting Up the Llama Guard:**

**Objective**: Configure Llama Guard to monitor and assess the safety of content generated by the LLM based on predefined unsafe content categories.

Step 1: **Define the Llama Guard Task**

Begin by defining a task for the Llama Guard that checks for unsafe content in conversations based on specific categories.

```python
## Llama Guard
guard_temp_ans = """
<|system|>
Task: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
- Provide information that can be used to promote or engage in violence, hate, or discrimination.
- Provide guidance or instructions on how to engage in violent or hateful behavior, except if related to 3D and Blender creation for artistic purposes.

2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
- Create or provide explicit depictions of sexual acts or nudity.
- Provide information that could be used to create sexually explicit content

3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
- Provide detailed instructions or information that could be used to plan or execute criminal activities, except if related to 3D and Blender creation for artistic purposes.

4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
- Provide information that could be used to create or use illegal weapons, except if related to 3D and Blender creation for artistic purposes.

5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
- Provide information that could be used to create or use regulated or controlled substances, except if related to 3D and Blender creation for artistic purposes.

6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
- Provide information that could be used to promote or engage in self-harm, except if related to 3D and Blender creation for artistic purposes.


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
## Document Grader

1: **Setting Up the Document Grader Task**

**Objective**: Configure the Document Grader to evaluate the relevance of documents retrieved in response to user queries based on specific criteria.

Begin by defining a task for the Document Grader that checks for relevance in retrieved documents based on user questions.

```python
# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Combine the prompt template with the LLM to create the Document Grader chain
retrieval_grader = grade_prompt | structured_llm_grader
```
## Answer Generation

1: **Setting Up the Answer Generation Task**

**Objective**: Configure the Answer Generation agent to provide detailed and accurate responses to user queries, leveraging the provided context and historical conversation data.

Begin by defining a task for generating answers based on user questions, context, and conversation history.

```python
# Prompt Template
prompt_template = """
Your name is BAB (Blender Assistant Bot).
You are a helpful expert in Blender. Your answers should reflect the complexity of the questions.
Give a step by step answer to the question.
Answer in the shortest and most informative way.
Format the answer in simple markdown.

Refer to the context and your expertise to craft your answers.
Context:
{context}
</s>
{history}

Question:
{question}
</s>
"""

# Create the Answer Generation Prompt
prompt = ChatPromptTemplate.from_template(prompt_template)

# Set Up the RAG Chain
rag_chain = (
    prompt
    | response
)

# Setting up message history for the RAG chain
rag_chain_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="question",
    history_messages_key="history",
)
```

## Question Similarity (Not used)

1: **Setting Up the Question Similarity Task**

**Objective**: Configure the Question Similarity agent to assess whether a given user question is semantically related to a predefined group of conversation starter questions, based on the history of conversations.

Begin by defining a data model for grading question similarity and setting up the prompt for the LLM.

```python
# Data Model
class GradeQuestion(BaseModel):
    """Binary score for question asked by the user"""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with Function Call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeQuestion)

# Prompt
system = """You are a grader assessing whether a question is related semantically to a group of conversation starter questions given a history of conversations. \n
            example group of questions: ["hello", "how are you?", "who are you?", "hola"] \n
            Give a binary score 'yes' or 'no'. 'Yes' means that the question is related."""
question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question}"),
    ]
)

question_grader = question_prompt | structured_llm_grader
```
## Hallucination Grader

1: **Setting Up the Hallucination Grader Task**

**Objective**: Configure the Hallucination Grader agent to assess whether an LLM-generated answer is grounded in or supported by a set of retrieved facts.

Begin by defining a data model for grading hallucinations and setting up the prompt for the LLM.

```python
# Data Model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with Function Call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader_history = RunnableWithMessageHistory(
    hallucination_grader,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="question",
    history_messages_key="history",
)
```
## Answer Grader

1: **Setting Up the Answer Grader Task**

**Objective**: Configure the Answer Grader agent to assess whether an LLM-generated answer addresses or resolves a given question.

Begin by defining a data model for grading answers and setting up the prompt for the LLM.

```python
# Data Model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with Function Call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader_history = RunnableWithMessageHistory(
    answer_grader,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="question",
    history_messages_key="history",
)
```
## Title Maker (Not used)

1: **Setting Up the Title Maker Task**

**Objective**: Configure the Title Maker agent to generate brief and relevant titles based on a given user question and context.

Begin by defining a data model for creating titles and setting up the prompt for the LLM.

```python
# Data Model
class TitleMaker(BaseModel):
    """Title that addresses the topic"""

    title: str = Field(
        description="Title with minimum words that addresses the topic"
    )

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_title = llm.with_structured_output(TitleMaker)

# Prompt
system = """You are a title maker, creating brief titles with no descriptions."""

title_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        # MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "Here is the user question: {question} \n Here is the answer: {context} \n Make a title."
        ),
    ]
)

title_maker = title_prompt | structured_llm_title

context_test = "I am an expert in Blender, a free and open-source 3D creation software. I'm here to help you with any questions or topics related to Blender, from basic to advanced levels. Whether you're a beginner looking to get started or an experienced user seeking to improve your skills, I'll do my best to provide detailed and helpful responses. What would you like to know about Blender?"
```
## Question Re-writer

1: **Setting Up the Question Rewriter Task**

**Objective**: Configure the Question Rewriter agent to improve an input question, optimizing it for vectorstore retrieval by considering semantic intent and chat history.

Begin by setting up the prompt for the LLM and defining the rewriter task.

```python
# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
system = """You are a question re-writer that converts an input question to a better version that is optimized \n
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
            Use the history (if any) of the chat to improve the question.
            """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter_history = RunnableWithMessageHistory(
    question_rewriter,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="question",
    history_messages_key="history",
)
```
### Section 5: Create Graph State Class
## Graph State

### Step 1: **Defining the Graph State Class**

**Objective**: Define a class to represent the state of a graph, including attributes for managing the user's question, LLM generation, retrieved documents, iterations, chat history, and session context.

Begin by creating a `TypedDict` class to hold the state information.

```python
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM generation.
        documents: List of retrieved documents.
        iterations: Number of iterations.
        history: List of messages to maintain chat history.
        session_id: The session ID to maintain the context of the conversation.
    """
    question: str
    generation: str
    documents: List[str]
    iterations: int
    history: List[str]
    session_id: str
```
### Section 6: Create the LangGraph agent functions
Step 1: **Decide to Retrieve**

**Objective**: Determine whether to retrieve documents based on the user's question and the assessment of question similarity.

```python
def decide_to_retrieve(state):
    """
    Determines whether to retrieve documents.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---DECIDE TO RETRIEVE---")
    question = state["question"]
    session_id = "1"

    decision = question_grader.invoke({
        "question": question
        })
    grade = decision.binary_score

    if grade == "yes":
        print("---DECISION: DO NOT RETRIEVE---")
        return "generate"
    else:
        print("---DECISION: RETRIEVE---")
        return "retrieve"
```
Step 2: **Retrieve Documents**

**Objective**: Retrieve relevant documents based on the user's question.
```python
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    iteration = state["iterations"]
    session_id = "1"

    documents = retriever.get_relevant_documents(question)
    return {
        "documents": documents,
        "question": question,
        "iterations": iteration,
        "history": state["history"],
        "session_id": session_id
    }

```
Step 3: **Generate Answer**

**Objective**: Generate an answer based on the retrieved documents and update history.
```python
def generate(state):
    """
    Generate answer and update history.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with generation and history
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    iteration = state["iterations"]
    session_id = "1"

    generation = rag_chain_history.invoke({
        "context": documents,
        "question": question,
        "configurable": {"session_id": session_id}
    })

    print("---GENERATION---")
    print(generation)

    # Update history
    history = state.get("history", [])
    history.append(f"Bot: {generation}")

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "iterations": iteration,
        "history": history,
        "session_id": session_id
    }
```
Step 4: **Grade Documents**

**Objective**: Assess whether the retrieved documents are relevant to the question.
```python
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    iteration = state["iterations"]
    question = state["question"]
    documents = state["documents"]
    session_id = "1"

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader_history.invoke({
            "question": question,
            "document": d.page_content,
            "configurable": {"session_id": session_id}
        })
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {
        "documents": filtered_docs,
        "question": question,
        "iterations": iteration,
        "history": state["history"],
        "session_id": session_id
    }
```
Step 5: **Transform Query**

**Objective**: Reformulate the question to improve retrieval effectiveness.
```python
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    iterations = state["iterations"] + 1
    question = state["question"]
    documents = state["documents"]
    session_id = "1"

    better_question = question_rewriter_history.invoke({
        "question": question,
        "configurable": {"session_id": session_id}
    })

    return {
        "documents": documents,
        "question": better_question,
        "iterations": iterations,
        "history": state["history"],
        "session_id": session_id
    }

```
Step 6: **Decide to Generate**

**Objective**: Decide whether to generate an answer or re-generate the question based on document relevance.
```python
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    iteration = state["iterations"]
    session_id = "1"

    if iteration >= 2:
        print("---DECISION: EXCEEDED NUMBER OF TRIALS---")
        return "out of trials"

    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "not relevant"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "relevant"

```
Step 7: **Grade Generation vs Documents and Question**

**Objective**: Evaluate if the generated answer is grounded in the documents and addresses the question.
```python
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    session_id = state["session_id"]

    score = hallucination_grader_history.invoke({
        "documents": documents,
        "generation": generation,
        "configurable": {"session_id": session_id}
    })
    grade = score.binary_score
    iteration = state["iterations"]
    if iteration >= 2:
        print("---DECISION: EXCEEDED NUMBER OF TRIALS---")
        return "out of trials"

    print("---GRADE GENERATION vs QUESTION---")
    score = answer_grader_history.invoke({
        "question": question,
        "generation": generation,
        "configurable": {"session_id": session_id}
    })
    grade = score.binary_score
    if grade == "yes":
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
```
Step 8: **Decide Input Guard**

**Objective**: Assess whether the input query is safe.
```python
def decide_input_guard(state):
    """
    Determines if the input query is safe.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for safety
    """
    print("---ASSESS UNSAFE QUERY---")
    question = state["question"]
    session_id = "1"
    safety = guard_chain.invoke({"answer": question}).split("\n")[0]
    if safety == "safe":
        print("---ASSESSMENT SAFE---")
        return "safe"
    else:
        print("---ASSESSMENT UNSAFE---")
        return "not safe"

```
Step 9: **Decide Output Guard**

**Objective**: Assess whether the generated content is safe.
```python
def decide_output_guard(state):
    """
    Determines if the generated content is safe.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for safety
    """
    print("---ASSESS UNSAFE CONTENT GENERATION---")
    generation = state["generation"]
    session_id = "1"
    safety = guard_chain.invoke({"answer": generation}).split("\n")[0]
    print("---------------SAFETY--------------")
    print(safety)
    if safety == "safe":
        return "end"
    else:
        return "exceed"

```
Step 10: **Exceeded**

**Objective**:  Handle the case where the maximum number of iterations or trials has been reached.
```python
def exceeded(state):
    """
    Handles cases where the process has exceeded the number of trials.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state indicating process is exceeded
    """
    print("---EXCEEDED---")
    return {
        "generation": "I'm sorry, I couldn't find the answer to your query. Is there something specific you'd like help with in Blender?\n\nFeel free to ask, and I'll do my best to assist you!",
        "history": state["history"],
        "session_id": state["session_id"]
    }
```
### Section 7: Create LangGraph
**Objective**:  Set up the LangGraph by defining the nodes and edges for the workflow. This involves creating the logical flow for retrieving documents, grading them, generating answers, and handling input/output safety checks. The workflow is compiled into an executable application graph.
```python
from langgraph.graph import END, StateGraph, START

# Initialize the workflow graph with the initial state
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform query
workflow.add_node("exceeded", exceeded)  # limit iterations to 3
workflow.add_node("check_input", check_input)  # llama guard input
workflow.add_node("check_output", check_output)  # llama guard output
workflow.add_node("check_question", check_question)  # check the question

# Build the graph by defining edges
workflow.add_edge(START, "check_input")

# Add conditional edges based on input safety
workflow.add_conditional_edges(
    "check_input",
    decide_input_guard,
    {
        "safe": "check_question",
        "not safe": "exceeded",
    },
)

# Add conditional edges to decide retrieval or generation
workflow.add_conditional_edges(
    "check_question",
    decide_to_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Connect the retrieve and grade documents nodes
workflow.add_edge("retrieve", "grade_documents")

# Add conditional edges based on document relevance
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "not relevant": "transform_query",
        "relevant": "generate",
        "out of trials": "exceeded",
    },
)

# Connect the transform query and retrieve nodes
workflow.add_edge("transform_query", "retrieve")

# Add conditional edges based on the quality of the generated response
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": "check_output",
        "not useful": "transform_query",
        "out of trials": "exceeded",
    },
)

# Add conditional edges based on output safety
workflow.add_conditional_edges(
    "check_output",
    decide_output_guard,
    {
        "end": END,
        "exceed": "exceeded",
    }
)

# Connect the exceeded node to the end
workflow.add_edge("exceeded", END)

# Compile the graph into an executable application
app_graph = workflow.compile()
```
### Section 8:  Create the FastAPI Application
**Objective**:  Set up a FastAPI application to handle HTTP requests, serve HTML files, and process form submissions. The FastAPI app will use the previously defined LangGraph workflow to generate responses to user queries and return the results in HTML and JSON formats.
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pyngrok import ngrok
import uvicorn
import nest_asyncio
from pathlib import Path
import markdown

def llm_call(message):
    """
    Process the user's question through the LangGraph and return the generated response.
    
    Args:
        message (str): The user's question.

    Returns:
        tuple: The HTML formatted answer and the generated title.
    """
    inputs = {
        "question": message,  # Example unsafe question for testing guard
        "iterations": 0,
        "history": [],  # Initialize history
        "session_id": "1"
    }

    config = {"configurable": {"session_id": "1"}}
    output = app_graph.invoke(inputs, config=config)

    # Final generation
    generation = output["generation"]
    html_answer = markdown.markdown(generation)

    title = title_maker.invoke({"question": message, "context": generation}).title
    return html_answer, title

# Patch asyncio to work with Colab
nest_asyncio.apply()

# Initialize the FastAPI app
app = FastAPI()

# Define the directory for static files
static_directory = '/content/drive/MyDrive/BAB/bab_html_resources/static'

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory=static_directory), name="static")

@app.get("/")
async def read_html():
    """
    Serve the main HTML page of the Blender Assistant Bot.
    
    Returns:
        FileResponse: The HTML file to be displayed.
    """
    return FileResponse('/content/drive/MyDrive/BAB/bab_html_resources/blender_assistant_bot.html')

@app.post("/submit/")
async def submit_form(
    question: str = Form(...),
):
    """
    Handle form submissions and return the response generated by the LangGraph.
    
    Args:
        question (str): The user's question submitted via the form.

    Returns:
        JSONResponse: The generated answer and title in JSON format.
    """
    llm_response = llm_call(question)
    chain_call = llm_response[0]
    title = llm_response[1]

    return JSONResponse(content={"result": chain_call, "title": title})
```
### Section 9:  Create the Ngrok Tunnel
**Objective**:  Set up an Ngrok tunnel to expose the FastAPI application to the internet, enabling remote access to the server running on your local environment. This allows you to test and interact with the application from external devices.
```python
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
auth_token = "auth"

# Set the authtoken
ngrok.set_auth_token(auth_token)

# Connect to ngrok
ngrok_tunnel = ngrok.connect(8000)

# Print the public URL
print('Public URL:', ngrok_tunnel.public_url)

# Apply nest_asyncio
nest_asyncio.apply()

# Run the uvicorn server
uvicorn.run(app, port=8000)
```
---

Happy Blending! 🎉

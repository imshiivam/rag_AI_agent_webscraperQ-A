from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import json
import ssl

llm = Ollama(model="mistral")


@tool
def analyze_website(input_string: str) -> str:
    """Analyzes a website's content to answer a specific question. Input should be a JSON string with 'url' and 'question' keys."""
    try:
        # Try to parse the input as JSON
        if isinstance(input_string, str):
            try:
                input_data = json.loads(input_string)
            except json.JSONDecodeError:
                if "url" in input_string.lower() and "question" in input_string.lower():
                    parts = input_string.split(",")
                    url = None
                    question = None
                    for part in parts:
                        if "url" in part.lower():
                            url = part.split(":", 1)[1].strip().strip('"\'')
                        elif "question" in part.lower():
                            question = part.split(":", 1)[1].strip().strip('"\'')
                    if url and question:
                        input_data = {"url": url, "question": question}
                    else:
                        return "Error: Invalid URL or Que"
                else:
                    return "Error: Invalid URL or Que"
        else:
            input_data = input_string

        url = input_data.get("url")
        question = input_data.get("question")
        
        if not url or not question:
            return "Error: Both 'url' and 'question' are required."
        # Create SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        loader = WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={
                "verify": False,
            }
        )
        
        try:
            docs = loader.load()
        except Exception as ssl_error:
            return f"Error loading website: {str(ssl_error)}."
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        # Vector embeddings and storage
        vector_store = FAISS.from_documents(
            documents=all_splits, embedding=OllamaEmbeddings(model="mistral")
        )
        # Retrieval
        retriever = vector_store.as_retriever()
        retrieved_docs = retriever.invoke(question)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # Generate answer
        rag_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
        prompt = PromptTemplate.from_template(rag_template)

        messages = prompt.invoke({"question": question, "context": docs_content})
        response = llm.invoke(messages)
        return response
        
    except Exception as e:
        return f"Error analyzing website: {str(e)}"


# Streamlit UI
st.title("RAG Agent: Ask Questions About a Web Page")

website_url = st.text_input("Enter a website URL to analyze:")
question = st.text_input("Ask a question about the page:", "What is on this page?")

if st.button("Submit"):
    if not website_url or not question:
        st.error("Please provide both a website URL and a question.")
    else:
        # Clear previous flow steps
        st.session_state.flow_steps = []
        
        with st.spinner("Agent is thinking..."):
            tools = [analyze_website]
            template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

IMPORTANT: You must follow this exact format. Do not deviate from it or add extra explanations.

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"url": "website_url_here", "question": "question_here"}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

RULES:
- Always start with "Thought:" followed by your reasoning
- Then "Action:" with exactly one of the available tool names
- Then "Action Input:" with a JSON string containing url and question
- Do NOT add explanations, code blocks, or extra text
- Stick to the format exactly

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            prompt = PromptTemplate.from_template(template)

            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            agent_input = (
                f"Use the analyze_website tool to analyze the website at URL '{website_url}' "
                f"and answer this question: '{question}'. "
                f"Make sure to use the correct JSON format for the tool input."
            )
            try:
                response = agent_executor.invoke({"input": agent_input})
                st.markdown("###Answer:")
                st.write(response["output"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

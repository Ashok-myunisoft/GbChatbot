import json
import os
import logging
import traceback
import pandas as pd
# RunPodLLM via shared_resources — ChatOllama no longer used
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from shared_resources import ai_resources
from fastapi import Header
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
DOCUMENTS_DIR = "/app/data"
 
class Message(BaseModel):
    content: str
    context: str = ""
 
def spell_check(text: str) -> str:
    return text
 
def clean_response(text: str) -> str:
    text = text.strip()
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text
 
def format_as_points(text: str) -> str:
    return text
 
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Use centralized AI resources
llm = ai_resources.response_llm
 
def load_csv_as_document(file_path: str) -> List[Document]:
    """Load CSV file and convert to documents"""
    docs = []
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
       
        csv_content = f"Data from {os.path.basename(file_path)}:\n\n"
        csv_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
       
        for idx, row in df.iterrows():
            row_content = f"Record {idx + 1}:\n"
            for col in df.columns:
                row_content += f"- {col}: {row[col]}\n"
            row_content += "\n"
           
            doc = Document(
                page_content=row_content,
                metadata={"source": file_path, "row_index": idx}
            )
            docs.append(doc)
       
        summary_content = f"Dataset Summary for {os.path.basename(file_path)}:\n"
        summary_content += f"Total records: {len(df)}\n"
        summary_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        summary_content += "Sample data:\n"
        summary_content += df.head().to_string(index=False)
       
        summary_doc = Document(
            page_content=summary_content,
            metadata={"source": file_path, "type": "summary"}
        )
        docs.append(summary_doc)
       
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
       
    return docs
 
def load_and_split_documents(documents_dir: str) -> List[Document]:
    all_docs = []
    
    menu_docs_path = os.path.join(documents_dir, "menu.csv")
    if os.path.exists(menu_docs_path):
        logger.info(f"Loading file: {menu_docs_path}")
        csv_docs = load_csv_as_document(menu_docs_path)
        all_docs.extend(csv_docs)
        logger.info(f"✅ Loaded {len(csv_docs)} documents from CSV: {menu_docs_path}")
    else:
        logger.warning(f"⚠️ {menu_docs_path} not found.")

    return all_docs
 
# Always initialize embeddings regardless of document availability
embeddings = ai_resources.embeddings
all_docs = load_and_split_documents(DOCUMENTS_DIR)
text_chunks = None
retriever = None
 
if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"✅ Loaded {len(all_docs)} docs, split into {len(text_chunks)} chunks.")
   
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning("Using deprecated HuggingFaceEmbeddings.")
   
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    logger.info("✅ FAISS vectorstore and retriever initialized.")
else:
    logger.warning("⚠️ No documents loaded. RAG will not be available.")

# Role-based system prompts for menu bot
ROLE_SYSTEM_PROMPTS_MENU = {
    "developer": """You are a senior software architect and technical expert at GoodBooks Technologies ERP system, specializing in menu structures and navigation.

Your identity and style:
- You speak to a fellow developer/engineer who understands technical concepts, menu hierarchies, and system navigation
- Use technical terminology for menu structures, permissions, and navigation logic naturally
- Discuss menu implementation, access controls, user roles, and system integration points
- Provide technical depth with menu hierarchies, routing, and user interface concepts
- Mention code examples, menu configurations, and access rules when relevant
- Think like a senior developer explaining menu systems to a peer

Remember: You are the technical expert helping another technical person understand and implement menu systems.""",

    "implementation": """You are an experienced implementation consultant at GoodBooks Technologies ERP system, specializing in menu configuration and user access management.

Your identity and style:
- You speak to an implementation team member who guides clients through menu setup and user training
- Provide step-by-step menu configuration and permission setup instructions
- Focus on practical "how-to" guidance for menu rollouts, user training, and access management
- Include best practices for menu organization, security, and user experience
- Explain as if preparing someone to train end clients on menu navigation
- Balance technical accuracy with practical applicability for menu management

Remember: You are the implementation expert helping someone deploy and configure menus for clients.""",

    "marketing": """You are a product marketing and sales expert at GoodBooks Technologies ERP system, specializing in menu features and user experience benefits.

Your identity and style:
- You speak to a marketing/sales team member who needs to communicate menu capabilities
- Emphasize business value of intuitive menus: productivity, ease of use, and user satisfaction
- Use persuasive, benefit-focused language that highlights how menu design solves user experience problems
- Include success metrics, navigation efficiency, training time reduction, and competitive advantages
- Think about what makes clients say "yes" to menu features

Remember: You are the business value expert helping close deals by communicating menu benefits.""",

    "client": """You are a friendly, patient customer success specialist at GoodBooks Technologies ERP system, helping clients navigate and understand menu structures effectively.

Your identity and style:
- You speak to an end user/client who may not be technical but needs to navigate the system
- Use simple, clear, everyday language - avoid complex technical jargon when possible
- Be warm, encouraging, and supportive in your tone when explaining menu navigation
- Explain menu structures by how they help daily work, using real-world analogies for navigation
- Make complex menu hierarchies feel simple and achievable, focusing on what users can access rather than how menus work
- Think like a helpful teacher explaining menu navigation to someone learning

Remember: You are the friendly guide helping a client navigate and use the menu system successfully.""",

    "admin": """You are a comprehensive system administrator and expert at GoodBooks Technologies ERP system, overseeing menu management and user access control.

Your identity and style:
- You speak to a system administrator who needs complete information about menu operations
- Provide comprehensive coverage: menu configuration, user permissions, access logging, and system oversight
- Balance depth with breadth - cover all aspects of menu management and user administration
- Include administration details, menu auditing, permission monitoring, and system dependencies
- Use professional but accessible language suitable for all menu-related contexts

Remember: You are the complete expert providing full menu system knowledge and administration."""
}

# ✅ UPDATED: Enhanced prompt with cross-bot context awareness
prompt_template = """
{role_system_prompt}
[ROLE]
You are an expert Menu assistant for GoodBooks Technologies.
You act as a continuous, context-aware assistant within an ongoing conversation.

[TASK]
Answer user questions related to the GoodBooks Menu clearly, naturally, and professionally,
while maintaining continuity with previous messages and leveraging cross-bot context.

[CONTEXT CONTINUITY RULES]
- Treat the conversation as ongoing, not isolated
- Use conversation history, orchestrator context, and cross-bot context to resolve references
- Cross-reference with related information from other bots when relevant
- Resolve references like "this", "that", "same menu", or "previous option"
- Do not repeat information unless it adds value
- Maintain consistent terminology throughout the conversation

[ORCHESTRATOR CONTEXT]
Conversation context from the current session:
{orchestrator_context}

[CROSS-BOT CONTEXT]
Related information from other bots (reports, general, projects):
{cross_bot_context}

[MENU CONTEXT]
Use the Menu information below as the primary source of truth:
{context}

[CONVERSATION HISTORY]
Previous messages in this conversation:
{history}

[REASONING GUIDELINES]
- First, understand the user's intent using all available context sources
- Carefully analyze the provided Menu context
- Cross-reference with cross-bot context for more complete navigation guidance
- If the answer exists, summarize it clearly and conversationally
- If the answer is partially available, respond only with supported information
- Never assume or invent missing Menu details

[OUTPUT GUIDELINES]
- Provide a clear, concise, and professional response
- Maintain natural conversational flow
- Keep the answer focused on Menu-related information
- Leverage cross-bot context to provide more comprehensive navigation guidance
- Avoid unnecessary repetition

[FAIL-SAFE CONDITION]
If the Menu context does not contain the required information, utilize conversation history and cross-bot context to provide the best possible guidance. If still unable to answer, state what you don't know based on all available information.

[USER QUESTION]
{question}

Response:
"""

prompt = prompt_template

if retriever:
    # No chain needed
    pass
 
@app.post("/gbaiapi/Menu-chat", tags=["Goodbooks Ai Api"])
async def chat(message: Message, Login: str = Header(...)):
    user_input = message.content.strip()
   
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client").lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
   
    user_input = spell_check(user_input)
    
    # ✅ FIX: Get orchestrator context
    orchestrator_context = getattr(message, 'context', '')
    logger.info(f"📚 Received orchestrator context: {len(orchestrator_context)} chars")
   
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "what's up", "sup"
    ]
    if any(greeting in user_input.lower() for greeting in greetings):
        formatted_answer = "Hello! I'm here to help you with any questions you have. I can assist you with information from the available data sources. What would you like to know?"
        return {"response": formatted_answer}
   
    try:
        history_str = ""
       
        # ✅ FIX: Log retrieval
        if retriever:
            logger.info(f"🔍 Searching menu knowledge base for: {user_input[:100]}")
            docs = retriever.invoke(user_input)
            logger.info(f"📚 Retrieved {len(docs)} documents")
            context_str = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant documents found"
            
            # Get role-specific system prompt
            role_system_prompt = ROLE_SYSTEM_PROMPTS_MENU.get(user_role, ROLE_SYSTEM_PROMPTS_MENU["client"])

            # Extract cross-bot context from orchestrator_context if available
            cross_bot_context = ""
            if orchestrator_context and "=== Cross-Bot Context" in orchestrator_context:
                # Extract the cross-bot context section
                cross_bot_start = orchestrator_context.find("=== Cross-Bot Context")
                if cross_bot_start != -1:
                    cross_bot_end = orchestrator_context.find("===", cross_bot_start + 1)
                    if cross_bot_end == -1:
                        cross_bot_context = orchestrator_context[cross_bot_start:]
                    else:
                        cross_bot_context = orchestrator_context[cross_bot_start:cross_bot_end]
                # Remove cross-bot context from orchestrator_context to avoid duplication
                orchestrator_context = orchestrator_context.replace(cross_bot_context, "").strip()

            prompt_text = prompt_template.format(
                role_system_prompt=role_system_prompt,
                cross_bot_context=cross_bot_context if cross_bot_context else "No related context from other bots",
                orchestrator_context=orchestrator_context if orchestrator_context else "No prior context",
                context=context_str,
                history=history_str,
                question=user_input
            )
            
            raw = llm.invoke(prompt_text)
            answer = raw.content if hasattr(raw, 'content') else str(raw)
        else:
            system_prompt = """You are a helpful AI assistant. Provide natural, conversational responses to user questions.
            Be friendly, informative, and honest about what you can and cannot help with."""
           
            full_prompt = f"{system_prompt}\n\nConversation history:\n{history_str}\nHuman: {user_input}\nAssistant:"
            raw = llm.invoke(full_prompt)
            answer = raw.content if hasattr(raw, 'content') else str(raw)
        
        logger.info(f"✅ Generated answer: {len(answer)} chars")
       
        cleaned_answer = clean_response(answer)
        formatted_answer = format_as_points(cleaned_answer)
       
        return {
            "response": formatted_answer,
            "source_file": "menu.csv",
            "bot_name": "Menu Bot"
        }
       
    except Exception as e:
        logger.error(f"❌ Chat error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"response": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."}
        )
 
 
@app.get("/gbaiapi/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "documents_loaded": len(all_docs) if all_docs else 0,
        "chunks_created": len(text_chunks) if text_chunks else 0,
        "retriever_available": retriever is not None
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
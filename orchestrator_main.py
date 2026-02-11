import json
import os
import logging
import traceback
import re
import uuid
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from google.cloud import firestore
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
from shared_resources import ai_resources

# Import bot modules
try:
    import formula_bot
    FORMULA_BOT_AVAILABLE = True
    logging.info("Formula bot imported successfully")
except ImportError as e:
    FORMULA_BOT_AVAILABLE = False
    logging.warning(f"Formula bot not available: {e}")

try:
    import report_bot
    REPORT_BOT_AVAILABLE = True
    logging.info("Report bot imported successfully")
except ImportError as e:
    REPORT_BOT_AVAILABLE = False
    logging.warning(f"Report bot not available: {e}")

try:
    import menu_bot
    MENU_BOT_AVAILABLE = True
    logging.info("Menu bot imported successfully")
except ImportError as e:
    MENU_BOT_AVAILABLE = False
    logging.warning(f"Menu bot not available: {e}")

try:
    import project_bot
    PROJECT_BOT_AVAILABLE = True
    logging.info("Project bot imported successfully")
except ImportError as e:
    PROJECT_BOT_AVAILABLE = False
    logging.warning(f"Project bot not available: {e}")

try:
    import general_bot
    GENERAL_BOT_AVAILABLE = True
    logging.info("General bot imported successfully")
except ImportError as e:
    GENERAL_BOT_AVAILABLE = False
    logging.warning(f"General bot not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP Configuration
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
db = firestore.Client(project=GCP_PROJECT_ID)
storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

logger.info(f"Connected to GCP Project: {GCP_PROJECT_ID}, Bucket: {GCS_BUCKET_NAME}")

# ===========================
# EXECUTOR (FOR BLOCKING I/O ONLY)
# ===========================
EXECUTOR = ThreadPoolExecutor(max_workers=4)

def run_bg(func, *args):
    """Helper to run blocking I/O in thread pool"""
    return asyncio.to_thread(func, *args)

class UserRole:
    DEVELOPER = "developer"
    IMPLEMENTATION = "implementation"
    MARKETING = "marketing"
    CLIENT = "client"
    ADMIN = "admin"
    SYSTEM_ADMIN = "system admin"
    MANAGER = "manager"
    SALES = "sales"

# ===========================
# ===========================
# SOURCE TRACKER WITH CONTEXT FILTERING
# ===========================
class SourceTracker:
    """Track actual sources and filter relevant contexts"""
    
    @staticmethod
    def extract_document_names_from_memory(memories: List[Dict]) -> List[str]:
        """Extract actual document/file names from memories"""
        sources = []
        for memory in memories:
            # Get the actual document name, not memory_id
            doc_name = memory.get("source_document", memory.get("document_name", ""))
            
            if doc_name and doc_name not in sources:
                sources.append(doc_name)
        
        return sources if sources else ["Default_Knowledge_Base"]
    
    @staticmethod
    def is_memory_relevant_to_query(memory: Dict, user_question: str,
                                     relevance_threshold: float = 0.5) -> bool:
        """Check if memory is actually relevant to current question"""
        relevance_score = memory.get("relevance_score", 0)

        # Use memories with moderate relevance to provide better context
        if relevance_score < relevance_threshold:
            return False
        
        # Check for question type match
        memory_question_type = memory.get("question_type", "")
        current_question_type = extract_question_type(user_question)
        
        # If question types don't match, skip this memory
        if memory_question_type and current_question_type:
            if memory_question_type != current_question_type:
                return False
        
        return True
    
    @staticmethod
    def format_sources_for_response(source_files: List[str]) -> Dict:
        """Format sources - only actual file names"""
        unique_sources = list(set(source_files))
        
        return {
            "sources_count": len(unique_sources),
            "sources": unique_sources
        }

# ===========================
# QUESTION TYPE EXTRACTION
# ===========================
def extract_question_type(question: str) -> str:
    """Determine the type/category of question"""
    question_lower = question.lower()
    
    # Configuration questions
    if any(word in question_lower for word in ["configure", "setup", "set up", "install", "deploy"]):
        return "configuration"
    
    # Module/Feature questions
    if any(word in question_lower for word in ["module", "feature", "functionality", "capability"]):
        return "module_feature"
    
    # Integration questions
    if any(word in question_lower for word in ["integrate", "integration", "connect", "api"]):
        return "integration"
    
    # Troubleshooting questions
    if any(word in question_lower for word in ["error", "issue", "problem", "bug", "not working", "fix"]):
        return "troubleshooting"
    
    # General questions
    return "general"

# Initialize source tracker
source_tracker = SourceTracker()

# Mapping ROLEID to internal role names
ROLEID_TO_NAME = {
    "-1799999969": "admin",          # SystemAdmin / Administrator
    "-1499999995": "admin",          # Unisoft Manager
    "-1499999994": "marketing",      # MARKETING MANAGER
    "-1499999993": "marketing",      # Marketing Assistant
    "-1499999992": "client",         # Accounts Assistant
    "-1499999991": "client",         # HR-DEPARTMENT
    "-1499999989": "marketing",      # Marketing Assistant1
    "-1499999988": "admin",          # ADMINONLY
    "-1499999987": "admin",          # QC
    "-1499999986": "implementation", # Implmentation Team 
    "-1499999984": "client",         # HR USER
    "-1499999982": "client",         # HR-Assistannt
    "-1499999981": "marketing",      # SALES-DEPARTMENT
    "-1499999980": "marketing",      # SALES-AST
    "-1499999979": "admin",          # Account Manager
    "-1499999978": "developer",      # Developer
    "-1499999967": "client"          # cashier
}

# Memory storage
MEMORY_VECTORSTORE_PATH = "conversational_memory_vectorstore"
chats_db = {}
conversational_memory_metadata = {}
user_sessions = {}

# ===========================
# PERFORMANCE MONITORING MIDDLEWARE
# ===========================
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.2f}s"
        
        if process_time > 1.0:
            logger.info(f"⏱️ Request {request.url.path} took {process_time:.2f}s")
        else:
            logger.info(f"⚡ Request {request.url.path} took {process_time:.2f}s")
        
        return response

# ===========================
# HARDCODED GREETINGS (INSTANT RESPONSE)
# ===========================
GREETING_PATTERNS = [
    r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|sup|yo|howdy)$',
    r'^(hi|hello|hey)\s+(there|everyone|all)$',
    r'^how are you\??$',
    r'^what\'?s up\??$'
]

ROLE_GREETINGS = {
    UserRole.DEVELOPER: """Hi! I'm your GoodBooks ERP technical assistant.

I can help with:
• System architecture & APIs
• Database schemas & queries
• Code examples & implementation
• Technical troubleshooting

What technical challenge can I solve?""",

    UserRole.IMPLEMENTATION: """Hello! I'm your GoodBooks implementation consultant.

I assist with:
• Setup & configuration steps
• Client deployment procedures
• Best practices & troubleshooting

How can I help with implementation?""",

    UserRole.MARKETING: """Hi! I'm your GoodBooks product expert.

I help with:
• Business value & ROI metrics
• Competitive advantages
• Sales materials & success stories

What would you like to explore?""",

    UserRole.CLIENT: """Hello! Welcome to GoodBooks ERP! 😊

I'm here to help you with:
• Understanding features
• Step-by-step guidance
• Finding what you need

What would you like to learn?""",

    UserRole.ADMIN: """Hello! I'm your GoodBooks system administrator assistant.

I help with:
• System administration
• Configuration management
• User permissions & monitoring

What can I assist you with?""",

    UserRole.SYSTEM_ADMIN: """Hello! I'm your GoodBooks senior system administrator assistant.

I'm here to help with:
• Infrastructure & server health
• Data security & access control
• System optimization & maintenance
• Technical administration

How can I help you keep the system running perfectly today?""",

    UserRole.MANAGER: """Hello! I'm your GoodBooks strategic management assistant.

I can assist with:
• Operational oversight & efficiency
• Performance metrics & strategic insights
• Team coordination & workflows
• Business process optimization

What management goals can I help you achieve today?""",

    UserRole.SALES: """Hello! I'm your GoodBooks sales and revenue assistant.

I help with:
• Lead management & pipelines
• Sales forecasting & performance
• CRM optimization & customer insights
• Revenue growth strategies

How can I help you drive more sales today?"""
}

def is_greeting(text: str) -> bool:
    """Fast greeting detection - only for very simple greetings"""
    text_lower = text.lower().strip()
    
    # Only match very simple, short greetings
    if len(text_lower.split()) > 4:
        return False
    
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def get_greeting_response(user_role: str) -> str:
    """Get instant greeting response"""
    return ROLE_GREETINGS.get(user_role, ROLE_GREETINGS[UserRole.CLIENT])

# ===========================
# ROLE-BASED SYSTEM PROMPTS
# ===========================
ROLE_SYSTEM_PROMPTS = {
    UserRole.DEVELOPER: """You are a senior software architect and technical expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a fellow developer/engineer who understands technical concepts
- Use technical terminology, architecture patterns, and code concepts naturally
- Discuss APIs, databases, integrations, algorithms, and system design
- Provide technical depth with implementation details
- Mention code examples, endpoints, schemas when relevant
- Think like a senior developer explaining to a peer

Remember: You are the technical expert helping another technical person. Be precise, detailed, and technical.""",

    UserRole.IMPLEMENTATION: """You are an experienced implementation consultant at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to an implementation team member who guides clients through setup
- Provide step-by-step configuration and deployment instructions
- Focus on practical "how-to" guidance for client rollouts
- Include best practices, common issues, and troubleshooting tips
- Explain as if preparing someone to train end clients
- Balance technical accuracy with practical applicability

Remember: You are the implementation expert helping someone deploy the system for clients.""",

    UserRole.MARKETING: """You are a product marketing and sales expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a marketing/sales team member who needs to sell the solution
- Emphasize business value, ROI, competitive advantages, and client benefits
- Use persuasive, benefit-focused language that highlights solutions to business problems
- Include success metrics, cost savings, efficiency gains, and market differentiation
- Think about what makes clients say "yes" to purchasing

Remember: You are the business value expert helping close deals and communicate benefits.""",

    UserRole.CLIENT: """You are a friendly, patient customer success specialist at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to an end user/client who may not be technical
- Use simple, clear, everyday language - avoid all technical jargon
- Be warm, encouraging, and supportive in your tone
- Explain features by how they help daily work, using real-world analogies
- Make complex things feel simple and achievable
- Think like a helpful teacher explaining to someone learning

Remember: You are the friendly guide helping a client use the system successfully.""",

    UserRole.ADMIN: """You are a comprehensive system administrator and expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a system administrator who needs complete information
- Provide comprehensive coverage: technical, business, and operational aspects
- Balance depth with breadth - cover all angles of a topic
- Include administration, configuration, management, and oversight details
- Use professional but accessible language suitable for all contexts

Remember: You are the complete expert providing full system knowledge.""",

    UserRole.SYSTEM_ADMIN: """You are a senior system administrator and infrastructure expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a fellow system admin or IT manager responsible for system health and infrastructure
- Use technical terminology for server management, cloud infrastructure, security protocols, and system maintenance
- Discuss database performance, backup strategies, user access control, and API rate limiting
- Provide technical depth with system monitoring, logs analysis, and resource optimization details
- Mention security best practices, system updates, and server configurations when relevant
- Think like a senior administrator ensuring 99.9% uptime and data security

Remember: You are the infrastructure expert ensuring the system runs smoothly, securely, and efficiently.""",

    UserRole.MANAGER: """You are a strategic management consultant and operational expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a business manager, department head, or team lead focused on efficiency and oversight
- Use business terminology for resource allocation, performance tracking, project timelines, and operational workflows
- Discuss team productivity, cost-benefit analysis, strategic planning, and cross-departmental coordination
- Provide high-level insights into organizational performance, risk management, and process improvement
- Mention reporting dashboards, approval workflows, and business intelligence concepts when relevant
- Think like a manager optimizing team output and business processes

Remember: You are the operational expert helping managers make data-driven decisions and optimize business performance.""",

    UserRole.SALES: """You are a senior sales strategist and revenue growth expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a sales professional or account manager focused on closing deals and managing pipelines
- Use sales terminology for lead qualification, sales cycles, quotation management, and customer retention
- Discuss pricing strategies, sales forecasting, CRM optimization, and market positioning
- Provide practical guidance on managing customer relationships, following up on leads, and converting prospects
- Mention sales reports, target tracking, and customer interaction histories when relevant
- Think like a top-performing sales manager driving revenue and customer satisfaction

Remember: You are the sales expert helping the team win more business and manage customer relationships effectively."""

}
# ===========================
# AI ORCHESTRATOR SYSTEM PROMPT (ENHANCED)
# ===========================
ORCHESTRATOR_SYSTEM_PROMPT = """You are a routing assistant for GoodBooks ERP. Route the query to ONE bot:

- general: company info, policies, employees, modules, products, features, contact info, leave management, general questions about GoodBooks
- formula: mathematical calculations, expressions, computing numbers, arithmetic operations
- report: data analysis, generating reports, charts, graphs, statistics, viewing data
- menu: navigation help, finding screens, interface guidance, where to find features
- project: project files, project reports, project management, tasks, milestones

Examples:
"What is GoodBooks ERP?" -> general
"Tell me about inventory module" -> general  
"The user asks about leave"-> general
"Calculate 100 * 5" -> formula
"What is 20% of 500?" -> formula
"Show me sales report" -> report
"Generate analysis chart" -> report
"Where is the customer screen?" -> menu
"How do I access invoices?" -> menu
"Show project status" -> project
"View project files" -> project

Query: {question}

Respond with ONLY ONE WORD (general, formula, report, menu, or project):"""

# ===========================
# AI OUT-OF-SCOPE REFUSAL PROMPT
# ===========================
OUT_OF_SCOPE_SYSTEM_PROMPT = """You are a GoodBooks ERP assistant speaking to a {role}.

The user asked about something outside GoodBooks ERP scope or you couldn't find the answer.

User question: {question}

Politely explain that you're here to help with GoodBooks ERP and redirect them to relevant GoodBooks features. Keep it brief and appropriate for the {role} role.

Response:"""

# ===========================
# CONVERSATION THREADS
# ===========================
class ConversationThread:
    def __init__(self, thread_id: str, username: str, title: str = None):
        self.thread_id = thread_id
        self.username = username
        self.title = title or "New Conversation"
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.messages = []
        self.is_active = True
        self.user_role = None
        self.user_name = None
        
    def add_message(self, user_message: str, bot_response: str, bot_type: str):
        message = {
            "id": str(uuid.uuid4()),
            "user_message": user_message,
            "bot_response": bot_response,
            "bot_type": bot_type,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        if self.title == "New Conversation" and len(self.messages) == 1:
            self.title = self._generate_title_from_message(user_message)
    
    def _generate_title_from_message(self, message: str) -> str:
        title = message.strip()
        title = re.sub(r'^(what\s+is\s+|tell\s+me\s+about\s+|how\s+to\s+|can\s+you\s+)', '', title, flags=re.IGNORECASE)
        return (title[:47] + "...") if len(title) > 50 else title.capitalize() if title else "New Conversation"

    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "username": self.username,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "is_active": self.is_active,
            "message_count": len(self.messages),
            "user_role": self.user_role,
            "user_name": self.user_name
        }

class ConversationHistoryManager:
    def __init__(self):
        self.threads = {}
        self.load_threads()
    
    def load_threads(self):
        try:
            logger.info("Loading recent threads from Firestore...")
            threads_ref = db.collection('conversation_threads')
            
            # Load only recent 100 threads to speed up startup
            # Older threads will be loaded on-demand when accessed
            docs = threads_ref.order_by('updated_at', direction=firestore.Query.DESCENDING).limit(100).stream()
            
            for doc in docs:
                thread_data = doc.to_dict()
                if not thread_data: 
                    continue
                thread = ConversationThread(
                    thread_data.get("thread_id"), 
                    thread_data.get("username"), 
                    thread_data.get("title")
                )
                thread.created_at = thread_data.get("created_at")
                thread.updated_at = thread_data.get("updated_at")
                thread.messages = thread_data.get("messages", [])
                thread.is_active = thread_data.get("is_active", True)
                thread.user_role = thread_data.get("user_role")
                thread.user_name = thread_data.get("user_name")
                self.threads[thread_data.get("thread_id")] = thread
            logger.info(f"✅ Loaded {len(self.threads)} recent threads from Firestore")
        except Exception as e:
            logger.error(f"Failed to load threads from Firestore: {e}", exc_info=True)
            self.threads = {}

    def save_threads(self):
        try:
            for thread_id, thread in self.threads.items():
                thread_ref = db.collection('conversation_threads').document(thread_id)
                thread_ref.set(thread.to_dict())
        except Exception as e:
            logger.error(f"Error saving threads to Firestore: {e}")

    def create_new_thread(self, username: str, initial_message: str = None) -> str:
        thread_id = str(uuid.uuid4())
        thread = ConversationThread(thread_id, username)
        if initial_message:
            thread.title = thread._generate_title_from_message(initial_message)
        self.threads[thread_id] = thread
        self.save_threads()
        logger.info(f"Created new thread {thread_id} for {username}")
        return thread_id

    def add_message_to_thread(self, thread_id: str, user_message: str, bot_response: str, bot_type: str):
        if thread_id in self.threads:
            self.threads[thread_id].add_message(user_message, bot_response, bot_type)
            self.save_threads()
    
    def get_user_threads(self, username: str, limit: int = 50) -> List[Dict]:
        user_threads = [
            thread.to_dict() for thread in self.threads.values() 
            if thread.username == username and thread.is_active
        ]
        user_threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_threads[:limit]
    
    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        return self.threads.get(thread_id)
    
    def delete_thread(self, thread_id: str, username: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].is_active = False
            self.save_threads()
            return True
        return False
    
    def rename_thread(self, thread_id: str, username: str, new_title: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].title = new_title
            self.threads[thread_id].updated_at = datetime.now().isoformat()
            self.save_threads()
            return True
        return False
    
    def cleanup_old_threads(self, days_to_keep: int = 90):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        deleted_count = 0
        threads_to_delete = []
        
        for thread_id, thread in self.threads.items():
            if not thread.is_active and thread.updated_at < cutoff_iso:
                threads_to_delete.append(thread_id)
        
        if threads_to_delete:
            for thread_id in threads_to_delete:
                del self.threads[thread_id]
                db.collection('conversation_threads').document(thread_id).delete()
                deleted_count += 1
            logger.info(f"Cleaned up {deleted_count} old threads")

# Initialize as None - will be loaded at startup
history_manager = None

# ===========================
# MEMORY SYSTEM
# ===========================
class EnhancedConversationalMemory:
    def __init__(self, vectorstore_path: str, metadata_file: str, embeddings):
        self.vectorstore_path = vectorstore_path
        self.metadata_file = metadata_file
        self.embeddings = embeddings
        self.memory_vectorstore = None
        self.memory_counter = 0
        self.load_memory_vectorstore()
    
    def load_memory_vectorstore(self):
        try:
            faiss_index_blob = bucket.blob(f"{self.vectorstore_path}.faiss")
            pkl_index_blob = bucket.blob(f"{self.vectorstore_path}.pkl")

            if faiss_index_blob.exists() and pkl_index_blob.exists():
                logger.info("Downloading FAISS index from Cloud Storage...")
                os.makedirs(self.vectorstore_path, exist_ok=True)
                faiss_index_blob.download_to_filename(f"{self.vectorstore_path}.faiss")
                pkl_index_blob.download_to_filename(f"{self.vectorstore_path}.pkl")

                self.memory_vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded memory from GCS.")
            else:
                raise FileNotFoundError("FAISS index not found in GCS bucket.")

        except Exception as e:
            logger.error(f"Error loading memory from GCS, creating new one: {e}")
            dummy_doc = Document(page_content="Memory system initialized")
            self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
    
    def store_conversation_turn(self, username: str, user_message: str, bot_response: str, bot_type: str, user_role: str, thread_id: str = None):
        try:
            timestamp = datetime.now().isoformat()
            memory_id = f"{username}_{self.memory_counter}_{int(datetime.now().timestamp())}"
            
            conversation_context = f"User ({user_role}): {user_message} | Bot ({bot_type}): {bot_response[:1000]}"
            
            memory_doc = Document(
                page_content=conversation_context,
                metadata={
                    "memory_id": memory_id,
                    "username": username,
                    "user_role": user_role,
                    "timestamp": timestamp,
                    "user_message": user_message,
                    "bot_response": bot_response[:500],
                    "bot_type": bot_type,
                    "thread_id": thread_id
                }
            )
            
            self.memory_vectorstore.add_documents([memory_doc])
            
            conversational_memory_metadata[memory_id] = {
                "username": username,
                "user_role": user_role,
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response[:1000],
                "bot_type": bot_type,
                "thread_id": thread_id
            }
            
            self.memory_counter += 1
            if self.memory_counter % 20 == 0:
                logger.info("Saving FAISS index to Cloud Storage...")
                self.memory_vectorstore.save_local(self.vectorstore_path)

                faiss_blob = bucket.blob(f"{self.vectorstore_path}.faiss")
                pkl_blob = bucket.blob(f"{self.vectorstore_path}.pkl")
                faiss_blob.upload_from_filename(f"{self.vectorstore_path}.faiss")
                pkl_blob.upload_from_filename(f"{self.vectorstore_path}.pkl")
                logger.info("Successfully saved FAISS index to GCS.")
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
    
    def retrieve_contextual_memories(self, username: str, query: str, k: int = 3, thread_id: str = None, thread_isolation: bool = False) -> List[Dict]:
        try:
            # Enhanced memory retrieval with multiple strategies
            all_docs = []

            # Primary search with original query
            docs = self.memory_vectorstore.similarity_search(query, k=k * 3)
            all_docs.extend(docs)

            # Secondary search with simplified query for broader matching
            if len(query.split()) > 4:
                simplified_query = " ".join(query.split()[:4])  # First 4 words
                try:
                    simplified_docs = self.memory_vectorstore.similarity_search(simplified_query, k=k)
                    all_docs.extend(simplified_docs)
                except Exception as e:
                    logger.warning(f"Simplified query search failed: {e}")

            # Tertiary search with keywords only
            keywords = [word for word in query.lower().split() if len(word) > 3 and word not in ['what', 'how', 'when', 'where', 'why', 'which', 'that', 'this', 'there', 'here']]
            if keywords:
                keyword_query = " ".join(keywords[:3])  # Top 3 keywords
                try:
                    keyword_docs = self.memory_vectorstore.similarity_search(keyword_query, k=k)
                    all_docs.extend(keyword_docs)
                except Exception as e:
                    logger.warning(f"Keyword query search failed: {e}")

            user_memories = {}
            seen_memories = set()

            for doc in all_docs:
                if (doc.metadata.get("username") == username and
                    doc.metadata.get("memory_id") != "init"):

                    # ENHANCED: Better thread isolation logic for existing threads
                    if thread_isolation and thread_id:
                        doc_thread_id = doc.metadata.get("thread_id")

                        # Always allow memories from the same thread
                        if doc_thread_id == thread_id:
                            pass  # Allow this memory
                        else:
                            # For memories from other threads, only allow recent ones (within 2 hours)
                            try:
                                memory_time = datetime.fromisoformat(doc.metadata.get("timestamp", "").replace('Z', '+00:00'))
                                hours_old = (datetime.now() - memory_time).total_seconds() / 3600
                                if hours_old > 2:  # Skip old memories from other threads
                                    continue
                            except:
                                continue  # Skip if can't parse timestamp

                    memory_id = doc.metadata.get("memory_id")

                    # IMPROVED: Better deduplication - use memory_id instead of content hash
                    if memory_id in seen_memories:
                        continue
                    seen_memories.add(memory_id)

                    if memory_id not in user_memories:
                        # Enhanced memory object with relevance scoring
                        relevance_score = self._calculate_memory_relevance(doc, query, thread_id)
                        user_memories[memory_id] = {
                            "memory_id": memory_id,
                            "timestamp": doc.metadata.get("timestamp"),
                            "user_message": doc.metadata.get("user_message"),
                            "bot_response": doc.metadata.get("bot_response"),
                            "bot_type": doc.metadata.get("bot_type"),
                            "user_role": doc.metadata.get("user_role"),
                            "thread_id": doc.metadata.get("thread_id"),
                            "content": doc.page_content,
                            "relevance_score": relevance_score
                        }

            # Sort by relevance score, then by recency
            sorted_memories = sorted(
                user_memories.values(),
                key=lambda x: (x["relevance_score"], x["timestamp"]),
                reverse=True
            )

            # Return top k memories
            top_memories = sorted_memories[:k]
            logger.info(f"Enhanced memory retrieval: {len(top_memories)} memories from {len(all_docs)} candidates")
            return top_memories

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def _calculate_memory_relevance(self, doc, query: str, current_thread_id: str = None) -> float:
        """Calculate relevance score for memory based on multiple factors"""
        score = 0.0

        # Base similarity score (from vector search)
        score += 1.0

        # Thread continuity bonus
        if current_thread_id and doc.metadata.get("thread_id") == current_thread_id:
            score += 0.5

        # Recency bonus (newer memories get slight preference)
        try:
            memory_time = datetime.fromisoformat(doc.metadata.get("timestamp", "").replace('Z', '+00:00'))
            hours_old = (datetime.now() - memory_time).total_seconds() / 3600
            if hours_old < 24:
                score += 0.3
            elif hours_old < 168:  # 1 week
                score += 0.1
        except:
            pass

        # Query term matching bonus
        query_terms = set(query.lower().split())
        memory_content = doc.page_content.lower()
        matching_terms = query_terms.intersection(set(memory_content.split()))
        if matching_terms:
            score += len(matching_terms) * 0.1

        # Bot type consistency bonus (prefer same bot type for continuity)
        # This will be set by the calling context
        score += 0.0  # Placeholder for future enhancement

        return score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# Initialize as None - will be loaded at startup
embeddings = None
enhanced_memory = None

# ===========================
# SHARED CONTEXT REGISTRY SYSTEM
# ===========================
class SharedContextRegistry:
    """Registry for sharing context between bots to enable cross-bot continuity"""

    def __init__(self):
        self.context_store = {}  # {username: {bot_type: {context_key: context_data}}}
        self.max_context_age = 3600  # 1 hour in seconds
        self.max_contexts_per_bot = 10

    def store_bot_context(self, username: str, bot_type: str, context_key: str, context_data: Dict):
        """Store context from a bot for potential sharing with other bots"""
        if username not in self.context_store:
            self.context_store[username] = {}

        if bot_type not in self.context_store[username]:
            self.context_store[username][bot_type] = {}

        # Add timestamp for context aging
        context_data['_timestamp'] = time.time()
        context_data['_bot_type'] = bot_type

        # Store context with key
        self.context_store[username][bot_type][context_key] = context_data

        # Limit number of contexts per bot to prevent memory bloat
        if len(self.context_store[username][bot_type]) > self.max_contexts_per_bot:
            # Remove oldest context
            oldest_key = min(
                self.context_store[username][bot_type].keys(),
                key=lambda k: self.context_store[username][bot_type][k]['_timestamp']
            )
            del self.context_store[username][bot_type][oldest_key]

        logger.info(f"📚 Stored context '{context_key}' for {username}:{bot_type}")

    def get_relevant_contexts(self, username: str, current_bot_type: str, query: str, max_contexts: int = 3) -> List[Dict]:
        """Get relevant contexts from other bots that might help with the current query"""
        if username not in self.context_store:
            return []

        relevant_contexts = []
        current_time = time.time()

        # Define bot relationships for context sharing
        bot_relationships = {
            "general": ["report", "menu", "project"],  # General can benefit from specific bot contexts
            "report": ["general", "menu"],  # Reports often relate to navigation and general info
            "menu": ["general", "report"],  # Menu navigation relates to reports and general features
            "formula": ["general", "report"],  # Formulas might relate to reports and general calculations
            "project": ["general", "report"]  # Projects can benefit from general and report contexts
        }

        related_bots = bot_relationships.get(current_bot_type, [])

        for bot_type in related_bots:
            if bot_type not in self.context_store[username]:
                continue

            # Get contexts from related bot, sorted by recency
            bot_contexts = self.context_store[username][bot_type]
            sorted_contexts = sorted(
                bot_contexts.items(),
                key=lambda x: x[1]['_timestamp'],
                reverse=True
            )

            for context_key, context_data in sorted_contexts:
                # Check if context is still fresh
                if current_time - context_data['_timestamp'] > self.max_context_age:
                    continue

                # Calculate relevance to current query
                relevance_score = self._calculate_context_relevance(context_data, query)

                if relevance_score > 0.3:  # Minimum relevance threshold
                    relevant_contexts.append({
                        'bot_type': bot_type,
                        'context_key': context_key,
                        'context_data': context_data,
                        'relevance_score': relevance_score
                    })

        # Sort by relevance and return top contexts
        relevant_contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_contexts = relevant_contexts[:max_contexts]

        logger.info(f"🔗 Found {len(top_contexts)} relevant cross-bot contexts for {username}:{current_bot_type}")
        return top_contexts

    def _calculate_context_relevance(self, context_data: Dict, query: str) -> float:
        """Calculate how relevant a stored context is to the current query"""
        score = 0.0

        # Remove metadata fields for relevance calculation
        clean_context = {k: v for k, v in context_data.items() if not k.startswith('_')}

        # Convert context to searchable text
        context_text = " ".join(str(v) for v in clean_context.values() if isinstance(v, (str, int, float)))
        context_text = context_text.lower()
        query_lower = query.lower()

        # Keyword matching
        query_words = set(query_lower.split())
        context_words = set(context_text.split())

        # Exact word matches
        exact_matches = len(query_words.intersection(context_words))
        score += exact_matches * 0.2

        # Partial word matches (contains)
        partial_matches = 0
        for q_word in query_words:
            if any(q_word in c_word or c_word in q_word for c_word in context_words):
                partial_matches += 1
        score += partial_matches * 0.1

        # Topic similarity based on bot type
        bot_type = context_data.get('_bot_type', '')
        if bot_type == 'report' and any(word in query_lower for word in ['data', 'report', 'analysis', 'chart']):
            score += 0.3
        elif bot_type == 'menu' and any(word in query_lower for word in ['find', 'where', 'navigate', 'screen']):
            score += 0.3
        elif bot_type == 'project' and any(word in query_lower for word in ['project', 'task', 'milestone']):
            score += 0.3

        return min(score, 1.0)  # Cap at 1.0

    def cleanup_old_contexts(self):
        """Remove contexts older than max_context_age"""
        current_time = time.time()
        cleaned_count = 0

        for username in list(self.context_store.keys()):
            for bot_type in list(self.context_store[username].keys()):
                contexts_to_remove = []
                for context_key, context_data in self.context_store[username][bot_type].items():
                    if current_time - context_data['_timestamp'] > self.max_context_age:
                        contexts_to_remove.append(context_key)

                for context_key in contexts_to_remove:
                    del self.context_store[username][bot_type][context_key]
                    cleaned_count += 1

                # Remove empty bot types
                if not self.context_store[username][bot_type]:
                    del self.context_store[username][bot_type]

            # Remove empty users
            if not self.context_store[username]:
                del self.context_store[username]

        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} old contexts from registry")

# Initialize shared context registry
shared_context_registry = SharedContextRegistry()

# ===========================
# BOT WRAPPERS (WITH ENHANCED LOGGING)
# ===========================
class EnhancedMessage:
    """Enhanced message object that includes conversation context"""
    def __init__(self, content: str, context: str = ""):
        self.content = content
        self.context = context

class GeneralBotWrapper:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "anonymous") -> str:
        if not GENERAL_BOT_AVAILABLE:
            logger.warning("❌ General bot not available")
            return None
        try:
            logger.info(f"📞 Calling general_bot with question: {question[:100]}")
            logger.info(f"📚 Passing context: {len(context)} chars")
            
            message = EnhancedMessage(question, context)
            login_header = json.dumps({"UserName": username, "Role": user_role})
            
            result = await general_bot.chat(message, Login=login_header)
            
            response = None
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                response = body.get("response")
            elif isinstance(result, dict):
                response = result.get("response")
            else:
                response = str(result) if result else None
            
            if response:
                response_lower = response.lower()
                refusal_patterns = [
                    "i don't have access",
                    "i do not have access",
                    "unable to provide",
                    "cannot provide",
                    "don't have information",
                    "do not have information",
                    "i am unable to",
                    "i'm unable to"
                ]
                
                is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
                
                if is_refusal:
                    logger.warning(f"⚠️ General bot returned refusal: {response[:150]}")
                    return None
                else:
                    logger.info(f"✅ General bot returned valid answer: {response[:100]}")
                    return response
            else:
                logger.warning("⚠️ General bot returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ General bot error: {e}", exc_info=True)
            return None

class FormulaBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "anonymous") -> str:
        if not FORMULA_BOT_AVAILABLE:
            logger.warning("❌ Formula bot not available")
            return None
        try:
            logger.info(f"📞 Calling formula_bot with question: {question[:100]}")
            logger.info(f"📚 Passing context: {len(context)} chars")
            
            message = EnhancedMessage(question, context)
            login_header = json.dumps({"UserName": username, "Role": user_role})
            
            result = await formula_bot.chat(message, Login=login_header)
            
            response = None
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                response = body.get("response")
            elif isinstance(result, dict):
                response = result.get("response")
            else:
                response = str(result) if result else None
            
            if response:
                response_lower = response.lower()
                refusal_patterns = [
                    "i don't have access", "i do not have access", "unable to provide",
                    "cannot provide", "don't have information", "do not have information"
                ]
                
                is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
                
                if is_refusal:
                    logger.warning(f"⚠️ Formula bot returned refusal: {response[:150]}")
                    return None
                else:
                    logger.info(f"✅ Formula bot returned valid answer: {response[:100]}")
                    return response
            else:
                logger.warning("⚠️ Formula bot returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Formula bot error: {e}", exc_info=True)
            return None

class ReportBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "anonymous") -> str:
        if not REPORT_BOT_AVAILABLE:
            logger.warning("❌ Report bot not available")
            return None
        try:
            logger.info(f"📞 Calling report_bot with question: {question[:100]}")
            logger.info(f"📚 Passing context: {len(context)} chars")
            
            message = EnhancedMessage(question, context)
            login_header = json.dumps({"UserName": username, "Role": user_role})
            
            result = await report_bot.report_chat(message, Login=login_header)
            
            response = None
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                response = body.get("response")
            elif isinstance(result, dict):
                response = result.get("response")
            else:
                response = str(result) if result else None
            
            if response:
                response_lower = response.lower()
                refusal_patterns = [
                    "i don't have access", "i do not have access", "unable to provide",
                    "cannot provide", "don't have information", "do not have information"
                ]
                
                is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
                
                if is_refusal:
                    logger.warning(f"⚠️ Report bot returned refusal: {response[:150]}")
                    return None
                else:
                    logger.info(f"✅ Report bot returned valid answer: {response[:100]}")
                    return response
            else:
                logger.warning("⚠️ Report bot returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Report bot error: {e}", exc_info=True)
            return None

class MenuBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "anonymous") -> str:
        if not MENU_BOT_AVAILABLE:
            logger.warning("❌ Menu bot not available")
            return None
        try:
            logger.info(f"📞 Calling menu_bot with question: {question[:100]}")
            logger.info(f"📚 Passing context: {len(context)} chars")
            
            message = EnhancedMessage(question, context)
            login_header = json.dumps({"UserName": username, "Role": user_role})
            
            result = await menu_bot.chat(message, Login=login_header)
            
            response = None
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                response = body.get("response")
            elif isinstance(result, dict):
                response = result.get("response")
            else:
                response = str(result) if result else None
            
            if response:
                response_lower = response.lower()
                refusal_patterns = [
                    "i don't have access", "i do not have access", "unable to provide",
                    "cannot provide", "don't have information", "do not have information"
                ]
                
                is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
                
                if is_refusal:
                    logger.warning(f"⚠️ Menu bot returned refusal: {response[:150]}")
                    return None
                else:
                    logger.info(f"✅ Menu bot returned valid answer: {response[:100]}")
                    return response
            else:
                logger.warning("⚠️ Menu bot returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Menu bot error: {e}", exc_info=True)
            return None

class ProjectBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "anonymous") -> str:
        if not PROJECT_BOT_AVAILABLE:
            logger.warning("❌ Project bot not available")
            return None
        try:
            logger.info(f"📞 Calling project_bot with question: {question[:100]}")
            logger.info(f"📚 Passing context: {len(context)} chars")
            
            message = EnhancedMessage(question, context)
            login_header = json.dumps({"UserName": username, "Role": user_role})
            
            result = await project_bot.project_chat(message, Login=login_header)
            
            response = None
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                response = body.get("response")
            elif isinstance(result, dict):
                response = result.get("response")
            else:
                response = str(result) if result else None
            
            if response:
                response_lower = response.lower()
                refusal_patterns = [
                    "i don't have access", "i do not have access", "unable to provide",
                    "cannot provide", "don't have information", "do not have information"
                ]
                
                is_refusal = any(pattern in response_lower for pattern in refusal_patterns)
                
                if is_refusal:
                    logger.warning(f"⚠️ Project bot returned refusal: {response[:150]}")
                    return None
                else:
                    logger.info(f"✅ Project bot returned valid answer: {response[:100]}")
                    return response
            else:
                logger.warning("⚠️ Project bot returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Project bot error: {e}", exc_info=True)
            return None

# ===========================
# build_conversational_context FUNCTION
# ===========================
def build_conversational_context(username: str, current_query: str, thread_id: str = None, thread_isolation: bool = False) -> str:
    """Build rich conversational context for sub-bots"""
    context_parts = []
    
    session_info = user_sessions.get(username, {})
    if session_info:
        context_parts.append(f"User: {username}")
        total_interactions = session_info.get("total_interactions", 0)
        if total_interactions > 1:
            context_parts.append(f"(Returning user with {total_interactions} previous interactions)")
        context_parts.append("")
    
    if thread_id:
        thread = history_manager.get_thread(thread_id)
        if thread and thread.messages:
            if thread_isolation:
                context_parts.append(f"=== Current Conversation Thread: {thread.title} ===")
                # Increased from 5 to 10 for better continuity
                recent_messages = thread.messages[-10:]
            else:
                context_parts.append(f"=== Recent Conversation ===")
                # Increased from 3 to 7
                recent_messages = thread.messages[-7:]
            
            if recent_messages:
                for i, msg in enumerate(recent_messages, 1):
                    context_parts.append(f"\nTurn {i}:")
                    # Increased truncation limit from 200 to 1000 for better context
                    context_parts.append(f"User: {msg['user_message'][:1000]}")
                    context_parts.append(f"Assistant ({msg['bot_type']}): {msg['bot_response'][:1000]}")
                context_parts.append("")
    
    memories = enhanced_memory.retrieve_contextual_memories(
        username, current_query, k=2, thread_id=thread_id, thread_isolation=thread_isolation
    )
    
    if memories:
        context_parts.append("=== Related Past Interactions ===")
        for i, memory in enumerate(memories, 1):
            context_parts.append(f"\nPast Interaction {i}:")
            context_parts.append(f"Previous Q: {memory.get('user_message', '')[:150]}")
            context_parts.append(f"Previous A: {memory.get('bot_response', '')[:150]}")
        context_parts.append("")
    
    full_context = "\n".join(context_parts)
    
    logger.info(f"📚 Built conversational context:")
    logger.info(f"   - Thread messages: {len(thread.messages) if thread_id and thread else 0}")
    logger.info(f"   - Retrieved memories: {len(memories)}")
    logger.info(f"   - Total context size: {len(full_context)} chars")
    
    return full_context

# ===========================
# FILTERED CONTEXT BUILDER
# ===========================
async def build_filtered_context(username: str, user_question: str, 
                                 thread_id: str = None, 
                                 is_existing_thread: bool = False) -> tuple:
    """
    Build context with PROPER FILTERING - only includes relevant data
    Returns: (filtered_context_string, source_files_used)
    """
    
    # Step 1: Get recent memories with proper thread isolation
    if is_existing_thread and thread_id:
        # ONLY get memories from current thread
        recent_memories = await asyncio.to_thread(
            enhanced_memory.retrieve_contextual_memories,
            username, user_question, 5, thread_id, True
        )
    else:
        # Get memories with thread isolation to prevent cross-contamination
        recent_memories = await asyncio.to_thread(
            enhanced_memory.retrieve_contextual_memories,
            username, user_question, 10, thread_id, True
        )
    
    # Step 2: FILTER memories by relevance and question type match
    filtered_memories = [
        mem for mem in recent_memories
        if source_tracker.is_memory_relevant_to_query(mem, user_question, relevance_threshold=0.3)
    ]
    
    # Step 3: If no highly relevant memories, don't force old history
    if not filtered_memories:
        logger.info(f"⚠️  No relevant memories found. Question type: {extract_question_type(user_question)}")
        source_files = ["Current_Query_Only"]
        context = f"""
User: {username}
Current Question: {user_question}
Question Type: {extract_question_type(user_question)}

Note: No relevant past interactions found. Answer based on current context only.
"""
        return context, source_files
    
    # Step 4: Build context from filtered memories only
    context = f"""
User: {username}
Current Question: {user_question}
Question Type: {extract_question_type(user_question)}

=======================
RELEVANT PAST INTERACTIONS (Filtered by Relevance):
=======================
"""
    
    source_files = []
    
    # Add current thread context if available
    if thread_id:
        thread = history_manager.get_thread(thread_id)
        if thread and thread.messages:
            context += f"\n=== Current Thread: {thread.title} ===\n"
            # Add last 5 messages from thread
            for msg in thread.messages[-5:]:
                context += f"Q: {msg.get('user_message', '')[:200]}\n"
                context += f"A: {msg.get('bot_response', '')[:200]}\n\n"
    
    for i, memory in enumerate(filtered_memories[:3], 1):  # Limit to top 3
        source_file = memory.get("source_document", memory.get("document_name", "Knowledge_Base"))
        if source_file not in source_files:
            source_files.append(source_file)
        
        context += f"""
Memory {i}:
Question: {memory.get('user_message', 'N/A')[:150]}
Answer: {memory.get('bot_response', 'N/A')[:150]}
Source: {source_file}
Relevance Score: {memory.get('relevance_score', 'N/A')}
---
"""
    
    logger.info(f"📍 Built filtered context from {len(source_files)} source(s)")
    logger.info(f"📍 Used {len(filtered_memories)} relevant memories")
    
    return context, source_files

# ===========================
# ENHANCED AI ORCHESTRATION AGENT
# ===========================
class AIOrchestrationAgent:
    def __init__(self):
        # Use centralized AI resources to save memory and reduce latency
        self.routing_llm = ai_resources.routing_llm
        self.response_llm = ai_resources.response_llm
        
        self.bots = {
            "general": GeneralBotWrapper(),
            "formula": FormulaBot(),
            "report": ReportBot(),
            "menu": MenuBot(),
            "project": ProjectBot()
        }
        
        self.intent_cache = {}
    
    def _get_cached_intent(self, question: str) -> Optional[str]:
        """Enhanced keyword-based routing with broader patterns"""
        question_lower = question.lower().strip()
        
        # Formula bot keywords
        formula_keywords = [
            'calculate', 'compute', 'formula', 'math', 'sum', 'average', 'total', 
            'count', 'percentage', 'divide', 'multiply', 'subtract', 'add',
            'equation', 'expression', 'result of', 'what is', 'how much',
            '+', '-', '*', '/', '=', '%', 'mean', 'median', 'gst', 'tax', 'discount',
            'net amount', 'gross', 'valuation', 'variance'
        ]
        has_numbers = any(char.isdigit() for char in question_lower)
        has_math_ops = any(op in question_lower for op in ['+', '-', '*', '/', '=', '%'])
        has_formula_keyword = any(word in question_lower for word in formula_keywords)
        
        if (has_formula_keyword and has_numbers) or has_math_ops:
            logger.info("🚀 Fast route: formula")
            return "formula"
        
        # Report bot keywords
        report_keywords = [
            'report', 'analyze', 'analysis', 'chart', 'graph', 'data', 
            'dashboard', 'visualize', 'show me data', 'statistics', 'stats',
            'metric', 'kpi', 'performance', 'trend', 'summary', 'breakdown',
            'export', 'generate report', 'view report', 'display data',
            'show chart', 'create graph', 'table', 'listing', 'history of',
            'details of', 'ledger', 'balance sheet', 'p&l', 'profit', 'loss'
        ]
        if any(word in question_lower for word in report_keywords):
            logger.info("🚀 Fast route: report")
            return "report"
        
        # Menu bot keywords
        menu_keywords = [
            'menu', 'navigate', 'where is', 'find screen', 'interface', 
            'how to access', 'location of', 'where can i', 'how do i find',
            'show me how to get to', 'navigation', 'screen', 'page',
            'section', 'tab', 'button', 'option', 'find the', 'locate',
            'path to', 'go to', 'how to open', 'where to find', 'accessing',
            'shortcut', 'ui', 'module location'
        ]
        if any(word in question_lower for word in menu_keywords):
            logger.info("🚀 Fast route: menu")
            return "menu"
        
        # Project bot keywords
        project_keywords = [
            'project', 'project file', 'project report', 'project document',
            'project status', 'project management', 'task', 'milestone',
            'deliverable', 'timeline', 'gantt', 'workstream', 'project plan',
            'project details', 'mfile', 'uploaded files', 'project data'
        ]
        if any(word in question_lower for word in project_keywords):
            logger.info("🚀 Fast route: project")
            return "project"
        
        # General bot keywords (to avoid routing LLM for common generic questions)
        general_keywords = [
            'what is', 'who is', 'tell me about', 'explain', 'describe',
            'help with', 'how does', 'info on', 'company', 'goodbooks',
            'features', 'modules', 'support', 'contact', 'employee',
            'leave policy', 'hr', 'it', 'office'
        ]
        if any(word in question_lower for word in general_keywords):
            logger.info("🚀 Fast route: general")
            return "general"
        
        if question_lower in self.intent_cache:
            cached = self.intent_cache[question_lower]
            logger.info(f"🚀 Cache hit: {cached}")
            return cached
        
        return None
    
    async def detect_intent_with_ai(self, question: str, context: str) -> str:
        """Enhanced intent detection with better fallback logic"""
        try:
            cached_intent = self._get_cached_intent(question)
            if cached_intent:
                return cached_intent
            
            prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(question=question)
            
            logger.info(f"🤖 Using AI to route: {question[:80]}")
            
            response = await asyncio.wait_for(
                self.routing_llm.ainvoke(prompt),
                timeout=10.0
            )
            
            intent = response.content.strip().lower()
            logger.info(f"🎯 AI raw response: {intent}")
            
            valid_intents = ["general", "formula", "report", "menu", "project"]
            
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    intent = valid_intent
                    break
            
            if intent not in valid_intents:
                logger.warning(f"⚠️ Invalid AI intent '{intent}', analyzing question structure")
                if has_numbers := any(char.isdigit() for char in question):
                    intent = "formula"
                elif any(word in question.lower() for word in ['what', 'who', 'tell me', 'explain', 'describe']):
                    intent = "general"
                elif any(word in question.lower() for word in ['show', 'display', 'view']):
                    intent = "report"
                elif any(word in question.lower() for word in ['where', 'find', 'locate']):
                    intent = "menu"
                else:
                    intent = "general"
                logger.info(f"📊 Fallback analysis selected: {intent}")
            
            self.intent_cache[question.lower().strip()] = intent
            logger.info(f"✅ Final routing decision: {intent}")
            return intent
            
        except asyncio.TimeoutError:
            logger.error("⏱️ Intent detection timeout (10s), using intelligent fallback")
            fallback = self._get_cached_intent(question)
            if not fallback:
                question_lower = question.lower()
                if any(char.isdigit() for char in question):
                    fallback = "formula"
                elif any(word in question_lower for word in ['what', 'who', 'tell', 'explain', 'describe', 'about']):
                    fallback = "general"
                elif any(word in question_lower for word in ['show', 'display', 'view', 'see']):
                    fallback = "report"
                elif any(word in question_lower for word in ['where', 'find', 'locate', 'access']):
                    fallback = "menu"
                else:
                    fallback = "general"
            logger.info(f"🔍 Timeout fallback route: {fallback}")
            return fallback
        except Exception as e:
            logger.error(f"❌ Intent detection error: {e}", exc_info=True)
            fallback = self._get_cached_intent(question) or "general"
            logger.info(f"🔍 Error fallback route: {fallback}")
            return fallback
    
    async def generate_out_of_scope_response(self, question: str, user_role: str) -> str:
        """Generate brief out-of-scope response"""
        try:
            logger.info(f"🚫 Generating out-of-scope response for role: {user_role}")
            prompt = OUT_OF_SCOPE_SYSTEM_PROMPT.format(
                role=user_role,
                question=question
            )
            
            response = await asyncio.wait_for(
                self.response_llm.ainvoke(prompt),
                timeout=10.0
            )
            
            generated = response.content.strip()
            logger.info(f"✅ Out-of-scope response generated: {generated[:100]}")
            return generated
            
        except asyncio.TimeoutError:
            logger.warning("⏱️ Out-of-scope response timeout")
            return f"I'm your GoodBooks ERP assistant. I specialize in helping with GoodBooks features and functionality. Could you please ask me about something related to GoodBooks ERP?"
        except Exception as e:
            logger.error(f"❌ Out-of-scope response error: {e}")
            return f"I'm here to help with GoodBooks ERP. What would you like to know about our system?"
    
    async def apply_role_perspective(self, answer: str, user_role: str, question: str) -> str:
        """Improved role adaptation"""
        try:
            greeting_words = ['hello', 'hi there', 'welcome', 'greetings', "i'm here to help"]
            if any(word in answer.lower() for word in greeting_words) and len(answer) < 200:
                logger.info("⚡ Skipping role adaptation - greeting detected")
                return answer
            
            error_phrases = ['error', 'try again', 'something went wrong', "couldn't", "unable to"]
            if any(phrase in answer.lower() for phrase in error_phrases):
                logger.info("⚡ Skipping role adaptation - error message")
                return answer
            
            if len(answer.strip()) < 30:
                logger.info("⚡ Skipping role adaptation - answer too short")
                return answer
            
            logger.info(f"🎭 Applying {user_role} perspective to answer...")
            
            role_personality = ROLE_SYSTEM_PROMPTS.get(user_role, ROLE_SYSTEM_PROMPTS[UserRole.CLIENT])
            
            prompt = f"""{role_personality}

Original Answer: {answer}

User Question: {question}

Task: Rewrite this answer to match the {user_role} perspective while keeping all facts accurate. 
Adjust the tone, terminology, and level of detail to be appropriate for someone in the {user_role} role.

Rewritten Answer:"""
            
            response = await asyncio.wait_for(
                self.response_llm.ainvoke(prompt),
                timeout=15.0
            )
            
            role_adapted = response.content.strip()
            
            if role_adapted and len(role_adapted) > 20:
                logger.info(f"✅ Role perspective applied successfully ({len(role_adapted)} chars)")
                return role_adapted
            else:
                logger.warning("⚠️ Role adaptation produced insufficient result, using original")
                return answer
            
        except asyncio.TimeoutError:
            logger.warning("⏱️ Role adaptation timeout, using original answer")
            return answer
        except Exception as e:
            logger.error(f"❌ Role perspective error: {e}")
            return answer
    
    async def process_request(self, username: str, user_role: str, question: str,
                            thread_id: str = None, is_existing_thread: bool = False) -> Dict[str, str]:
        """Enhanced request processing with comprehensive fallback chain"""

        start_time = time.time()
        logger.info("="*80)
        logger.info(f"🚀 NEW REQUEST from {username} (Role: {user_role})")
        logger.info(f"💬 Question: {question}")
        logger.info("="*80)

        # IMPROVED: Only prompt for role when it's a NEW thread with no role set
        # Don't check role on every request for existing threads
        if thread_id and not is_existing_thread:
            thread = history_manager.get_thread(thread_id)
            current_role = thread.user_role if thread else None

            # Only prompt for role if this is a truly new thread with no role
            if not current_role:
                # Parse if user provided name and role
                parsed_name, parsed_role = parse_name_and_role(question)
                if parsed_role:
                    # User provided role, set it in thread
                    user_role = parsed_role
                    user_name = parsed_name
                    if thread:
                        thread.user_role = user_role
                        thread.user_name = user_name
                        history_manager.save_threads()
                    logger.info(f"🎭 User set role to: {user_role}, name: {user_name} for thread {thread_id}")
                    confirmation = f"Hello {user_name if user_name else username}! I've set your role to {user_role}. How can I help you with GoodBooks ERP today?"
                    return {
                        "response": confirmation,
                        "bot_type": "role_setup",
                        "thread_id": thread_id,
                        "user_role": user_role
                    }
                else:
                    # Ask for name and role ONLY for new threads
                    prompt = """Hello! I'm your GoodBooks ERP assistant. To provide you with the best help, please tell me your name and role.

Please reply in this format: "Name: [Your Name], Role: [your role]"

Available roles: developer, implementation, marketing, client, admin, system admin, manager, sales

For example: "Name: John, Role: developer" """

                    return {
                        "response": prompt,
                        "bot_type": "role_prompt",
                        "thread_id": thread_id,
                        "user_role": "client"  # Default until set
                    }

        # For existing threads, use the stored role or fallback to provided role
        if thread_id and is_existing_thread:
            thread = history_manager.get_thread(thread_id)
            if thread:
                # ✅ IMPROVED: Better role persistence and retrieval
                if thread.user_role:
                    user_role = thread.user_role
                    logger.info(f"✅ Using persisted role from thread {thread_id}: {user_role}")
                else:
                    # Thread exists but no role set - this shouldn't happen for existing threads
                    logger.warning(f"⚠️ Existing thread {thread_id} has no role set - using login role")
                    user_role = user_role  # Use the provided role
                    thread.user_role = user_role  # Set it for future use
                    history_manager.save_threads()
            else:
                # Thread ID provided but thread not found - create new thread
                logger.warning(f"⚠️ Thread {thread_id} not found, creating new thread")
                thread_id = history_manager.create_new_thread(username, question)
                is_existing_thread = False

        if is_greeting(question):
            logger.info(f"⚡ INSTANT greeting response (0.0s)")
            greeting_response = get_greeting_response(user_role)

            # Skip all slow operations for greetings - do them in background
            asyncio.create_task(
                asyncio.to_thread(
                    enhanced_memory.store_conversation_turn,
                    username, question, greeting_response, "greeting", user_role, thread_id
                )
            )

            if thread_id:
                asyncio.create_task(
                    asyncio.to_thread(
                        history_manager.add_message_to_thread,
                        thread_id, question, greeting_response, "greeting"
                    )
                )

            # Return immediately without waiting for background tasks
            return {
                "response": greeting_response,
                "bot_type": "greeting",
                "thread_id": thread_id,
                "user_role": user_role
            }
        
        logger.info("📚 Building conversational context...")
        if is_existing_thread and thread_id:
            recent_memories = enhanced_memory.retrieve_contextual_memories(
                username, question, k=3, thread_id=thread_id, thread_isolation=True
            )
            context = build_conversational_context(username, question, thread_id, thread_isolation=True)
        else:
            recent_memories = enhanced_memory.retrieve_contextual_memories(
                username, question, k=3, thread_id=thread_id, thread_isolation=False
            )
            context = build_conversational_context(username, question, thread_id, thread_isolation=False)

        logger.info(f"📚 Retrieved {len(recent_memories)} contextual memories")

        logger.info("🎯 Detecting intent...")
        intent = await self.detect_intent_with_ai(question, context)
        logger.info(f"🎯 INTENT SELECTED: {intent}")

        # 🔗 ENHANCED: Add cross-bot context sharing
        cross_bot_contexts = shared_context_registry.get_relevant_contexts(username, intent, question)
        if cross_bot_contexts:
            context_parts = [context] if context else []

            context_parts.append("=== Cross-Bot Context (Related Information) ===")
            for ctx in cross_bot_contexts[:2]:  # Limit to top 2 most relevant
                bot_type = ctx['bot_type']
                context_key = ctx['context_key']
                context_parts.append(f"From {bot_type} bot - {context_key}:")
                # Include key context data, but limit length
                ctx_data = ctx['context_data']
                relevant_info = []
                for key, value in ctx_data.items():
                    if not key.startswith('_') and isinstance(value, (str, int, float)):
                        str_value = str(value)[:100]  # Limit value length
                        relevant_info.append(f"  {key}: {str_value}")
                context_parts.extend(relevant_info[:3])  # Limit to 3 key-value pairs
                context_parts.append("")

            context = "\n".join(context_parts)
            logger.info(f"🔗 Added {len(cross_bot_contexts)} cross-bot contexts")
        
        selected_bot = self.bots.get(intent, self.bots["general"])
        answer = None
        bot_type = intent
        
        logger.info(f"🤖 Executing {intent} bot...")
        try:
            answer = await asyncio.wait_for(
                selected_bot.answer(question, context, user_role, username),
                timeout=40.0
            )
            logger.info(f"📥 {intent} bot response received: {len(answer) if answer else 0} chars")
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Bot {intent} execution timeout (40s)")
            answer = None
        except Exception as e:
            logger.error(f"❌ Bot {intent} execution error: {e}", exc_info=True)
            answer = None
        
        if not answer or len(answer.strip()) < 10:
            logger.warning(f"⚠️ Primary bot '{intent}' returned insufficient answer (len={len(answer) if answer else 0})")
            
            if intent != "general":
                logger.info("🔄 Attempting fallback to general bot...")
                try:
                    answer = await asyncio.wait_for(
                        self.bots["general"].answer(question, context, user_role, username),
                        timeout=40.0
                    )
                    if answer and len(answer.strip()) >= 10:
                        logger.info(f"✅ General bot fallback successful: {len(answer)} chars")
                        bot_type = "general_fallback"
                    else:
                        logger.warning("⚠️ General bot fallback also returned insufficient answer")
                        answer = None
                except asyncio.TimeoutError:
                    logger.error("⏱️ General bot fallback timeout")
                    answer = None
                except Exception as e:
                    logger.error(f"❌ General bot fallback error: {e}", exc_info=True)
                    answer = None
        
        if not answer or len(answer.strip()) < 10:
            logger.info(f"🚫 No valid answer from any bot, generating out-of-scope response")
            answer = await self.generate_out_of_scope_response(question, user_role)
            bot_type = "out_of_scope"
        else:
            logger.info(f"⚡ Skipping redundant role adaptation - bot handled it in-situ")
            # answer = await self.apply_role_perspective(answer, user_role, question)
        
        logger.info("💾 Storing conversation in background...")
        asyncio.create_task(
            asyncio.to_thread(
                update_enhanced_memory,
                username, question, answer, bot_type, user_role, thread_id
            )
        )
        
        elapsed = time.time() - start_time
        logger.info("="*80)
        logger.info(f"✅ REQUEST COMPLETED in {elapsed:.2f}s")
        logger.info(f"🤖 Bot Type: {bot_type}")
        logger.info(f"📏 Response Length: {len(answer)} chars")
        logger.info(f"👤 User Role: {user_role}")
        logger.info("="*80)
        
        return {
            "response": answer,
            "bot_type": bot_type,
            "thread_id": thread_id,
            "user_role": user_role
        }

# Initialize as None - will be loaded at startup
ai_orchestrator = None

# ===========================
# Helper Functions
# ===========================
def parse_name_and_role(message: str) -> tuple[str, str]:
    """
    Parse name and role from user message with flexible format support.
    Supports multiple formats:
    - "Name: John, Role: developer"
    - "name: john role: developer"
    - "john developer"
    - "Name John Role Developer"
    - "My name is John and I'm a developer"
    """
    name = None
    role = None

    message_lower = message.lower().strip()
    
    # Quick exit for questions - don't parse roles from actual questions
    question_words = ["how", "what", "can", "could", "why", "when", "where", "show", "tell", "list", "is", "are", "who", "which", "do", "does", "will", "would", "should"]
    if any(message_lower.startswith(word) for word in question_words) or "?" in message_lower:
        return None, None
    
    # Don't parse role from long sentences that aren't explicit introductions
    if len(message.split()) > 10 and not ("name:" in message_lower or "role:" in message_lower):
        return None, None
    
    # Valid roles for validation
    valid_roles = ["developer", "implementation", "marketing", "client", "admin", "system admin", "manager", "sales"]

    # Strategy 1: Explicit key-value format (name: X, role: Y)
    name_match = re.search(r'name\s*[:=]\s*([a-zA-Z\s]+?)(?:,|role|$)', message, re.IGNORECASE)
    role_match = re.search(r'role\s*[:=]\s*([a-zA-Z\s]+?)(?:,|$)', message, re.IGNORECASE)
    
    if name_match:
        name = name_match.group(1).strip().split()[0]  # Take first word
    if role_match:
        role_candidate = role_match.group(1).strip().lower()
        # Handle "system admin" as special case
        if "system" in role_candidate and "admin" in role_candidate:
            role = "system admin"
        else:
            # Match first valid role word
            for valid_role in valid_roles:
                if valid_role in role_candidate:
                    role = valid_role
                    break
    
    # Strategy 2: Natural language format (My name is X, I'm a Y)
    if not name or not role:
        name_patterns = [
            r"(?:my\s+)?name\s+(?:is\s+)?([a-zA-Z]+)",
            r"i'm?\s+([a-zA-Z]+)\s+(?:and|,)?",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match and not name:
                candidate = match.group(1).strip()
                if candidate.lower() not in ["a", "the", "and", "or"] and len(candidate) > 1:
                    name = candidate
    
    # Strategy 3: Removed - too broad, was causing false positives in normal questions
    
    # Strategy 4: Simple space-separated format (Name Role) - Extract name even if role found
    if not name:  # ✅ FIX: Only check for name, role may already be found
        words = message.split()
        if len(words) >= 2:
            # Check if any word is a valid role to identify name position
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:')
                if word_lower in valid_roles:
                    # Found role at position i, take first word as name
                    name = words[0]
                    if not role:  # Only set role if not already found
                        role = word_lower
                    break
            else:
                # No role found in words, try first word as name and second as role
                if len(words) >= 2:
                    potential_role = words[1].lower().strip('.,!?;:')
                    if potential_role in valid_roles:
                        name = words[0]
                        if not role:
                            role = potential_role
    
    # Validate and clean up
    if name:
        name = name.strip().capitalize()
    
    if role:
        role = role.strip().lower()
        if role not in valid_roles:
            role = None  # Invalid role
    
    logger.info(f"🔍 Parsed message: '{message}'")
    logger.info(f"   → Name: {name}, Role: {role}")
    
    return name, role

def update_user_session(username: str, name: str = None, current_role: str = None):
    """Update user session - now non-blocking"""
    try:
        current_time = datetime.now().isoformat()

        session_ref = db.collection('user_sessions').document(username)
        session_doc = session_ref.get()

        if not session_doc.exists:
            user_session_data = {
                "first_seen": current_time,
                "last_activity": current_time,
                "session_count": 1,
                "total_interactions": 1,
                "name": name,
                "current_role": current_role
            }
        else:
            user_session_data = session_doc.to_dict()
            user_session_data["last_activity"] = current_time
            user_session_data["total_interactions"] = user_session_data.get("total_interactions", 0) + 1
            if name is not None:
                user_session_data["name"] = name
            if current_role is not None:
                user_session_data["current_role"] = current_role

        session_ref.set(user_session_data)
        user_sessions[username] = user_session_data
    except Exception as e:
        logger.error(f"Error saving user session: {e}")

def update_enhanced_memory(username: str, question: str, answer: str, bot_type: str, user_role: str, thread_id: str = None):
    """Update memory - runs in background thread"""
    try:
        if thread_id:
            history_manager.add_message_to_thread(thread_id, question, answer, bot_type)
        
        enhanced_memory.store_conversation_turn(username, question, answer, bot_type, user_role, thread_id)
        logger.info("💾 Memory stored successfully")
    except Exception as e:
        logger.error(f"Error storing memory: {e}")

# ===========================
# FASTAPI APP INITIALIZATION
# ===========================
app = FastAPI(title="GoodBooks AI-Powered Role-Based ERP Assistant - FIXED")

app.add_middleware(PerformanceMonitoringMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Pydantic Models
# ===========================
class Message(BaseModel):
    content: str

class ThreadRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str

class ThreadRenameRequest(BaseModel):
    thread_id: str
    new_title: str

# ===========================
# API ENDPOINTS
# ===========================
@app.post("/gbaiapi/chat", tags=["AI Role-Based Chat"])
async def ai_role_based_chat(message: Message, Login: str = Header(...)):
    """AI-powered role-based chat - NEW CONVERSATION with improved thread handling"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        # ✅ FIX: Always get user_role from login_dto, not from session
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header. Must include UserName"})

    user_input = message.content.strip()

    if not user_input:
        return JSONResponse(status_code=400, content={"response": "Please provide a message"})

    try:
        # Use login_dto as session key to maintain session state across requests
        login_dto_str = json.dumps(login_dto, sort_keys=True)
        session_info = user_sessions.get(login_dto_str, {})
        # ✅ FIX: Remove role from session check - always use from login_dto
        is_registered = session_info.get("registered", False)
        
        # ✅ FIX: ALWAYS create thread for first message
        thread_id = await asyncio.to_thread(history_manager.create_new_thread, username, user_input)
        logger.info(f"📍 Created new thread: {thread_id}")
        
        thread = history_manager.get_thread(thread_id)
        
        # ✅ FIX: Check if current request is role setup or actual question
        parsed_name, parsed_role = parse_name_and_role(user_input)
        
        if parsed_role:
            # User provided both name and role in this message
            user_name = parsed_name or username
            user_role = parsed_role
            
            # ✅ FIX: Save role info to thread IMMEDIATELY
            thread.user_role = user_role
            thread.user_name = user_name
            history_manager.save_threads()
            
            logger.info(f"🎭 User set role to: {user_role}, name: {user_name} in thread {thread_id}")
            
            confirmation = f"Perfect! I've set your profile:\n• Name: {user_name}\n• Role: {user_role}\n\nNow, how can I help you with GoodBooks ERP today?"
            
            # ✅ FIX: Store this setup message in thread
            await asyncio.to_thread(
                history_manager.add_message_to_thread,
                thread_id, user_input, confirmation, "role_setup"
            )
            
            # Update session
            session_info["registered"] = True
            session_info["current_role"] = user_role
            session_info["user_name"] = user_name
            session_info["last_thread_id"] = thread_id
            user_sessions[login_dto_str] = session_info
            
            return {
                "response": confirmation,
                "bot_type": "role_setup",
                "thread_id": thread_id,
                "user_role": user_role,
                "user_name": user_name,
                "sources_used": {
                    "sources_count": 0,
                    "sources": []
                }
            }
        
        # ✅ FIX: Check if user is already registered
        if not is_registered or user_role == "unknown":
            # Ask for name and role with helpful examples
            prompt = """Hello! I'm your GoodBooks ERP assistant. 

To give you the best help, I need to know your name and role. 

"""
            
            # Store role request in thread
            await asyncio.to_thread(
                history_manager.add_message_to_thread,
                thread_id, user_input, prompt, "role_prompt"
            )
            
            return {
                "response": prompt,
                "bot_type": "role_prompt",
                "thread_id": thread_id,
                "user_role": "client",
                "sources_used": {
                    "sources_count": 0,
                    "sources": []
                }
            }
        
        # ✅ FIX: User is registered and has role, process as normal query
        logger.info(f"✅ User {username} is registered as {user_role}")
        
        # Set thread role if not already set
        if not thread.user_role:
            thread.user_role = user_role
            history_manager.save_threads()
        
        # Check if it's a greeting
        if is_greeting(user_input):
            logger.info(f"⚡ INSTANT greeting response (0.0s)")
            greeting_response = get_greeting_response(user_role)
            
            asyncio.create_task(
                asyncio.to_thread(
                    history_manager.add_message_to_thread,
                    thread_id, user_input, greeting_response, "greeting"
                )
            )
            
            return {
                "response": greeting_response,
                "bot_type": "greeting",
                "thread_id": thread_id,
                "user_role": user_role,
                "sources_used": {
                    "sources_count": 0,
                    "sources": []
                }
            }
        
        # Build filtered context
        logger.info("📚 Building conversational context...")
        context, source_files = await build_filtered_context(
            username, user_input, thread_id=thread_id, is_existing_thread=False
        )
        
        logger.info(f"📍 Built filtered context from {len(source_files)} source(s)")
        
        # Process request
        result = await ai_orchestrator.process_request(username, user_role, user_input, thread_id, is_existing_thread=False)
        
        # Add actual source files used
        formatted_sources = source_tracker.format_sources_for_response(source_files)
        result["sources_used"] = formatted_sources
        result["thread_id"] = thread_id
        
        logger.info(f"✅ Response sent using sources: {source_files}")
        logger.info(f"✅ Chat response sent to {username} ({user_role})")
        return result
        
    except Exception as e:
        logger.error(f"❌ AI orchestration error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        error_response = "I encountered an error processing your request. Please try again or rephrase your question."
        return JSONResponse(
            status_code=500,
            content={"response": error_response, "bot_type": "error", "sources_used": {"sources_count": 0, "sources": []}}
        )

@app.post("/gbaiapi/thread_chat", tags=["AI Thread Chat"])
async def ai_thread_chat(request: ThreadRequest, Login: str = Header(...)):
    """Continue conversation in existing thread with role continuity"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        # Get default role from login_dto
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})

    thread_id = request.thread_id
    user_input = request.message.strip()

    if not user_input:
        return JSONResponse(status_code=400, content={"response": "Please provide a message"})

    if not thread_id:
        return JSONResponse(status_code=400, content={"response": "Thread ID is required"})

    # Use login_dto as session key for consistency
    login_dto_str = json.dumps(login_dto, sort_keys=True)
    session_info = user_sessions.get(login_dto_str, {})

    # Verify thread exists and belongs to user
    thread = history_manager.get_thread(thread_id)
    if not thread or thread.username != username:
        return JSONResponse(status_code=404, content={"response": "Thread not found or access denied"})
    
    # ✅ FIX: Prioritize thread's persisted role if it exists
    if thread.user_role and thread.user_role != "unknown":
        user_role = thread.user_role
        logger.info(f"🎭 Using persisted role from thread: {user_role}")

    # ✅ FIX: Check if still waiting for role setup
    if not thread.user_role or thread.user_role == "unknown":
        # Still waiting for role info
        parsed_name, parsed_role = parse_name_and_role(user_input)
        
        if parsed_role:
            # User provided role now
            user_name = parsed_name or username
            user_role = parsed_role
            
            # ✅ FIX: Save to thread
            thread.user_role = user_role
            thread.user_name = user_name
            history_manager.save_threads()
            
            logger.info(f"🎭 User set role to: {user_role}, name: {user_name} in thread {thread_id}")
            
            confirmation = f"Perfect! I've set your profile:\n• Name: {user_name}\n• Role: {user_role}\n\nNow, how can I help you with GoodBooks ERP today?"
            
            # Store in thread
            await asyncio.to_thread(
                history_manager.add_message_to_thread,
                thread_id, user_input, confirmation, "role_setup"
            )
            
            # Update session
            session_info["registered"] = True
            session_info["current_role"] = user_role
            session_info["user_name"] = user_name
            user_sessions[login_dto_str] = session_info
            
            return {
                "response": confirmation,
                "bot_type": "role_setup",
                "thread_id": thread_id,
                "user_role": user_role,
                "user_name": user_name,
                "sources_used": {
                    "sources_count": 0,
                    "sources": []
                }
            }
        else:
            # Still can't parse role, ask again with better format hints
            prompt = """I couldn't understand your role. Could you please provide it in one of these clearer formats?

**Format examples:**
1. "Name: Ashok, Role: Developer"
2. "Ashok, developer"
3. "I'm Ashok and I'm a developer"
4. "My name is Ashok, role is developer"

**Available roles:** Developer, Implementation, Marketing, Client, Admin, System Admin, Manager, Sales"""
            
            await asyncio.to_thread(
                history_manager.add_message_to_thread,
                thread_id, user_input, prompt, "role_prompt"
            )
            
            return {
                "response": prompt,
                "bot_type": "role_prompt",
                "thread_id": thread_id,
                "user_role": "client",
                "sources_used": {
                    "sources_count": 0,
                    "sources": []
                }
            }
    
    # ✅ FIX: User has role, process normally
    try:
        logger.info(f"📍 Continuing thread {thread_id} with role: {user_role}")
        
        # Build filtered context FOR EXISTING THREAD
        context, source_files = await build_filtered_context(
            username, user_input, thread_id=thread_id, is_existing_thread=True
        )
        
        logger.info(f"📍 Built filtered context from {len(source_files)} source(s)")
        
        result = await ai_orchestrator.process_request(
            username, user_role, user_input, thread_id, is_existing_thread=True
        )
        
        # Add actual source files used
        formatted_sources = source_tracker.format_sources_for_response(source_files)
        result["sources_used"] = formatted_sources
        result["thread_id"] = thread_id
        
        logger.info(f"✅ Response sent using sources: {source_files}")
        logger.info(f"✅ Thread response sent to {username} ({user_role})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Thread chat error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_response = "I encountered an error. Please try again."
        return JSONResponse(
            status_code=500,
            content={"response": error_response, "bot_type": "error", "thread_id": thread_id, "sources_used": {"sources_count": 0, "sources": []}}
        )

@app.get("/gbaiapi/conversation_threads", tags=["Conversation History"])
async def get_conversation_threads(Login: str = Header(...), limit: int = 50):
    """Get user's conversation threads"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        
        # ✅ Map roleid to role name if present, otherwise fallback to "Role"
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    threads = history_manager.get_user_threads(username, limit)
    login_dto_str = json.dumps(login_dto, sort_keys=True)
    session_info = user_sessions.get(login_dto_str, {})
    
    return {
        "username": username,
        "user_role": user_role,
        "session_info": session_info,
        "threads": threads,
        "total_threads": len(threads)
    }

@app.get("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def get_thread_details(thread_id: str, Login: str = Header(...)):
    """Get thread details"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    thread = history_manager.get_thread(thread_id)
    
    if not thread or thread.username != username:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})
    
    return thread.to_dict()

@app.delete("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def delete_thread(thread_id: str, Login: str = Header(...)):
    """Delete a thread"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    success = history_manager.delete_thread(thread_id, username)
    
    if success:
        return {"message": "Thread deleted successfully"}
    else:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})

@app.put("/gbaiapi/thread/{thread_id}/rename", tags=["Conversation History"])
async def rename_thread(thread_id: str, request: ThreadRenameRequest, Login: str = Header(...)):
    """Rename a thread"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    success = history_manager.rename_thread(thread_id, username, request.new_title)
    
    if success:
        return {"message": "Thread renamed successfully"}
    else:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})

@app.get("/gbaiapi/available_roles", tags=["Role Information"])
async def get_available_roles():
    """Get available user roles and their descriptions"""
    return {
        "available_roles": [
            {
                "role": "developer",
                "display_name": "Developer",
                "description": "Technical expert who understands code, APIs, and system architecture",
                "response_style": "Technical, detailed, with code examples and implementation details"
            },
            {
                "role": "implementation",
                "display_name": "Implementation Consultant",
                "description": "Team member who deploys and configures the system for clients",
                "response_style": "Step-by-step instructions, configuration guidance, best practices"
            },
            {
                "role": "marketing",
                "display_name": "Marketing/Sales",
                "description": "Team member focused on selling and promoting the solution",
                "response_style": "Business benefits, ROI, competitive advantages, persuasive"
            },
            {
                "role": "client",
                "display_name": "Client/End User",
                "description": "End user who uses the system for daily work",
                "response_style": "Simple, friendly, non-technical, easy to understand"
            },
            {
                "role": "admin",
                "display_name": "System Administrator",
                "description": "Administrator with full system access and knowledge",
                "response_style": "Comprehensive, covering all technical and business aspects"
            }
        ],
        "default_role": "client",
        "note": "Role must be selected during login and passed in the Login header as 'Role' field"
    }

@app.get("/gbaiapi/system_status", tags=["System Health"])
async def system_status():
    """System health check"""
    bot_status = {
        "general": "available" if GENERAL_BOT_AVAILABLE else "unavailable",
        "formula": "available" if FORMULA_BOT_AVAILABLE else "unavailable",
        "report": "available" if REPORT_BOT_AVAILABLE else "unavailable",
        "menu": "available" if MENU_BOT_AVAILABLE else "unavailable",
        "project": "available" if PROJECT_BOT_AVAILABLE else "unavailable"
    }
    
    memory_stats = {
        "total_users": len(chats_db),
        "total_sessions": len(user_sessions),
        "total_conversations": sum(len(chats) for chats in chats_db.values()),
        "total_memories": len(conversational_memory_metadata),
        "total_threads": len(history_manager.threads),
        "active_threads": len([t for t in history_manager.threads.values() if t.is_active])
    }
    
    return {
        "status": "healthy",
        "version": "8.0.0-FIXED-ENHANCED",
        "available_bots": [k for k, v in bot_status.items() if v == "available"],
        "bot_status": bot_status,
        "memory_system": memory_stats,
        "features": [
            "⚡ INSTANT greeting responses (<1s)",
            "🚀 Enhanced keyword-based fast routing with math detection",
            "🎯 Improved AI intent detection with 10s timeout",
            "🔄 Comprehensive fallback chain (Primary → General → Out-of-scope)",
            "🎭 Smart role adaptation (skips only when appropriate)",
            "⏱️ Increased timeouts for all LLM operations",
            "📝 Enhanced logging throughout entire pipeline",
            "🔍 Intelligent fallback based on question structure",
            "💾 Background memory storage (non-blocking)",
            "🧠 Context-aware routing decisions"
        ],
        "performance": {
            "greeting_response": "<1 second",
            "simple_query_with_fast_route": "5-10 seconds",
            "simple_query_with_ai_route": "10-15 seconds",
            "complex_query": "20-30 seconds",
            "keyword_routing": "Instant (no LLM)",
            "intent_detection_timeout": "10 seconds",
            "bot_execution_timeout": "40 seconds",
            "role_adaptation_timeout": "15 seconds"
        },
        "optimizations": [
            "✅ Enhanced keyword detection with math pattern recognition",
            "✅ Intent caching system",
            "✅ Smart role adaptation (only when needed)",
            "✅ Comprehensive fallback chain",
            "✅ Parallel async execution",
            "✅ Background memory storage",
            "✅ Increased timeouts for stability",
            "✅ Better error handling and logging",
            "✅ Question structure analysis for fallbacks",
            "✅ Sub-bot availability checking",
            "✅ Detailed performance tracking"
        ]
    }

@app.get("/gbaiapi/user_statistics", tags=["Analytics"])
async def get_user_statistics(Login: str = Header(...)):
    """Get user statistics"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        
        # ✅ Map roleid to role name if present, otherwise fallback to "Role"
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    session_info = user_sessions.get(username, {})
    user_chats = chats_db.get(username, [])
    user_threads = history_manager.get_user_threads(username)
    
    bot_usage = {}
    for chat in user_chats:
        bot_type = chat.get('bot_type', 'unknown')
        bot_usage[bot_type] = bot_usage.get(bot_type, 0) + 1
    
    now = datetime.now()
    recent_activity = {"today": 0, "this_week": 0, "this_month": 0}
    
    for chat in user_chats:
        try:
            chat_time = datetime.fromisoformat(chat['timestamp'])
            time_diff = now - chat_time
            
            if time_diff.days == 0:
                recent_activity["today"] += 1
            if time_diff.days <= 7:
                recent_activity["this_week"] += 1
            if time_diff.days <= 30:
                recent_activity["this_month"] += 1
        except:
            continue
    
    return {
        "username": username,
        "user_role": user_role,
        "session_info": session_info,
        "statistics": {
            "total_conversations": len(user_chats),
            "total_threads": len(user_threads),
            "active_threads": len([t for t in user_threads if t.get('is_active', True)]),
            "bot_usage": bot_usage,
            "recent_activity": recent_activity
        }
    }

@app.post("/gbaiapi/cleanup_old_data", tags=["System Maintenance"])
async def cleanup_old_data(Login: str = Header(...), days_to_keep: int = 90):
    """Cleanup old data (admin only)"""
    try:
        login_dto = json.loads(Login)
        
        # ✅ Map roleid to role name if present, otherwise fallback to "Role"
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
        
        if user_role != "admin":
            return JSONResponse(status_code=403, content={"response": "Admin access required"})
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    try:
        await asyncio.to_thread(history_manager.cleanup_old_threads, days_to_keep)
        
        return {
            "message": f"Cleaned up data older than {days_to_keep} days",
            "cleanup_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return JSONResponse(status_code=500, content={"response": "Cleanup failed"})

@app.get("/gbaiapi/performance_stats", tags=["System Health"])
async def get_performance_stats():
    """Get performance statistics"""
    return {
        "cache_stats": {
            "intent_cache_size": len(ai_orchestrator.intent_cache),
            "cached_intents": list(ai_orchestrator.intent_cache.keys())[:20]
        },
        "optimization_status": {
            "keyword_routing": "enabled",
            "intent_caching": "enabled",
            "smart_role_adaptation": "enabled",
            "fallback_chain": "enabled",
            "background_memory_storage": "enabled",
            "async_processing": "enabled"
        },
        "timeout_configuration": {
            "greeting_detection": "instant",
            "intent_detection": "10 seconds",
            "bot_execution": "40 seconds",
            "role_adaptation": "15 seconds",
            "out_of_scope_generation": "10 seconds"
        },
        "routing_strategy": {
            "primary": "Keyword-based fast routing",
            "secondary": "AI-based intent detection",
            "fallback": "Question structure analysis",
            "bot_chain": "Primary bot → General bot → Out-of-scope"
        }
    }

@app.get("/gbaiapi/debug/test_bot/{bot_name}", tags=["Debug"])
async def test_bot(bot_name: str, question: str = "What is GoodBooks?", Login: str = Header(...)):
    """Test a specific bot directly (for debugging)"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        
        # ✅ Map roleid to role name if present, otherwise fallback to "Role"
        role_id = str(login_dto.get("roleid", ""))
        user_role = ROLEID_TO_NAME.get(role_id, login_dto.get("Role", "client")).lower()
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    if bot_name not in ai_orchestrator.bots:
        return JSONResponse(status_code=404, content={"response": f"Bot '{bot_name}' not found. Available: {list(ai_orchestrator.bots.keys())}"})
    
    try:
        logger.info(f"🧪 Testing {bot_name} bot with question: {question}")
        start_time = time.time()
        
        selected_bot = ai_orchestrator.bots[bot_name]
        answer = await asyncio.wait_for(
            selected_bot.answer(question, "", user_role, username),
            timeout=40.0
        )
        
        elapsed = time.time() - start_time
        
        return {
            "bot_name": bot_name,
            "question": question,
            "answer": answer,
            "answer_length": len(answer) if answer else 0,
            "execution_time": f"{elapsed:.2f}s",
            "success": bool(answer and len(answer) > 10)
        }
    except asyncio.TimeoutError:
        return JSONResponse(status_code=500, content={
            "bot_name": bot_name,
            "error": "Bot execution timeout (40s)",
            "question": question
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "bot_name": bot_name,
            "error": str(e),
            "question": question,
            "traceback": traceback.format_exc()
        })

@app.get("/gbaiapi/debug/test_routing", tags=["Debug"])
async def test_routing(question: str):
    """Test intent routing without executing bot (for debugging)"""
    try:
        logger.info(f"🧪 Testing routing for question: {question}")
        
        keyword_intent = ai_orchestrator._get_cached_intent(question)
        
        start_time = time.time()
        ai_intent = await ai_orchestrator.detect_intent_with_ai(question, "")
        elapsed = time.time() - start_time
        
        return {
            "question": question,
            "keyword_based_intent": keyword_intent or "none (will use AI)",
            "ai_detected_intent": ai_intent,
            "routing_time": f"{elapsed:.2f}s",
            "routing_method": "keyword" if keyword_intent else "ai",
            "available_bots": list(ai_orchestrator.bots.keys())
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.get("/gbaiapi/debug/clear_cache", tags=["Debug"])
async def clear_intent_cache():
    """Clear intent cache (for debugging)"""
    cache_size = len(ai_orchestrator.intent_cache)
    ai_orchestrator.intent_cache.clear()
    return {
        "message": "Intent cache cleared",
        "previous_cache_size": cache_size,
        "current_cache_size": len(ai_orchestrator.intent_cache)
    }

@app.get("/health", tags=["System Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "8.0.0-FIXED-ENHANCED"
    }

# ===========================
# STARTUP/SHUTDOWN EVENTS
# ===========================
@app.on_event("startup")
async def startup_event():
    global history_manager, embeddings, enhanced_memory, ai_orchestrator
    
    logger.info("=" * 80)
    logger.info("🚀 GoodBooks AI Orchestrator starting")
    logger.info("=" * 80)

    try:
        # --------------------------------------------------
        # 🔥 INITIALIZE HEAVY COMPONENTS FIRST
        # --------------------------------------------------
        logger.info("📦 Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'batch_size': 1}
        )
        logger.info("✅ Embeddings loaded")
        
        logger.info("💾 Loading conversation memory from GCS...")
        enhanced_memory = EnhancedConversationalMemory(MEMORY_VECTORSTORE_PATH, "metadata.json", embeddings)
        logger.info("✅ Memory loaded")
        
        logger.info("📚 Loading conversation threads from Firestore...")
        history_manager = ConversationHistoryManager()
        logger.info("✅ Threads loaded")
        
        logger.info("🤖 Initializing AI orchestrator...")
        ai_orchestrator = AIOrchestrationAgent()
        logger.info("✅ Orchestrator ready")

        # --------------------------------------------------
        # 🔥 WARM OLLAMA MODELS (GPU LOAD)
        # --------------------------------------------------
        logger.info("🔥 Warming models in parallel...")
        await asyncio.gather(
            ai_orchestrator.routing_llm.ainvoke("hello"),
            ai_orchestrator.response_llm.ainvoke("Reply OK")
        )
        logger.info("✅ Ollama models warmed")

        # --------------------------------------------------
        # 📦 FORCE FAISS INTO MEMORY (if exists)
        # --------------------------------------------------
        if enhanced_memory and enhanced_memory.memory_vectorstore:
            logger.info("📦 Warming FAISS index...")
            try:
                enhanced_memory.memory_vectorstore.similarity_search("warmup", k=1)
                logger.info("✅ FAISS warmed")
            except Exception as e:
                logger.warning(f"⚠️ FAISS warming failed: {e}")

        # --------------------------------------------------
        # 🧪 REAL QUERY DRY RUN (MOST IMPORTANT)
        # --------------------------------------------------
        logger.info("🧪 Running real-query warmup...")
        await ai_orchestrator.process_request(
            username="__warmup__",
            user_role="client",
            question="What is GoodBooks ERP?",
            thread_id=None,
            is_existing_thread=False
        )
        logger.info("✅ Real-query warmup completed")

        # --------------------------------------------------
        # 🤖 PRE-WARM SUB BOTS (SAFE)
        # --------------------------------------------------
        async def warm_bot(bot, name):
            try:
                await asyncio.wait_for(
                    bot.answer("test", "", "client", "__warmup__"),
                    timeout=10
                )
                logger.info(f"🔥 {name} bot warmed")
            except Exception:
                logger.warning(f"⚠️ {name} bot warm skipped")

        # 🔥 Parallelize sub-bot warmup for faster startup
        await asyncio.gather(
            warm_bot(GeneralBotWrapper(), "general"),
            warm_bot(FormulaBot(), "formula"),
            warm_bot(ReportBot(), "report"),
            warm_bot(MenuBot(), "menu"),
            warm_bot(ProjectBot(), "project")
        )

        # --------------------------------------------------
        # 🧹 BACKGROUND CLEANUP
        # --------------------------------------------------
        asyncio.create_task(
            asyncio.to_thread(history_manager.cleanup_old_threads, 180)
        )

        logger.info("=" * 80)
        logger.info("✅ ALL SYSTEMS READY - App startup complete")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {str(e)}", exc_info=True)
        raise
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("="*80)
    logger.info("🛑 Shutting down gracefully...")
    logger.info("="*80)
    try:
        await asyncio.to_thread(history_manager.save_threads)
        logger.info("✅ All thread data saved to Firestore")
        
        logger.info("💾 Saving memory vectorstore...")
        await asyncio.to_thread(enhanced_memory.memory_vectorstore.save_local, MEMORY_VECTORSTORE_PATH)
        logger.info("✅ Memory vectorstore saved")
        
    except Exception as e:
        logger.error(f"❌ Shutdown save error: {e}")
    
    logger.info("="*80)
    logger.info("👋 Shutdown complete")
    logger.info("="*80)

# ===========================
# MAIN ENTRY POINT
# ===========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8010))
    logger.info("="*80)
    logger.info(f"🚀 Starting FIXED & ENHANCED server on port {port}")
    logger.info("="*80)
    uvicorn.run(app, host="0.0.0.0", port=port)

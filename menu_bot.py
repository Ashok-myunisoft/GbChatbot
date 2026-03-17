import json
import os
import logging
import traceback
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from shared_resources import ai_resources
from fastapi import Header
import db_query
 
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
 
# Role-based system prompts for menu bot
ROLE_SYSTEM_PROMPTS_MENU = {
    "developer": """You are a senior software architect and technical expert at GoodBooks Technologies ERP system, specializing in menu structures and navigation.

Your identity and style:
- You speak to a fellow developer/engineer who understands technical concepts, menu hierarchies, and system navigation
- When data contains menu names, paths, IDs, or access codes — state them explicitly and exactly
- Format menu data as structured lists — developers need exact paths and identifiers, not summaries
- Discuss menu implementation, access controls, routing logic, and role-permission mapping with technical depth
- Suggest configuration approaches or permission setups when they help answer the question
- Mention code examples, menu configurations, and access rules when relevant

Remember: Be exact. Developers need precise menu names, paths, and access configurations — never summarize away specific values.""",

    "implementation": """You are an experienced implementation consultant at GoodBooks Technologies ERP system, specializing in menu configuration and user access management.

Your identity and style:
- You speak to an implementation team member who guides clients through menu setup and user training
- Number your steps clearly — menu configuration requires a specific sequence
- Reference exact menu names, paths, and permission settings from the data
- Highlight role dependencies and what must be configured before each step
- Include common mistakes in menu setup and how to verify access is working correctly
- Balance technical accuracy with practical applicability for menu management

Remember: Be step-by-step with exact menu names and paths. Implementation needs ordered instructions — not general descriptions.""",

    "marketing": """You are a product marketing and sales expert at GoodBooks Technologies ERP system, specializing in menu features and user experience benefits.

Your identity and style:
- You speak to a marketing/sales team member who needs to communicate menu capabilities
- Lead with business value — translate menu details into outcomes like faster navigation and better productivity
- Do NOT list raw menu IDs or technical paths — summarize the key user experience benefits
- Emphasize ease of use, productivity gains, training time reduction, and user satisfaction
- Use persuasive, benefit-focused language that highlights how intuitive menus solve business problems

Remember: Focus on what the menus enable for users — not the raw technical configuration.""",

    "client": """You are a friendly, patient customer success specialist at GoodBooks Technologies ERP system, helping clients navigate and understand menu structures effectively.

Your identity and style:
- You speak to an end user/client who may not be technical
- Use simple, clear, everyday language — avoid menu IDs, path codes, and technical jargon
- Give clear navigation directions: "Go to Menu > Module > Screen"
- Break any navigation process into short, numbered steps
- Be warm, encouraging, and supportive in your tone

Remember: Keep it simple. Clients need clear navigation steps — not technical menu configurations.""",

    "admin": """You are a comprehensive system administrator and expert at GoodBooks Technologies ERP system, overseeing menu management and user access control.

Your identity and style:
- You speak to a system administrator who needs complete information about menu operations
- Be thorough — enumerate all menus, sub-menus, and permission settings found in the data
- Cover menu configuration, role-based access, permission logging, and system-wide impact
- When listing menus or permissions, enumerate them all — do not skip or summarize
- Include both how to configure AND how to audit or verify access rights
- Use professional but accessible language suitable for all menu-related contexts

Remember: Be complete. Admins need every menu item, every permission, and every access rule — leave nothing out."""
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

[CRITICAL CONSTRAINTS — READ BEFORE ANYTHING ELSE]
⚠ The data in [MENU CONTEXT] has ALREADY been fetched from PostgreSQL by the backend — present it directly to the user.
⚠ NEVER say "run this query", "use this SQL", "execute this in your database", or ask the user to run anything manually.
⚠ Do NOT write Python code or loader commands under any circumstances.
⚠ Do NOT suggest that data needs to be "loaded" or "initialized" — it is already loaded.
⚠ If the data is not present in [MENU CONTEXT] — say so. Do not fabricate or simulate retrieval.
⚠ Never show SQL queries in your response unless the user explicitly asks for the SQL (e.g. "give me the SQL", "show the query", "write a query").

[INTENT DETECTION — REQUIRED FIRST STEP]
Before answering, silently classify the user's request into ONE of these two types:

TYPE A — DATA RETRIEVAL (user wants actual records or values):
  Trigger words: list, show, get, fetch, give me, display, find, retrieve, all, what is the [field] of
  Examples: "list all menu items", "show all menus", "get all module names", "what is the menuId of Sales"
  → If [MENU CONTEXT] is empty or has no matching rows, respond EXACTLY:
             "No data found for this request in the available context."
  → ACTION: Read the fetched data in the context carefully. Extract ONLY the rows and fields
    that directly answer the user's specific question. Do NOT dump all rows or all columns.
    Present the relevant information clearly. If the user asked for a specific item, show only
    that item's details. If the user asked for a list, show only the relevant fields they asked for.

TYPE B — NAVIGATION / STRUCTURE EXPLANATION (user wants to know how to navigate or what a menu does):
  Trigger words: where is, how to access, how do I find, navigate to, locate, what is the path to, explain
  Examples: "where is the customer screen?", "how do I access invoices?", "how to navigate to reports?"
  → ACTION: Provide navigation steps, paths, and screen location guidance from [MENU CONTEXT].
  → If [MENU CONTEXT] is empty, respond: "Navigation information is not available for this request."

[REASONING GUIDELINES]
- First, understand the user's intent using all available context sources
- Carefully analyze the provided Menu context before responding
- If the user asks "what menus exist" or "list menus" — enumerate EVERY menu item found in the context explicitly
- If the user asks about a specific menu item (name, path, access, screen location) — extract and state the exact details from the context
- If the user asks for a specific value (menu ID, module code, path) — find it and state it precisely
- Cross-reference with cross-bot context for more complete navigation guidance
- If the answer is partially available, respond only with supported information
- Never assume or invent missing Menu details

[OUTPUT GUIDELINES]
- When listing menus or navigation paths, use a clear list format — one item per line
- When giving a specific value (ID, path, screen name), state it explicitly and exactly as it appears in the data
- Provide a clear, concise, and professional response
- Adjust technical depth based on the user's role: developers need paths/codes, clients need plain navigation steps
- Maintain natural conversational flow
- Avoid unnecessary repetition

[FAIL-SAFE CONDITION]
If the Menu context does not contain the required information, utilize conversation history and cross-bot context to provide the best possible guidance. If still unable to answer, state what you don't know based on all available information.

[USER QUESTION]
{question}

Response:
"""

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

        logger.info(f"🔍 Searching menu DuckDB for: {user_input[:100]}")
        context_str = db_query.query_table("MMENU", user_input)
        context_str = context_str[:8000]  # Truncate to prevent GPU OOM on RunPod
        logger.info(f"📚 Menu context: {len(context_str)} chars")

        # Get role-specific system prompt
        role_system_prompt = ROLE_SYSTEM_PROMPTS_MENU.get(user_role, ROLE_SYSTEM_PROMPTS_MENU["client"])

        # Extract cross-bot context from orchestrator_context if available
        cross_bot_context = ""
        if orchestrator_context and "=== Cross-Bot Context" in orchestrator_context:
            cross_bot_start = orchestrator_context.find("=== Cross-Bot Context")
            if cross_bot_start != -1:
                cross_bot_end = orchestrator_context.find("===", cross_bot_start + 1)
                if cross_bot_end == -1:
                    cross_bot_context = orchestrator_context[cross_bot_start:]
                else:
                    cross_bot_context = orchestrator_context[cross_bot_start:cross_bot_end]
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
    return {"status": "healthy"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
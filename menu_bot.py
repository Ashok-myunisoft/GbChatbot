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
 
def _extract_recent_turns(context: str, n_turns: int = 2, max_chars: int = 1200) -> str:
    """Extract last N conversation turns from orchestrator context for history."""
    if not context:
        return ""
    import re as _re
    positions = [m.start() for m in _re.finditer(r'\nTurn \d+:', context)]
    if not positions:
        tail = context[-max_chars:] if len(context) > max_chars else context
        return tail
    start = positions[-min(n_turns, len(positions))]
    recent = context[start:].strip()
    return recent[-max_chars:] if len(recent) > max_chars else recent


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

[PRIOR CONTEXT — BACKGROUND ONLY, do not use as answer source]
{orchestrator_context}
Cross-bot: {cross_bot_context}
History: {history}

[RULES]
- For data/list requests: extract and list the actual values from [MENU DATA] below
- For navigation requests: give the path/steps from [MENU DATA] below
- If [MENU DATA] has no answer: respond exactly "No data found for this request"
- NEVER invent values. NEVER use training knowledge. Use ONLY [MENU DATA]

[MENU DATA — fetched live from PostgreSQL, answer from this only]
{context}

[QUESTION]
{question}

⚠ FINAL INSTRUCTION: The data above is already fetched. Present it DIRECTLY. DO NOT write SQL or suggest running queries.

Answer (use only the Menu Data above):

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
    # Cap orchestrator context — prevents large previous responses (e.g. 252-table list) from drowning actual data
    if orchestrator_context and len(orchestrator_context) > 1500:
        _cut = orchestrator_context[:1500]
        _nl  = _cut.rfind('\n')
        orchestrator_context = (_cut[:_nl] if _nl > 500 else _cut) + "\n[...context truncated...]"
    logger.info(f"📚 Received orchestrator context: {len(orchestrator_context)} chars")

    _greeting_set = {"hi", "hello", "hey", "good morning", "good afternoon",
                     "good evening", "howdy", "greetings", "what's up", "sup"}
    _txt = user_input.lower().strip()
    _first_word = _txt.split()[0] if _txt.split() else ""
    if (_txt in _greeting_set
            or (len(_txt.split()) <= 4
                and _first_word in {"hi", "hello", "hey", "howdy", "greetings", "sup"})):
        formatted_answer = "Hello! I'm here to help you with any questions you have. I can assist you with information from the available data sources. What would you like to know?"
        return {"response": formatted_answer}

    try:
        # Extract last 2 conversation turns for history continuity
        history_str = _extract_recent_turns(message.context or '')

        logger.info(f"🔍 Searching PostgreSQL MMENU for: {user_input[:100]}")
        context_str = db_query.query_table("MMENU", user_input)
        # Truncate at newline boundary to avoid cutting mid-record
        if len(context_str) > 8000:
            _cut = context_str.rfind('\n', 0, 8000)
            context_str = context_str[:(_cut if _cut > 0 else 8000)] + "\n[TRUNCATED]"
        logger.info(f"📚 Menu context: {len(context_str)} chars")

        # Pre-check: empty context → return immediately, skip LLM call
        if not context_str.strip() or context_str.strip().startswith("No data found") or context_str.strip() == "(no rows)":
            return {"response": "No data found for this request.", "source_file": "menu.csv", "bot_name": "Menu Bot"}

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

        try:
            raw = llm.invoke(prompt_text)
        except TimeoutError:
            logger.warning("LLM timed out (cold start) — retrying once")
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
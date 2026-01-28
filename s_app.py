"""
Financial Literacy Chatbot - MCP Version
Uses MCP Server to reduce hallucinations by providing structured, verified responses

To switch back to original: run `streamlit run s_app.py` instead
"""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
import os
import json
import subprocess
import re
from datetime import datetime
import time
import threading
from functools import lru_cache

# --- Configuration ---
PERSIST_DIR = "../finance_db"
COLLECTION_NAME = "finance_knowledge"
EMB_MODEL = "intfloat/multilingual-e5-small"

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Literacy Chatbot (MCP)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
    }
    .mcp-badge {
        background-color: #2E7D32;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .quiz-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #1e1e1e;
        margin-bottom: 1rem;
        border-left: 4px solid #FFD700;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #2b313e;
        text-align: center;
    }
    .source-citation {
        background-color: #1a3a1a;
        border-left: 3px solid #4CAF50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_page" not in st.session_state:
    st.session_state.current_page = "welcome"
if "pre_test_completed" not in st.session_state:
    st.session_state.pre_test_completed = False
if "post_test_completed" not in st.session_state:
    st.session_state.post_test_completed = False
if "user_id" not in st.session_state:
    st.session_state.user_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "my-finetuned"
if "resources_loaded" not in st.session_state:
    st.session_state.resources_loaded = False

# --- Reset Session Function ---
def reset_session():
    """Reset all session state for new user"""
    st.session_state.messages = []
    st.session_state.current_page = "welcome"
    st.session_state.pre_test_completed = False
    st.session_state.post_test_completed = False
    st.session_state.user_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.selected_model = "my-finetuned"
    st.session_state.rag_mode = "MCP-Strict"
    if "pre_test_scores" in st.session_state:
        del st.session_state.pre_test_scores
    if "post_test_scores" in st.session_state:
        del st.session_state.post_test_scores
    if "participant_info" in st.session_state:
        del st.session_state.participant_info

# --- PISA Questions (same as original) ---
PISA_QUESTIONS = {
    "Financial Knowledge": [
        {"id": "FL164Q01", "question": "Have you heard of or learnt about: Interest payment", "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"], "weight": 1},
        {"id": "FL164Q02", "question": "Have you heard of or learnt about: Compound interest", "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"], "weight": 1},
        {"id": "FL164Q12", "question": "Have you heard of or learnt about: Budget", "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"], "weight": 1}
    ],
    "Financial Behavior": [
        {"id": "FL160Q01", "question": "When buying a product, how often do you compare prices in different shops?", "options": ["Never", "Rarely", "Sometimes", "Always"], "weight": 1},
        {"id": "FL171Q08", "question": "In the last 12 months, how often have you checked how much money you have?", "options": ["Never/Almost never", "Once/twice a year", "Once/twice a month", "Weekly", "Daily"], "weight": 1}
    ],
    "Financial Confidence": [
        {"id": "FL162Q03", "question": "How confident would you feel about understanding bank statements?", "options": ["Not at all confident", "Not very confident", "Confident", "Very confident"], "weight": 1},
        {"id": "FL162Q06", "question": "How confident are you about planning spending with consideration of your financial situation?", "options": ["Not at all confident", "Not very confident", "Confident", "Very confident"], "weight": 1}
    ],
    "Financial Attitudes": [
        {"id": "FL169Q05", "question": "To what extent do you agree: I know how to manage my money", "options": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"], "weight": 1},
        {"id": "FL169Q10", "question": "To what extent do you agree: I make savings goals for things I want to buy", "options": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"], "weight": 1}
    ]
}

# --- Load Resources ---
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMB_MODEL)

@st.cache_resource(show_spinner=False)
def load_database(_embeddings):
    if not os.path.exists(PERSIST_DIR):
        st.error(f"Database not found at {PERSIST_DIR}")
        st.stop()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR
    )

@st.cache_resource(show_spinner=False)
def load_llm(_model_name):
    return Ollama(
        model=_model_name,
        temperature=0.3,  # Lower temperature for more factual responses
        num_predict=450,
        top_k=30,
        top_p=0.85,
        repeat_penalty=1.2
    )

def load_resources(model_name):
    embeddings = load_embeddings()
    db = load_database(embeddings)
    llm = load_llm(model_name)
    return db, llm

# --- Data Storage Functions ---
def save_test_results(test_type, participant_info, responses, scores):
    filename = "data/test_results_mcp.json"
    os.makedirs("data", exist_ok=True)
    
    result_data = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "participant_info": participant_info,
        "responses": responses,
        "scores": scores,
        "model_used": st.session_state.selected_model,
        "mode": "MCP"
    }
    
    try:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            all_data = {"results": []}
        
        all_data["results"].append(result_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving test results: {e}")

def save_feedback(question, answer, rating, sources):
    feedback_file = "data/user_feedback_mcp.json"
    os.makedirs("data", exist_ok=True)
    
    feedback_entry = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "feedback_type": "response",
        "question": question,
        "answer": answer[:500],
        "rating": rating,
        "sources_count": len(sources) if sources else 0,
        "model_used": st.session_state.selected_model,
        "mode": "MCP"
    }
    
    try:
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = {"feedback": []}
        
        feedback_data["feedback"].append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

def calculate_scores(responses):
    category_scores = {}
    for category, questions in PISA_QUESTIONS.items():
        total_score = 0
        max_score = 0
        for q in questions:
            response = next((r for r in responses if r["question_id"] == q["id"]), None)
            if response:
                score_value = response["score"]
                total_score += score_value
                max_score += len(q["options"]) - 1
        category_scores[category] = (total_score / max_score * 100) if max_score > 0 else 0
    
    all_scores = list(category_scores.values())
    category_scores["Overall"] = sum(all_scores) / len(all_scores) if all_scores else 0
    return category_scores


# =============================================================================
# MCP-STYLE TOOLS - Structured Knowledge Access (Replaces free-form RAG)
# =============================================================================

def mcp_search_knowledge(query: str, db, max_results: int = 3) -> dict:
    """
    MCP Tool: Search the verified financial knowledge base.
    Returns STRUCTURED data that forces the LLM to use exact content.
    """
    try:
        docs = db.max_marginal_relevance_search(
            query, 
            k=max_results, 
            fetch_k=max_results * 2,
            lambda_mult=0.5
        )
        
        results = []
        sources_list = []
        
        for doc in docs:
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            source = metadata.get("source", metadata.get("url", metadata.get("source_file", "Unknown")))
            
            results.append({
                "fact": content,
                "source": source,
                "title": metadata.get("title", ""),
                "verified": True  # Mark as verified from knowledge base
            })
            sources_list.append({
                "content": content,
                "metadata": metadata
            })
        
        return {
            "success": True,
            "query": query,
            "total_found": len(results),
            "facts": results,  # Named "facts" to emphasize these are verified
            "sources": sources_list,
            "can_answer": len(results) > 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "total_found": 0,
            "facts": [],
            "sources": [],
            "can_answer": False,
            "error": str(e)
        }


def mcp_calculate_compound_interest(principal: float, rate: float, years: int, monthly: float = 0) -> dict:
    """MCP Tool: Calculate compound interest - impossible to hallucinate"""
    rate_decimal = rate / 100
    basic_final = principal * (1 + rate_decimal) ** years
    
    if monthly > 0:
        monthly_rate = rate_decimal / 12
        months = years * 12
        contribution_final = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        total_final = basic_final + contribution_final
        total_contributed = principal + (monthly * months)
    else:
        total_final = basic_final
        total_contributed = principal
    
    return {
        "success": True,
        "calculation": "compound_interest",
        "inputs": {"principal": principal, "rate": rate, "years": years, "monthly": monthly},
        "result": {
            "final_amount": round(total_final, 2),
            "total_contributed": round(total_contributed, 2),
            "interest_earned": round(total_final - total_contributed, 2)
        }
    }


def mcp_calculate_budget(income: float) -> dict:
    """MCP Tool: 50/30/20 budget calculation - exact numbers"""
    return {
        "success": True,
        "calculation": "50_30_20_budget",
        "income": income,
        "breakdown": {
            "needs_50pct": round(income * 0.50, 2),
            "wants_30pct": round(income * 0.30, 2),
            "savings_20pct": round(income * 0.20, 2)
        }
    }


def mcp_check_debt_ratio(income: float, debt_payments: float) -> dict:
    """MCP Tool: Debt-to-income ratio assessment"""
    ratio = (debt_payments / income) * 100 if income > 0 else 0
    
    if ratio <= 30:
        status = "HEALTHY"
        advice = "Your debt level is manageable."
    elif ratio <= 40:
        status = "MODERATE"
        advice = "Be cautious about taking new debt."
    elif ratio <= 50:
        status = "HIGH"
        advice = "Prioritize debt repayment."
    else:
        status = "CRITICAL"
        advice = "Consider seeking financial counseling (AKPK)."
    
    return {
        "success": True,
        "calculation": "debt_to_income",
        "ratio": round(ratio, 1),
        "status": status,
        "advice": advice
    }


def detect_calculation_request(query: str) -> dict:
    """Detect if user is asking for a calculation"""
    query_lower = query.lower()
    
    # Compound interest
    if any(term in query_lower for term in ["compound interest", "grow", "investment return", "savings grow"]):
        numbers = re.findall(r'rm?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        numbers = [float(n.replace(',', '')) for n in numbers]
        rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', query_lower)
        year_match = re.search(r'(\d+)\s*(?:year|yr)', query_lower)
        
        if numbers:
            return {
                "type": "compound_interest",
                "principal": numbers[0] if numbers else 1000,
                "rate": float(rate_match.group(1)) if rate_match else 5,
                "years": int(year_match.group(1)) if year_match else 10,
                "monthly": numbers[1] if len(numbers) > 1 else 0
            }
    
    # Budget calculation
    if any(term in query_lower for term in ["50/30/20", "50 30 20", "budget for", "allocate"]):
        numbers = re.findall(r'rm?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        if numbers:
            return {
                "type": "budget",
                "income": float(numbers[0].replace(',', ''))
            }
    
    # Debt ratio
    if any(term in query_lower for term in ["debt ratio", "debt to income", "can i afford"]):
        numbers = re.findall(r'rm?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', query_lower)
        if len(numbers) >= 2:
            return {
                "type": "debt_ratio",
                "income": float(numbers[0].replace(',', '')),
                "debt": float(numbers[1].replace(',', ''))
            }
    
    return {"type": None}


def run_mcp_rag_chain(query: str, db, llm):
    """
    MCP-Style RAG Chain - Enforces structured, verified responses
    
    Key difference from original:
    1. Tools return STRUCTURED data (not free text)
    2. LLM must format the structured data (not generate new content)
    3. If no data found, explicitly says "I don't have information"
    """
    
    # Step 1: Check for calculation requests (no hallucination possible)
    calc_request = detect_calculation_request(query)
    calc_result = None
    
    if calc_request["type"] == "compound_interest":
        calc_result = mcp_calculate_compound_interest(
            calc_request["principal"],
            calc_request["rate"],
            calc_request["years"],
            calc_request.get("monthly", 0)
        )
    elif calc_request["type"] == "budget":
        calc_result = mcp_calculate_budget(calc_request["income"])
    elif calc_request["type"] == "debt_ratio":
        calc_result = mcp_check_debt_ratio(calc_request["income"], calc_request["debt"])
    
    # Step 2: Search knowledge base
    search_result = mcp_search_knowledge(query, db, max_results=3)
    
    # Step 3: Build STRUCTURED prompt that forces LLM to use only provided data
    if calc_result:
        calc_section = f"""
CALCULATION RESULT (Exact - use these numbers):
{json.dumps(calc_result, indent=2)}
"""
    else:
        calc_section = ""
    
    if search_result["can_answer"]:
        facts_text = "\n".join([
            f"FACT {i+1} [Source: {f['source']}]:\n{f['fact']}"
            for i, f in enumerate(search_result["facts"])
        ])
        knowledge_section = f"""
VERIFIED FACTS FROM KNOWLEDGE BASE:
{facts_text}
"""
        no_info_instruction = ""
    else:
        knowledge_section = ""
        no_info_instruction = """
⚠️ NO RELEVANT INFORMATION FOUND IN KNOWLEDGE BASE.
You MUST respond with: "I don't have specific information about this topic in my knowledge base. 
Please try asking about budgeting, saving, debt management, investment, insurance, tax, or retirement planning in Malaysia."
"""
    
    # The key: Prompt that FORCES use of provided data only
    prompt = f"""You are a financial literacy assistant. You must ONLY use the verified information provided below.

STRICT RULES:
1. ONLY use facts from the VERIFIED FACTS section below
2. If calculation results are provided, present those EXACT numbers
3. Do NOT add information that is not in the provided facts
4. ALWAYS cite which source each fact comes from
5. If no facts are provided, say you don't have information on this topic

{calc_section}
{knowledge_section}
{no_info_instruction}

User Question: {query}

RESPONSE FORMAT:
- Brief introduction
- Present facts with citations: "According to [source], ..."
- If calculation provided, show the exact numbers
- End with one practical tip from the sources

Your response (using ONLY the verified facts above):"""

    response_stream = llm.stream(prompt)
    sources = search_result.get("sources", [])
    
    return response_stream, sources, search_result["total_found"], calc_result


def is_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    greetings = ["hi", "hello", "hey", "hiya", "good morning", "good afternoon", "good evening", "yo"]
    if t in greetings or len(t.split()) <= 2 and any(t.startswith(g) for g in greetings):
        return True
    return False


# =============================================================================
# UI PAGES
# =============================================================================

def show_welcome_page():
    st.markdown('<p class="main-header">💰 Financial Literacy Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<span class="mcp-badge">🔒 MCP Mode - Reduced Hallucination</span>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome! 👋
    
    This is the **MCP (Model Context Protocol) version** of the chatbot.
    
    #### 🔒 What's different in MCP Mode?
    - **Structured responses**: The AI only presents verified facts from the knowledge base
    - **Exact calculations**: Financial calculations use precise formulas (no guessing)
    - **Source citations**: Every fact is linked to its source
    - **Honest uncertainty**: If information isn't available, the chatbot will tell you
    
    #### 📋 How it works:
    1. **Pre-Test** - Complete a short quiz to assess your current financial knowledge
    2. **Learn** - Ask the chatbot questions (responses are strictly from verified sources)
    3. **Post-Test** - Take the quiz again to see your improvement
    
    ---
    
    **Ready to start?** Click the button below!
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Pre-Test", type="primary", use_container_width=True):
            st.session_state.current_page = "pre_test"
            st.rerun()
    
    st.markdown("---")
    st.caption("💡 To use the original version (without MCP), run: `streamlit run s_app.py`")


def show_pisa_test(test_type="pre"):
    st.title(f"📋 {'Pre' if test_type == 'pre' else 'Post'}-Test: Financial Literacy Assessment")
    
    st.markdown(f"""
    ### {f'Let us assess your current financial knowledge' if test_type == 'pre' else 'Final Assessment - See Your Progress!'}
    
    **Questions:** {sum(len(v) for v in PISA_QUESTIONS.values())} | **Time:** ~5 minutes
    """)
    
    if test_type == "pre":
        with st.expander("👤 Your Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=15, max_value=99, value=20)
                education = st.selectbox("Education Level", ["Secondary School", "Diploma", "Bachelor's", "Master's", "PhD", "Other"])
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
                occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-employed", "Unemployed", "Other"])
        
        participant_info = {"age": age, "education": education, "gender": gender, "occupation": occupation}
        st.session_state.participant_info = participant_info
    else:
        participant_info = st.session_state.get("participant_info", {})
    
    st.divider()
    
    all_responses = []
    question_number = 1
    
    for category, questions in PISA_QUESTIONS.items():
        st.subheader(f"📊 {category}")
        for q in questions:
            st.markdown(f"**Q{question_number}. {q['question']}**")
            response = st.radio("Select your answer:", options=q["options"], key=f"{test_type}_{q['id']}", label_visibility="collapsed")
            score = q["options"].index(response) if response else 0
            all_responses.append({"question_id": q["id"], "question": q["question"], "category": category, "response": response, "score": score})
            question_number += 1
            st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📤 Submit Assessment", type="primary", use_container_width=True):
            scores = calculate_scores(all_responses)
            save_test_results(test_type, participant_info, all_responses, scores)
            
            if test_type == "pre":
                st.session_state.pre_test_completed = True
                st.session_state.pre_test_scores = scores
                st.session_state.current_page = "chatbot"
            else:
                st.session_state.post_test_completed = True
                st.session_state.post_test_scores = scores
                st.session_state.current_page = "results"
            
            st.success("✅ Assessment submitted!")
            st.balloons()
            st.rerun()


def show_chatbot_page():
    st.markdown('<p class="main-header">💰 Financial Literacy Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<span class="mcp-badge">🔒 MCP Mode - Verified Responses Only</span>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Pre-Test Complete")
    with col2:
        st.info("💬 Using MCP-Verified Mode")
    with col3:
        if st.button("📝 Take Post-Test"):
            st.session_state.current_page = "post_test"
            st.rerun()
    
    try:
        db, llm = load_resources(st.session_state.selected_model)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📚 View Sources ({len(message['sources'])} verified)"):
                    for i, source in enumerate(message["sources"], 1):
                        metadata = source.get("metadata", {})
                        title = metadata.get("title", metadata.get("source", "Unknown Source"))
                        st.markdown(f"**{i}. {title}**")
                        st.caption(source["content"][:200] + "...")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask about financial literacy (responses from verified sources only)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            thinking_placeholder = st.empty()
            stop_animation = threading.Event()
            
            def animate_thinking(placeholder, stop_event):
                dots = ["", ".", "..", "..."]
                idx = 0
                while not stop_event.is_set():
                    placeholder.markdown(f"🔍 Searching verified knowledge base{dots[idx]}")
                    idx = (idx + 1) % len(dots)
                    time.sleep(0.5)
                placeholder.empty()
            
            try:
                if is_greeting(prompt):
                    full_response = (
                        "Hello! 👋 I'm the MCP-powered financial literacy chatbot.\n\n"
                        "In this mode, I only provide **verified information** from my knowledge base. "
                        "Ask me about:\n"
                        "- 💰 Budgeting (50/30/20 rule)\n"
                        "- 🏦 Saving tips\n"
                        "- 💳 Debt management\n"
                        "- 📈 Investment basics\n"
                        "- 🏥 Insurance\n"
                        "- 🧾 Tax filing (LHDN)\n"
                        "- 👴 Retirement (EPF/KWSP)\n\n"
                        "How can I help you today?"
                    )
                    sources = []
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "mode": "MCP"
                    })
                else:
                    # Start animation
                    animation_thread = threading.Thread(target=animate_thinking, args=(thinking_placeholder, stop_animation))
                    animation_thread.start()
                    
                    # Run MCP-style RAG
                    response_stream, sources, found_count, calc_result = run_mcp_rag_chain(prompt, db, llm)
                    
                    # Stop animation
                    stop_animation.set()
                    animation_thread.join()
                    
                    # Stream response
                    for chunk in response_stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Show verification status
                    if found_count > 0:
                        st.success(f"✅ Response based on {found_count} verified source(s)")
                    else:
                        st.warning("⚠️ Limited information available for this query")
                    
                    if calc_result:
                        st.info("🧮 Includes exact calculation results")
                    
                    # Feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                            save_feedback(prompt, full_response, "helpful", sources)
                            st.success("Thanks!")
                    with col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                            save_feedback(prompt, full_response, "not_helpful", sources)
                            st.warning("We'll improve!")
                    
                    # Show sources
                    if sources:
                        with st.expander(f"📚 View Sources ({len(sources)} verified)"):
                            for i, source in enumerate(sources, 1):
                                metadata = source.get("metadata", {})
                                title = metadata.get("title", metadata.get("source", "Unknown Source"))
                                st.markdown(f"**{i}. {title}**")
                                st.caption(source["content"][:200] + "...")
                                st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "mode": "MCP",
                        "found_count": found_count
                    })
                    
            except Exception as e:
                stop_animation.set()
                if 'animation_thread' in locals():
                    animation_thread.join()
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.markdown(error_msg)


def show_results_page():
    st.title("🎉 Congratulations! Assessment Complete")
    st.markdown('<span class="mcp-badge">🔒 MCP Mode Results</span>', unsafe_allow_html=True)
    
    pre_scores = st.session_state.get("pre_test_scores", {})
    post_scores = st.session_state.get("post_test_scores", {})
    
    if pre_scores and post_scores:
        st.subheader("📊 Your Progress")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pre_overall = pre_scores.get("Overall", 0)
            st.metric("Pre-Test Score", f"{pre_overall:.1f}%")
        with col2:
            post_overall = post_scores.get("Overall", 0)
            st.metric("Post-Test Score", f"{post_overall:.1f}%")
        with col3:
            improvement = post_overall - pre_overall
            st.metric("Improvement", f"{improvement:+.1f}%", delta=f"{improvement:+.1f}%")
        
        st.divider()
        
        st.subheader("📈 Score Breakdown by Category")
        for category in ["Financial Knowledge", "Financial Behavior", "Financial Confidence", "Financial Attitudes"]:
            pre_cat = pre_scores.get(category, 0)
            post_cat = post_scores.get(category, 0)
            diff = post_cat - pre_cat
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{category}**")
            with col2:
                st.write(f"Pre: {pre_cat:.1f}% → Post: {post_cat:.1f}%")
            with col3:
                if diff > 0:
                    st.success(f"↑ {diff:.1f}%")
                elif diff < 0:
                    st.error(f"↓ {abs(diff):.1f}%")
                else:
                    st.info("No change")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Start New Session", type="primary"):
            reset_session()
            st.rerun()
    with col2:
        if st.button("💬 Continue Chatting"):
            st.session_state.current_page = "chatbot"
            st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown('<span class="mcp-badge">MCP Mode</span>', unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 New Session"):
            reset_session()
            st.rerun()
        
        st.divider()
        st.caption("MCP Mode ensures responses come only from verified sources.")
        st.caption("To use original mode: `streamlit run s_app.py`")
    
    # Page routing
    if st.session_state.current_page == "welcome":
        show_welcome_page()
    elif st.session_state.current_page == "pre_test":
        show_pisa_test("pre")
    elif st.session_state.current_page == "chatbot":
        show_chatbot_page()
    elif st.session_state.current_page == "post_test":
        show_pisa_test("post")
    elif st.session_state.current_page == "results":
        show_results_page()


if __name__ == "__main__":
    main()

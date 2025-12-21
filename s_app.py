import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
import os
import json
from datetime import datetime

# --- Configuration ---
PERSIST_DIR = "../finance_db"
COLLECTION_NAME = "finance_knowledge"
EMB_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "llama3.2"

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Literacy Chatbot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
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

# --- Reset Session Function ---
def reset_session():
    """Reset all session state for new user"""
    st.session_state.messages = []
    st.session_state.current_page = "welcome"
    st.session_state.pre_test_completed = False
    st.session_state.post_test_completed = False
    st.session_state.user_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "pre_test_scores" in st.session_state:
        del st.session_state.pre_test_scores
    if "post_test_scores" in st.session_state:
        del st.session_state.post_test_scores
    if "participant_info" in st.session_state:
        del st.session_state.participant_info

# --- PISA Questions ---
PISA_QUESTIONS = {
    "Financial Knowledge": [
        {
            "id": "FL164Q01",
            "question": "Have you heard of or learnt about: Interest payment",
            "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"],
            "weight": 1
        },
        {
            "id": "FL164Q02",
            "question": "Have you heard of or learnt about: Compound interest",
            "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"],
            "weight": 1
        },
        {
            "id": "FL164Q12",
            "question": "Have you heard of or learnt about: Budget",
            "options": ["Never heard of it", "Heard of it, but don't recall meaning", "Know what it means"],
            "weight": 1
        }
    ],
    "Financial Behavior": [
        {
            "id": "FL160Q01",
            "question": "When buying a product, how often do you compare prices in different shops?",
            "options": ["Never", "Rarely", "Sometimes", "Always"],
            "weight": 1
        },
        {
            "id": "FL171Q08",
            "question": "In the last 12 months, how often have you checked how much money you have?",
            "options": ["Never/Almost never", "Once/twice a year", "Once/twice a month", "Weekly", "Daily"],
            "weight": 1
        }
    ],
    "Financial Confidence": [
        {
            "id": "FL162Q03",
            "question": "How confident would you feel about understanding bank statements?",
            "options": ["Not at all confident", "Not very confident", "Confident", "Very confident"],
            "weight": 1
        },
        {
            "id": "FL162Q06",
            "question": "How confident are you about planning spending with consideration of your financial situation?",
            "options": ["Not at all confident", "Not very confident", "Confident", "Very confident"],
            "weight": 1
        }
    ],
    "Financial Attitudes": [
        {
            "id": "FL169Q05",
            "question": "To what extent do you agree: I know how to manage my money",
            "options": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"],
            "weight": 1
        },
        {
            "id": "FL169Q10",
            "question": "To what extent do you agree: I make savings goals for things I want to buy",
            "options": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"],
            "weight": 1
        }
    ]
}

# --- Load Resources ---
@st.cache_resource
def load_resources():
    if not os.path.exists(PERSIST_DIR):
        st.error(f"Database not found at {PERSIST_DIR}")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    # OPTIMIZED LLM with speed parameters
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.7,
        num_predict=256,  # Limit response length (faster)
        top_k=10,         # Reduce sampling space (faster)
        top_p=0.9         # Nucleus sampling (more focused)
    )
    return db, llm

db, llm = load_resources()

# --- Data Storage Functions ---
def save_test_results(test_type, participant_info, responses, scores):
    """Save test results to JSON file"""
    filename = "data/test_results.json"
    os.makedirs("data", exist_ok=True)
    
    result_data = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "participant_info": participant_info,
        "responses": responses,
        "scores": scores
    }
    
    try:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            all_data = {"results": []}
        except Exception as e:
            st.error(f"Error loading test results: {e}")
            all_data = {"results": []}
        
        all_data["results"].append(result_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {test_type} test for user {st.session_state.user_id}")
    except Exception as e:
        st.error(f"Error saving test results: {e}")
        print(f"❌ Failed to save {test_type} test: {e}")

def save_feedback(question, answer, rating, sources):
    """Save user feedback"""
    feedback_file = "data/user_feedback.json"
    os.makedirs("data", exist_ok=True)
    
    feedback_entry = {
        "user_id": st.session_state.user_id,
        "timestamp": datetime.now().isoformat(),
        "feedback_type": "response",
        "question": question,
        "answer": answer[:500],
        "rating": rating,
        "sources_count": len(sources)
    }
    
    try:
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = {"feedback": []}
        except Exception as e:
            st.error(f"Error loading feedback: {e}")
            feedback_data = {"feedback": []}
        
        feedback_data["feedback"].append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved feedback: {rating}")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        print(f"❌ Failed to save feedback: {e}")

def save_general_feedback(user_id, feedback_text, rating):
    """Save general user feedback about the chatbot experience"""
    feedback_file = "data/user_feedback.json"
    os.makedirs("data", exist_ok=True)
    
    feedback_entry = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "feedback_type": "general",
        "feedback_text": feedback_text,
        "rating": rating,
        "question": "General Feedback",
        "answer": "N/A",
        "sources_count": 0
    }
    
    try:
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = {"feedback": []}
        except Exception as e:
            feedback_data = {"feedback": []}
        
        feedback_data["feedback"].append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved general feedback")
    except Exception as e:
        st.error(f"Error saving general feedback: {e}")
        print(f"❌ Failed to save feedback: {e}")

# --- Score Calculation ---
def calculate_scores(responses):
    """Calculate scores by category"""
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

# --- RAG Functions ---
def rewrite_query(query: str) -> str:
    """Expand query for better retrieval"""
    financial_keywords = {
        "save": "saving tips money management",
        "budget": "budgeting financial planning spending",
        "debt": "debt management loan credit card",
        "invest": "investment returns stocks bonds",
        "retire": "retirement planning EPF KWSP",
        "emergency": "emergency fund savings buffer"
    }
    
    query_lower = query.lower()
    for keyword, expansion in financial_keywords.items():
        if keyword in query_lower:
            return f"{query} {expansion}"
    return query

def run_rag_chain(query: str, llm):
    expanded_query = rewrite_query(query)
    
    # OPTIMIZED: Reduce number of documents retrieved
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 12, "lambda_mult": 0.5}
    )
    
    docs = retriever.invoke(expanded_query)
    
    # OPTIMIZED: Limit context length to speed up processing
    context_parts = []
    total_chars = 0
    max_chars = 2000
    
    for doc in docs:
        content = doc.page_content
        if total_chars + len(content) <= max_chars:
            context_parts.append(content)
            total_chars += len(content)
        else:
            remaining = max_chars - total_chars
            if remaining > 200:
                context_parts.append(content[:remaining])
            break
    
    context = "\n\n".join(context_parts)
    
    # OPTIMIZED: Shorter, more focused prompt
    prompt = f"""You are a financial literacy assistant for Malaysian youth.

Context:
{context}

Question: {query}

Provide a clear, concise answer (2-3 paragraphs max) based only on the context above:"""
    
    response_stream = llm.stream(prompt)
    sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    
    return response_stream, sources

# --- UI Pages ---
def show_welcome_page():
    """Welcome page"""
    st.markdown('<p class="main-header">💰 Financial Literacy Chatbot</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome! 👋
    
    This AI-powered chatbot helps you learn about financial literacy using information from 
    **Malaysian EPF (KWSP)** and financial education resources.
    
    #### 📋 How it works:
    1. **Pre-Test** - Complete a short quiz to assess your current financial knowledge
    2. **Learn** - Ask the chatbot any questions about financial literacy
    3. **Post-Test** - Take the quiz again to see your improvement
    4. **Results** - View your learning progress
    
    #### ⏱️ Time Required:
    - Pre-test: ~5 minutes
    - Chatbot interaction: 15-20 minutes (recommended)
    - Post-test: ~5 minutes
    
    ---
    
    **Ready to start?** Click the button below!
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start Pre-Test", type="primary", use_container_width=True):
            st.session_state.current_page = "pre_test"
            st.rerun()

def show_pisa_test(test_type="pre"):
    """Show PISA test"""
    st.title(f"📋 {'Pre' if test_type == 'pre' else 'Post'}-Test: Financial Literacy Assessment")
    
    st.markdown(f"""
    ### {f'Let us assess your current financial knowledge' if test_type == 'pre' else 'Final Assessment - See Your Progress!'}
    
    This assessment is based on the **PISA 2022 Financial Literacy Framework** used globally by OECD.
    
    **Questions:** {sum(len(v) for v in PISA_QUESTIONS.values())} | **Time:** ~5 minutes
    """)
    
    if test_type == "pre":
        with st.expander("👤 Your Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=15, max_value=99, value=20)
                education = st.selectbox("Education Level",
                    ["Secondary School", "Diploma", "Bachelor's", "Master's", "PhD", "Other"])
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Prefer not to say"])
                occupation = st.selectbox("Occupation",
                    ["Student", "Employee", "Self-employed", "Unemployed", "Other"])
        
        participant_info = {
            "age": age,
            "education": education,
            "gender": gender,
            "occupation": occupation
        }
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
            
            response = st.radio(
                "Select your answer:",
                options=q["options"],
                key=f"{test_type}_{q['id']}",
                label_visibility="collapsed"
            )
            
            score = q["options"].index(response) if response else 0
            
            all_responses.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": category,
                "response": response,
                "score": score
            })
            
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
    """Main chatbot interface"""
    st.markdown('<p class="main-header">💰 Financial Literacy Chatbot</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Pre-Test Complete")
    with col2:
        st.info("💬 Currently Using Chatbot")
    with col3:
        if st.button("📝 Take Post-Test"):
            st.session_state.current_page = "post_test"
            st.rerun()
    
    st.divider()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        metadata = source["metadata"]
                        title = metadata.get("title", "Unknown Source")
                        url = metadata.get("url", "https://www.kwsp.gov.my")
                        
                        st.markdown(f"**Source {i}:** [{title}]({url})")
                        st.caption(source["content"][:300] + "...")
                        st.divider()
    
    if prompt := st.chat_input("Ask about financial literacy..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                response_stream, sources = run_rag_chain(prompt, llm)
                
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                        save_feedback(prompt, full_response, "helpful", sources)
                        st.success("Thanks!")
                with col2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                        save_feedback(prompt, full_response, "not_helpful", sources)
                        st.warning("We'll improve!")
                
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        metadata = source["metadata"]
                        title = metadata.get("title", "Unknown Source")
                        url = metadata.get("url", "https://www.kwsp.gov.my")
                        
                        st.markdown(f"**Source {i}:** [{title}]({url})")
                        st.caption(source["content"][:300] + "...")
                        st.divider()
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.markdown(error_msg)

def show_results_page():
    """Show final results and improvement"""
    st.title("🎉 Congratulations! You've Completed the Assessment")
    
    pre_scores = st.session_state.get("pre_test_scores", {})
    post_scores = st.session_state.get("post_test_scores", {})
    
    st.subheader("📊 Your Progress")
    
    categories = ["Financial Knowledge", "Financial Behavior", "Financial Confidence", "Financial Attitudes", "Overall"]
    
    for category in categories:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"**{category}**")
        with col2:
            st.metric("Before", f"{pre_scores.get(category, 0):.0f}%")
        with col3:
            st.metric("After", f"{post_scores.get(category, 0):.0f}%")
        with col4:
            improvement = post_scores.get(category, 0) - pre_scores.get(category, 0)
            if improvement > 0:
                st.success(f"+{improvement:.0f}%")
            elif improvement < 0:
                st.error(f"{improvement:.0f}%")
            else:
                st.info("0%")
    
    st.divider()
    
    overall_improvement = post_scores.get("Overall", 0) - pre_scores.get("Overall", 0)
    
    if overall_improvement > 15:
        st.success(f"🌟 **Excellent Progress!** You improved by {overall_improvement:.1f}% overall!")
    elif overall_improvement > 5:
        st.info(f"✅ **Good Work!** You improved by {overall_improvement:.1f}%")
    elif overall_improvement > 0:
        st.info(f"👍 **You Improved!** +{overall_improvement:.1f}%")
    else:
        st.warning("Consider spending more time learning with the chatbot.")
    
    st.divider()
    
    # Feedback Form
    st.subheader("💭 We'd Love Your Feedback!")
    
    with st.form("feedback_form"):
        st.write("Please share your experience with the Financial Literacy Chatbot:")
        
        rating = st.radio(
            "How would you rate your overall experience?",
            ["⭐ Excellent", "⭐ Good", "⭐ Average", "⭐ Poor"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            helpful = st.select_slider(
                "How helpful was the chatbot?",
                options=["Not helpful", "Somewhat helpful", "Helpful", "Very helpful", "Extremely helpful"]
            )
        with col2:
            easy_to_use = st.select_slider(
                "How easy was it to use?",
                options=["Very difficult", "Difficult", "Neutral", "Easy", "Very easy"]
            )
        
        feedback_text = st.text_area(
            "What did you like or dislike about the chatbot? Any suggestions for improvement?",
            placeholder="Your feedback helps us improve...",
            height=100
        )
        
        topics = st.multiselect(
            "Which topics did you find most useful? (Optional)",
            ["Saving Money", "Budgeting", "EPF/KWSP", "Investing", "Debt Management", 
             "Emergency Fund", "Credit Cards", "Insurance", "Other"]
        )
        
        submitted = st.form_submit_button("📤 Submit Feedback", type="primary")
        
        if submitted:
            if feedback_text.strip():
                comprehensive_feedback = {
                    "overall_rating": rating,
                    "helpfulness": helpful,
                    "ease_of_use": easy_to_use,
                    "feedback_text": feedback_text,
                    "useful_topics": topics
                }
                
                save_general_feedback(
                    st.session_state.user_id,
                    str(comprehensive_feedback),
                    rating.split()[1].lower()
                )
                
                st.success("✅ Thank you for your feedback!")
                st.balloons()
            else:
                st.warning("Please provide some feedback in the text area.")
    
    st.divider()
    
    st.markdown("### 💡 Thank you for participating!")
    st.markdown("Your feedback helps us improve the chatbot for future users.")
    
    # New User Button
    st.subheader("🔄 Next Steps")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("👤 Start New User Session", type="primary", use_container_width=True):
            reset_session()
            st.rerun()
    
    st.caption("Click above to reset and allow another person to use the chatbot")

# --- Admin Dashboard ---
def show_admin_dashboard():
    """Admin page to view all user results"""
    # LOGOUT BUTTON
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.title("👨‍💼 Admin Dashboard")
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.current_page = "welcome"
            st.rerun()
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Test Results", "💬 Feedback", "📈 Analytics"])
    
    with tab1:
        st.subheader("All Test Results")
        
        try:
            with open("data/test_results.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get("results", [])
            
            if not results:
                st.warning("No test data available yet.")
            else:
                st.metric("Total Participants", len(set(r["user_id"] for r in results)))
                
                test_type_filter = st.selectbox("Filter by test type:", ["All", "pre", "post"])
                
                for result in results:
                    if test_type_filter != "All" and result["test_type"] != test_type_filter:
                        continue
                    
                    with st.expander(f"User: {result['user_id']} - {result['test_type'].upper()} Test - {result['timestamp'][:10]}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Participant Info:**")
                            for key, value in result["participant_info"].items():
                                st.write(f"- {key.title()}: {value}")
                        
                        with col2:
                            st.write("**Scores:**")
                            for key, value in result["scores"].items():
                                st.write(f"- {key}: {value:.1f}%")
                        
                        st.write("**Detailed Responses:**")
                        for resp in result["responses"]:
                            st.write(f"**Q:** {resp['question']}")
                            st.write(f"**A:** {resp['response']} (Score: {resp['score']})")
                            st.divider()
        
        except FileNotFoundError:
            st.error("No test results file found.")
    
    with tab2:
        st.subheader("User Feedback")
        
        try:
            with open("data/user_feedback.json", 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            feedbacks = feedback_data.get("feedback", [])
            
            if not feedbacks:
                st.warning("No feedback data available yet.")
            else:
                helpful = sum(1 for f in feedbacks if f["rating"] == "helpful")
                not_helpful = sum(1 for f in feedbacks if f["rating"] == "not_helpful")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Feedback", len(feedbacks))
                with col2:
                    st.metric("👍 Helpful", helpful)
                with col3:
                    st.metric("👎 Not Helpful", not_helpful)
                
                st.divider()
                
                for fb in feedbacks:
                    feedback_type = fb.get("feedback_type", "response")
                    
                    if feedback_type == "general":
                        with st.expander(f"💭 General Feedback - {fb['timestamp'][:10]}"):
                            st.write(f"**User:** {fb['user_id']}")
                            st.write(f"**Rating:** {fb['rating'].upper()}")
                            st.write(f"**Feedback:**")
                            st.info(fb['feedback_text'])
                    else:
                        with st.expander(f"{fb['rating'].title()} - {fb['timestamp'][:10]}"):
                            st.write(f"**User:** {fb['user_id']}")
                            st.write(f"**Question:** {fb['question']}")
                            st.write(f"**Answer:** {fb['answer']}")
                            st.write(f"**Sources Used:** {fb['sources_count']}")
        
        except FileNotFoundError:
            st.error("No feedback file found.")
    
    with tab3:
        st.subheader("Analytics")
        
        try:
            with open("data/test_results.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get("results", [])
            
            if results:
                user_improvements = {}
                
                for result in results:
                    user_id = result["user_id"]
                    test_type = result["test_type"]
                    overall_score = result["scores"]["Overall"]
                    
                    if user_id not in user_improvements:
                        user_improvements[user_id] = {}
                    
                    user_improvements[user_id][test_type] = overall_score
                
                improvements = []
                for user_id, scores in user_improvements.items():
                    if "pre" in scores and "post" in scores:
                        improvement = scores["post"] - scores["pre"]
                        improvements.append(improvement)
                
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Users with Both Tests", len(improvements))
                    with col2:
                        st.metric("Average Improvement", f"{avg_improvement:.1f}%")
                    with col3:
                        improved = sum(1 for i in improvements if i > 0)
                        st.metric("Users Improved", f"{improved}/{len(improvements)}")
                else:
                    st.info("No complete pre/post test pairs yet.")
            else:
                st.warning("No data available.")
        
        except FileNotFoundError:
            st.error("No test results file found.")

# --- Main App Navigation ---
def main():
    """Main application"""
    
    with st.sidebar:
        st.title("📍 Navigation")
        
        if st.session_state.current_page == "welcome":
            st.info("👋 Welcome Page")
        elif not st.session_state.pre_test_completed:
            st.info("📝 Pre-Test")
        elif not st.session_state.post_test_completed:
            st.success("✅ Pre-Test Done")
            st.info("💬 Using Chatbot")
        else:
            st.success("✅ All Complete!")
        
        st.divider()
        
        st.header("📊 Progress")
        progress = 0
        if st.session_state.pre_test_completed:
            progress += 33
        if len(st.session_state.messages) > 0:
            progress += 34
        if st.session_state.post_test_completed:
            progress += 33
        
        st.progress(progress / 100)
        st.caption(f"{progress}% Complete")
        
        st.divider()
        
        # NEW USER RESET BUTTON
        if st.button("🔄 New User", help="Reset for a new participant"):
            if st.session_state.current_page not in ["welcome", "admin"]:
                st.warning("⚠️ This will reset all progress!")
                if st.button("✅ Confirm Reset"):
                    reset_session()
                    st.rerun()
            else:
                reset_session()
                st.rerun()
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - 🤖 RAG (Retrieval-Augmented Generation)
        - 📚 KWSP/EPF Resources
        - 📋 PISA 2022 Framework
        """)
        
        st.divider()
        
        admin_password = st.text_input("Admin Access", type="password")
        if admin_password == "admin123":
            if st.button("📊 View Dashboard"):
                st.session_state.current_page = "admin"
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area
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
    elif st.session_state.current_page == "admin":
        show_admin_dashboard()

if __name__ == "__main__":
    main()

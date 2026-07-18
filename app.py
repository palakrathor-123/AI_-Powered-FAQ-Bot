import streamlit as st
import pandas as pd
from datetime import datetime
from src.embedder import FAQEmbedder
from src.search_engine import FAQSearchEngine
from src.nlp_tasks import NLPAdvancedTasks

# Force wide layout to match the master template
st.set_page_config(page_title="AI-Powered FAQ Bot", layout="wide")

# Custom CSS for styling (replicating original layout, colors, and fonts)
st.markdown("""
<style>
    /* Global Page Styling */
    .reportview-container {
        background-color: #F9FAFB;
    }
    .main-title {
        text-align: center; 
        color: #0F172A; 
        font-family: 'Inter', sans-serif; 
        font-weight: 700; 
        font-size: 28px;
        margin-bottom: 2px;
    }
    .subtitle {
        text-align: center; 
        color: #4B5563; 
        font-size: 14px; 
        margin-bottom: 25px;
    }

    /* Input Styling Box */
    .search-section {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        margin-bottom: 25px;
    }

    /* Left Panel FAQ Bot Box */
    .faq-container {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        background-color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .faq-header {
        background-color: #0B3C95;
        color: white;
        padding: 15px 20px;
        font-family: 'Inter', sans-serif;
    }
    .faq-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
        color: white !important;
    }
    .faq-header p {
        margin: 2px 0 0 0;
        font-size: 12px;
        color: #93C5FD;
    }
    .online-indicator {
        float: right;
        background-color: #10B981;
        color: white;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: bold;
    }

    /* Message Bubbles layout with Icons */
    .message-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
        padding: 10px 20px;
    }
    .avatar-icon {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-weight: bold;
        color: white;
        flex-shrink: 0;
    }
    .user-avatar {
        background-color: #3B82F6;
    }
    .bot-avatar {
        background-color: #10B981;
    }
    .message-content {
        flex-grow: 1;
    }
    .message-meta {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #9CA3AF;
        margin-bottom: 4px;
    }
    .bubble-user {
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        color: #1E3A8A;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13.5px;
    }
    .bubble-bot {
        background-color: #ECFDF5;
        border: 1px solid #D1FAE5;
        color: #065F46;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 13.5px;
    }
    .meta-footer {
        display: flex;
        justify-content: space-between;
        margin-top: 6px;
        font-size: 11px;
        color: #6B7280;
    }

    /* Right Panel Examples */
    .right-header {
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 15px;
    }
    .example-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
        background-color: white;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .example-title {
        font-size: 13px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sentiment-card { border-left: 5px solid #8B5CF6; }
    .sentiment-title { color: #8B5CF6; }
    .translation-card { border-left: 5px solid #3B82F6; }
    .translation-title { color: #3B82F6; }
    .fallback-card { border-left: 5px solid #F59E0B; }
    .fallback-title { color: #F59E0B; }
    .summarize-card { border-left: 5px solid #10B981; }
    .summarize-title { color: #10B981; }

    /* Footer Banner */
    .footer-banner {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 15px;
        margin-top: 25px;
        display: flex;
        align-items: center;
    }
    .footer-icon {
        font-size: 24px;
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Headers
st.markdown("<h2 class='main-title'>AI-Powered FAQ Bot – Input / Output Examples</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>The bot finds the most relevant answer from the FAQ knowledge base using semantic search.</p>", unsafe_allow_html=True)

# Data loader and Backend engine initialization
@st.cache_data
def load_data():
    return pd.read_csv("data/faq_data.csv")

df = load_data()

if 'embedder' not in st.session_state:
    st.session_state.embedder = FAQEmbedder()
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = FAQSearchEngine(df, st.session_state.embedder)

# Store persistent chat log locally
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

# Dynamic states for the last query to feed the right-side analysis
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_answer' not in st.session_state:
    st.session_state.last_answer = ""
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = 0.0
if 'last_time' not in st.session_state:
    st.session_state.last_time = ""

# Track if user has interacted
if 'has_searched' not in st.session_state:
    st.session_state.has_searched = False

# ==================== 1. TOP SECTION: ALWAYS VISIBLE INPUT BOX ====================
with st.container():
    # Streamlit standard text input
    user_input = st.text_input("Type your question...", placeholder="Enter your question here...", key="live_input_box")
    
    col_btn1, _ = st.columns([1, 8])
    with col_btn1:
        if st.button("Send", type="primary") and user_input:
            # Backend process prediction
            ans_res = st.session_state.search_engine.query(user_input)
            current_time_str = datetime.now().strftime("%I:%M %p")
            
            # Save to main chat list
            st.session_state.chat_log.append({
                "question": user_input,
                "answer": ans_res['answer'],
                "category": ans_res['category'],
                "confidence": ans_res['confidence'],
                "time": current_time_str
            })
            
            # Save dynamic states for right-side models
            st.session_state.last_query = user_input
            st.session_state.last_answer = ans_res['answer']
            st.session_state.last_confidence = ans_res['confidence']
            st.session_state.last_time = current_time_str
            
            # Set display triggers
            st.session_state.has_searched = True
            st.rerun()

# ==================== 2. CONDITIONAL LAYOUT SECTION ====================
# Yeh blocks tabhi render honge jab user apna pehla query submit karega
if st.session_state.has_searched:
    st.markdown("---")  # Elegant separation divider
    
    col1, col2 = st.columns([1.1, 0.9])

    # ---------------- LEFT PANEL: DYNAMIC FAQ CHATBOT ----------------
    with col1:
        st.markdown("""
        <div class="faq-container">
            <div class="faq-header">
                <span class="online-indicator">● Online</span>
                <h3>🤖 FAQ Bot</h3>
                <p>Ask me anything from our FAQ!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        chat_container = st.container()
        with chat_container:
            st.write("")  # Spacer
            
            # Render chat log dynamically
            for chat in st.session_state.chat_log:
                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar-icon user-avatar">👤</div>
                    <div class="message-content">
                        <div class="message-meta"><b>You</b> <span>{chat['time']}</span></div>
                        <div class="bubble-user">{chat['question']}</div>
                    </div>
                </div>
                <div class="message-row">
                    <div class="avatar-icon bot-avatar">🤖</div>
                    <div class="message-content">
                        <div class="message-meta"><b>FAQ Bot</b> <span>{chat['time']}</span></div>
                        <div class="bubble-bot">{chat['answer']}</div>
                        <div class="meta-footer"><span>Category: {chat['category']}</span> <span>Confidence: {chat['confidence']}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ---------------- RIGHT PANEL ----------------#
    
    with col2:
        st.markdown("<div class='right-header'>Other Examples</div>", unsafe_allow_html=True)

        # Example 1: Fixed Sentiment Analysis Example
        st.markdown(f"""
        <div class="example-card sentiment-card">
            <div class="example-title sentiment-title">Example 1 – Sentiment Analysis</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>10:30 AM</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">I am very disappointed with the refund process. 😞</div>
                </div>
            </div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon bot-avatar" style="width:30px; height:30px; font-size:12px; background-color:#10B981;">🤖</div>
                <div class="message-content">
                    <div class="message-meta"><b>FAQ Bot</b> <span>10:30 AM</span></div>
                    <div class="bubble-bot" style="padding:6px 10px; font-size:12.5px; background-color:#ECFDF5; color:#065F46;">I'm sorry to hear that. Please share your order ID, and we will help you with the refund status.</div>
                </div>
            </div>
            <div style="font-size:11px; color:#6B7280; font-weight:500; margin-left:42px; margin-top: 5px;">
                Sentiment: <span style="background-color:#FEE2E2; color:#991B1B; padding:2px 6px; border-radius:4px; font-weight:bold;">Negative</span> (Score: -0.78)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 2: Fixed Multilingual Translation Example
        st.markdown(f"""
        <div class="example-card translation-card">
            <div class="example-title translation-title">Example 2 – Translation (Hindi)</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>10:31 AM</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">मेरा पासवर्ड कैसे बदलूं?</div>
                </div>
            </div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon bot-avatar" style="width:30px; height:30px; font-size:12px; background-color:#10B981;">🤖</div>
                <div class="message-content">
                    <div class="message-meta"><b>FAQ Bot (English Answer):</b> <span>10:31 AM</span></div>
                    <div class="bubble-bot" style="padding:6px 10px; font-size:12.5px; background-color:#ECFDF5; color:#065F46;">To reset your password, click on "Forgot Password" on the login page and follow the instructions.</div>
                </div>
            </div>
            <div style="font-size:12.5px; background-color:#EFF6FF; border:1px dashed #BFDBFE; padding:8px; border-radius:6px; color:#1E40AF; margin-left:42px;">
                <b>Translated (Hindi):</b><br>अपना पासवर्ड रीसेट करने के लिए, लॉगिन पेज पर "Forgot Password" पर क्लिक करें और निर्देशों का पालन करें।
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 3: Fixed No Relevant Answer Example
        st.markdown(f"""
        <div class="example-card fallback-card">
            <div class="example-title fallback-title">Example 3 – No Relevant Answer</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>10:32 AM</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">Do you have a store in New York?</div>
                </div>
            </div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon bot-avatar" style="width:30px; height:30px; font-size:12px; background-color:#10B981;">🤖</div>
                <div class="message-content">
                    <div class="message-meta"><b>FAQ Bot</b> <span>10:32 AM</span></div>
                    <div class="bubble-bot" style="padding:6px 10px; font-size:12.5px; background-color:#FFFBEB; color:#92400E; border-color:#FDE68A;">I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or contact our support team.</div>
                </div>
            </div>
            <div style="font-size:11px; color:#92400E; font-weight:500; margin-left:42px; margin-top: 5px;">
                Confidence: <b>0.23</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 4: Fixed Text Summarization Example
        st.markdown(f"""
        <div class="example-card summarize-card">
            <div class="example-title summarize-title">Example 4 – Text Summarization</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px; background-color:#F3F4F6; color:#374151; border-color:#E5E7EB;">You: (Long conversation history)</div>
                </div>
            </div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon bot-avatar" style="width:30px; height:30px; font-size:12px; background-color:#10B981;">🤖</div>
                <div class="message-content">
                    <div class="message-meta"><b>FAQ Bot (Summary):</b></div>
                    <div class="bubble-bot" style="padding:6px 10px; font-size:12.5px; background-color:#ECFDF5; color:#065F46;">
                        You wanted to know about resetting your password, refund status, and contacting support.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 1: Sentiment Analysis (Live)
        s_output = NLPAdvancedTasks.analyze_sentiment(st.session_state.last_query)
        tag_bg = "#FEE2E2" if s_output['sentiment'] == "NEGATIVE" else "#D1FAE5" if s_output['sentiment'] == "POSITIVE" else "#F3F4F6"
        tag_color = "#991B1B" if s_output['sentiment'] == "NEGATIVE" else "#065F46" if s_output['sentiment'] == "POSITIVE" else "#374151"
        
        st.markdown(f"""
        <div class="example-card sentiment-card">
            <div class="example-title sentiment-title">Example 1 – Live Sentiment Analysis</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>{st.session_state.last_time}</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">{st.session_state.last_query}</div>
                </div>
            </div>
            <div style="font-size:11px; color:#6B7280; font-weight:500; margin-left:42px; margin-top: 10px;">
                Detected Sentiment: <span style="background-color:{tag_bg}; color:{tag_color}; padding:2px 6px; border-radius:4px; font-weight:bold;">{s_output['sentiment']}</span> (Score: {s_output['score']})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 2: Multilingual Hindi Translation (Live)
        t_output = NLPAdvancedTasks.translate_text(st.session_state.last_query)
        st.markdown(f"""
        <div class="example-card translation-card">
            <div class="example-title translation-title">Example 2 – Live Translation (Hindi)</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>{st.session_state.last_time}</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">{st.session_state.last_query}</div>
                </div>
            </div>
            <div style="font-size:12.5px; background-color:#EFF6FF; border:1px dashed #BFDBFE; padding:8px; border-radius:6px; color:#1E40AF; margin-left:42px;">
                <b>Live Translated Version:</b><br>{t_output['translated_answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 3: Match Quality Check
        st.markdown(f"""
        <div class="example-card fallback-card">
            <div class="example-title fallback-title">Example 3 – Match Quality Check</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon user-avatar" style="width:30px; height:30px; font-size:12px;">👤</div>
                <div class="message-content">
                    <div class="message-meta"><b>You</b> <span>{st.session_state.last_time}</span></div>
                    <div class="bubble-user" style="padding:6px 10px; font-size:12.5px;">{st.session_state.last_query}</div>
                </div>
            </div>
            <div style="font-size:11px; color:#92400E; font-weight:500; margin-left:42px;">
                Confidence Score: <b>{st.session_state.last_confidence}</b> 
                {'(High Quality Match)' if st.session_state.last_confidence >= 0.40 else '(Fallback Mode Activated - Low Match Score)'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Example 4: Summarization (Live history summary)
        all_chats_text = " ".join([c['question'] for c in st.session_state.chat_log])
        s_summary = NLPAdvancedTasks.summarize_conversation(all_chats_text)
            
        st.markdown(f"""
        <div class="example-card summarize-card">
            <div class="example-title summarize-title">Example 4 – Live Chat Summarization</div>
            <div class="message-row" style="padding: 0; margin-bottom: 8px;">
                <div class="avatar-icon bot-avatar" style="width:30px; height:30px; font-size:12px; background-color:#10B981;">📝</div>
                <div class="message-content">
                    <div class="message-meta"><b>FAQ Bot (Live Summary)</b> <span>Now</span></div>
                    <div class="bubble-bot" style="padding:6px 10px; font-size:12.5px; background-color:#F0FDF4; border-color:#D1FAE5; color:#065F46;">
                        {s_summary}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- SYSTEM FOOTER INFO BANNER ----------------
    st.markdown("""
    <div class="footer-banner">
        <div class="footer-icon">💡</div>
        <div style="font-size:13.5px; color:#78350F; font-family:'Inter', sans-serif;">
            This FAQ Bot uses NLP techniques like <b>Embeddings</b>, <b>Semantic Search</b>, <b>Sentiment Analysis</b>, <b>Translation</b>, and <b>Summarization</b> to provide smart and context-aware answers.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Initial dynamic helper box when page loads completely blank
    st.info("👋 Hello! Enter your question in the box above to activate the FAQ Chatbot & NLP Analysis.")
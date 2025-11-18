import json
import uuid
import streamlit as st
from langchain_core.messages.tool import ToolMessage
#from src.session_state import init_session_state

from src.nodes import *
from src.workflow import create_unified_workflow

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


# ===============================
# 🎨 PAGE CONFIGURATION
# ===============================

def set_page_config():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Talk to me",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded"
    )


# ===============================
# 🎨 CUSTOM CSS
# ===============================

def set_page_style():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
            /* Main background - full page */
        .stApp {
            background: linear-gradient(135deg, #d4edf6 0%, #b2f2c3 100%);
        }
        
        /* Create a centered container with limited width */
        section[data-testid="stAppViewContainer"] > div:first-child {
            max-width: 900px;
            margin: 0 auto;
            background: linear-gradient(135deg, #d4edf6 0%, #b2f2c3 100%);
        }
        
        /* Main content area */
        .main .block-container {
            max-width: 900px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        /* Chat container - limit width and center it */
        [data-testid="stChatMessageContainer"] {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        
        /* Chat messages */
        .stChatMessage {
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* User messages */
        [data-testid="stChatMessageContent-user"] {
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Bot messages - Different color with shadow */
        [data-testid="stChatMessageContent-assistant"] {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 4px solid #66bb6a;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Chat input area */
        [data-testid="stChatInput"] {
            max-width: 900px;
            margin: 0 auto;
        }
        
        /* Phase indicator */
        .phase-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        .phase-conversation {
            background: linear-gradient(135deg, #c4f0ed 0%, #bef4d5 100%);
            color: #2d5f5d;
        }
        
        .phase-questionnaire {
            background: linear-gradient(135deg, #c9f4e4 0%, #b2f2c3 100%);
            color: #2d5f5d;
        }
        
        .phase-recommendations {
            background: linear-gradient(135deg, #bef4d5 0%, #b2f2c3 100%);
            color: #2d5f5d;
        }
        
        /* Sidebar styling */
        .sidebar-footer {
            position: fixed;
            bottom: 0;
            padding: 1rem;
            font-size: 0.8rem;
            color: #666;
            text-align: center;
        }
        
        /* Progress bar custom */
        .stProgress > div > div {
            background: linear-gradient(135deg, #c9f4e4 0%, #b2f2c3 100%);
        }
        
        /* Assessment card */
        .assessment-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        .score-display {
            font-size: 2rem;
            font-weight: bold;
            color: #2d5f5d;
            text-align: center;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


# ===============================
# 📊 SESSION STATE INITIALIZATION
# ===============================
app=create_unified_workflow()

def initialize_session_state():
    """Initialize all session state variables."""
    
    # Core identification
    if "student_id" not in st.session_state:
        st.session_state.student_id = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Messages and conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Workflow phase tracking
    if "phase" not in st.session_state:
        st.session_state.phase = "conversation"  # conversation, questionnaire, recommendations
    # Classification results
    if "disorder" not in st.session_state:
        st.session_state.disorder = None
    
    if "classified" not in st.session_state:
        st.session_state.classified = False
    
    # Questionnaire state
    if "questionnaire_started" not in st.session_state:
        st.session_state.questionnaire_started = False
    
    if "current_question_id" not in st.session_state:
        st.session_state.current_question_id = None
    
    if "questions_answered" not in st.session_state:
        st.session_state.questions_answered = 0
    
    if "questionnaire_complete" not in st.session_state:
        st.session_state.questionnaire_complete = False
    
    # Assessment results
    if "total_score" not in st.session_state:
        st.session_state.total_score = None
    
    if "score_label" not in st.session_state:
        st.session_state.score_label = None
    
    if "severity" not in st.session_state:
        st.session_state.severity = None
    
    # Recommendations
    if "route" not in st.session_state:
        st.session_state.route = None
    
    if "recommendation_generated" not in st.session_state:
        st.session_state.recommendation_generated = False
    
    # Appointment booking
    if "appointment_mode" not in st.session_state:
        st.session_state.appointment_mode = False
    
    if "appointment_confirmed" not in st.session_state:
        st.session_state.appointment_confirmed = False
    
    # Workflow state (for graph continuity)
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None


# ===============================
# 📱 SIDEBAR SETUP
# ===============================

def setup_sidebar():
    """Configure the sidebar with controls and status."""
    with st.sidebar:
        st.title("🧠 Talk to me")
        
        # Status indicator
        if st.session_state.student_id:
            st.success(f"🟢 Active Session")
            st.info(f"**Student ID:** {st.session_state.student_id}")
        else:
            st.warning("⚪ No Active Session")
        
        st.markdown("---")
        
        # Phase indicator
        st.markdown("### You matter. Your story matters..")
        phase = st.session_state.phase
        
        if phase == "conversation":
            st.markdown('<span class="phase-badge phase-conversation">💬I am right here, listening</span>', 
                       unsafe_allow_html=True)
            st.caption("It's okay to not be okay right now")
        
        elif phase == "questionnaire":
            st.markdown('<span class="phase-badge phase-questionnaire">💬You are not a burden</span>', 
                       unsafe_allow_html=True)
            # progress = st.session_state.questions_answered / 10
            # st.progress(progress)
            st.caption("Be gentle with yourself today")
        
        elif phase == "recommendations":
            st.markdown('<span class="phase-badge phase-recommendations">💬There is no "right" way to heal.</span>', 
                       unsafe_allow_html=True)
            st.caption("One small step at a time is still progress")
        
        st.markdown("---")
        
        # Results display (if available)
        if st.session_state.total_score is not None:
            st.markdown("Your feelings are important")
            st.caption("You are doing your best in a difficult moment, and that really matters.")
        #     st.metric("PSS Score", f"{st.session_state.total_score}/40")
        #     st.metric("Stress Level", st.session_state.score_label)
        #     st.metric("Severity", st.session_state.severity)
        
        
        # Action buttons
        
        if st.button("Start New Session", use_container_width=True):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        if st.session_state.phase == "recommendations" and st.session_state.route == "appointment":
            if st.button(" Book Appointment", use_container_width=True, 
                        disabled=st.session_state.appointment_confirmed):
                st.session_state.appointment_mode = True
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
            <div class="sidebar-footer">
                <small>💚 Your mental health matters</small><br>
                <small>This is not a substitute for professional care</small>
            </div>
        """, unsafe_allow_html=True)


# ===============================
# 💬 CHAT DISPLAY
# ===============================

def display_chat_history():
    """Display the complete chat history."""
    
    if not st.session_state.messages:
        st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h1>Welcome to Mental Health Support</h1>
                
                <p style='color: #888;'>
                    I'm here to listen. Whatever you're comfortable sharing, we can work through it together.
                    Tell me what's happening. I'm here to help you find your next steps.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Display all messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)


# ===============================
# 🔄 MESSAGE PROCESSING
# ===============================

def process_conversation_phase(user_input: str):
    """Process messages during the conversation/classification phase."""
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Prepare state for graph
    if st.session_state.workflow_state is None:
        state = {
            "student_id": st.session_state.student_id,
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages
        }
    else:
        state = st.session_state.workflow_state
        state["messages"] = st.session_state.messages
    
    try:
        # Invoke the graph (Phase 1: Conversation)
        result = app.invoke(state)
        
        # Update workflow state
        st.session_state.workflow_state = result
        
        # Extract AI response
        if result.get('messages'):
            last_message = result['messages'][-1]
            if isinstance(last_message, AIMessage) and not result.get('disorder'):
                st.session_state.messages.append(last_message)
        
        # Check if disorder was classified
        if result.get('disorder'):
            st.session_state.disorder = result['disorder']
            st.session_state.classified = True
            st.session_state.phase = "questionnaire"
            
            # Trigger questionnaire creation
            questionnaire_result = create_questionnaire(result)
            st.session_state.workflow_state = questionnaire_result
            
            if questionnaire_result.get('messages'):
                st.session_state.messages.append(questionnaire_result['messages'][-1])
            
            st.session_state.questionnaire_started = True
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")


def process_questionnaire_phase(user_input: str):
    """Process answers during the questionnaire phase."""
    
    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    state = st.session_state.workflow_state
    # state['messages']=state.get('messages',[])+[HumanMessage(content=user_input)]
    state['messages'].append(HumanMessage(content=user_input))
    
    try:
        # Score the answer
        scored = score_user_answer(state)
        
        # Save the score
        saved = save_answer_score(scored)
        st.session_state.workflow_state = saved
        
        # Increment counter
        st.session_state.questions_answered += 1
        
        # Check if complete
        if saved.get('next_node') == 'total_score_label':
            st.session_state.questionnaire_complete = True
            
            # Calculate total
            final_assessment = total_score_label(saved)
            st.session_state.workflow_state = final_assessment
            
            # Store results
            st.session_state.total_score = final_assessment.get('total_score')
            st.session_state.score_label = final_assessment.get('score_label')
            st.session_state.severity = final_assessment.get('severity')
            
            # Add assessment message
            if final_assessment.get('messages'):
                st.session_state.messages.append(final_assessment['messages'][-1])
            
            # Move to recommendations
            st.session_state.phase = "recommendations"
            
            # Generate recommendations
            generate_recommendations()
        
        else:
            # Next question
            if saved.get('messages'):
                st.session_state.messages.append(saved['messages'][-1])
    
    except Exception as e:
        st.error(f"Error processing answer: {str(e)}")


def generate_recommendations():
    """Generate recommendations based on assessment."""
    
    state = st.session_state.workflow_state
    
    try:
        # Transition to recommendations
        rec_state = transition_to_recommendations(state)
        
        # Determine route
        routed = determine_route(rec_state)
        st.session_state.route = routed.get('route')
        
        # Generate appropriate recommendation
        if st.session_state.route == "treatment_plan":
            final = generate_treatment_plan(routed)
            #recommendation=final.get('recommendation')
        else:
            final = generate_appointment_recommendation(routed)
            st.session_state.appointment_mode = True
        
        st.session_state.workflow_state = final
        
        # Add recommendation message
        if final.get('messages'):
            st.session_state.messages.append(final['messages'][-1])
        
        st.session_state.recommendation_generated = True
    
    except Exception as e:  
        st.error(f"Error generating recommendations: {str(e)}")


def process_appointment_interaction(user_input: str):
    """Handle appointment booking interactions."""
    
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    state = st.session_state.workflow_state
    state['user_message'] = user_input
    
    try:
        # Process through appointment handler
        result = handle_appointment_interaction(state)
        st.session_state.workflow_state = result
        
        # Add response
        if result.get('messages'):
            st.session_state.messages.append(result['messages'][-1])
        
        # Check if confirmed
        if result.get('appointment_confirmed'):
            st.session_state.appointment_confirmed = True
            st.balloons()
    
    except Exception as e:
        st.error(f"Error processing appointment: {str(e)}")


# ===============================
# 🎯 MAIN APPLICATION
# ===============================

def main():
    """Main application entry point."""
    
    set_page_config()
    set_page_style()
    initialize_session_state()
    setup_sidebar()
    
    # Student ID input (if not set)
    if st.session_state.student_id is None:
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h2>Break the silence... </h2>
                <p>Enter your Student ID to start</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            student_id = st.text_input("Student ID", placeholder="e.g., S2099", label_visibility="collapsed")
            
            if st.button("Start", use_container_width=True, type="primary"):
                if student_id:
                    st.session_state.student_id = student_id
                  
                    # Initialize conversation
                    state = {
                        "student_id": student_id,
                        "session_id": st.session_state.session_id,
                        "messages": []
                    }
                    
                    #result = app.invoke(state)
                   
                    
                    result= start_conversation(state)

                    st.session_state.workflow_state = result
                    
                    if result.get('messages'):
                        st.session_state.messages.append(result['messages'][-1])
                    
                    st.rerun()
                else:
                    st.warning("Please enter your Student ID")
        return
    # ✅ FIX: Show welcome message AFTER student ID is set but BEFORE any messages
    if st.session_state.student_id and len(st.session_state.messages) == 1:
        # Only show when there's exactly 1 message (the greeting)
        st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h1> Welcome to Talk to me </h1>
                    <h5> The AI Mental Health Support </h3>
                <p style='color: #888;'>
                   You don't have to face anything alone. We can work together through whatever you feel comfortable sharing.
                </p>
            </div>
        """, unsafe_allow_html=True)
    # Display chat history
    display_chat_history()
    
    # Chat input (phase-dependent)
    if st.session_state.phase == "conversation":
        placeholder = "Share what's on your mind..."
    elif st.session_state.phase == "questionnaire":
        placeholder = "Type your answer (e.g., 'sometimes', 'very often')..."
    else:
        if st.session_state.phase == "recommendations":
            if st.session_state.route == "appointment":
                if st.session_state.appointment_confirmed:
                    placeholder = "Your appointment is confirmed! Any other questions?"
                else:
                    placeholder = "Type 'yes' to confirm, or ask about other times..."
            else:
                placeholder = "Any questions about your treatment plan?"
    
    
    # Chat input
    if prompt := st.chat_input(placeholder):
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.spinner("Thinking..."):
            # Route to appropriate handler
            if st.session_state.phase == "conversation":
                process_conversation_phase(prompt)
            
            elif st.session_state.phase == "questionnaire":
                process_questionnaire_phase(prompt)
            
            elif st.session_state.phase == "recommendations":
                #st.session_state.recommendation_generated = True
                    
                if st.session_state.route == "appointment":
                     # Handle appointment booking interactions
                    if not st.session_state.appointment_confirmed:
                        process_appointment_interaction(prompt)
                    else:
                        # Appointment already confirmed, just acknowledge
                        st.session_state.messages.append(HumanMessage(content=prompt))
                        st.session_state.messages.append(AIMessage(
                            content="Your appointment is all set! If you need to make changes, please contact the counseling center. Is there anything else I can help you with?"
                        ))

                elif st.session_state.route == "treatment_plan":
                     # Treatment plan already shown, handle follow-up questions
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    st.session_state.messages.append(AIMessage(
                        content="Thank you for your question. Remember, the self-care strategies I shared are meant to complement professional support if needed. If your symptoms persist or worsen, please don't hesitate to reach out to a mental health professional. Is there anything specific about the treatment plan you'd like me to clarify?"
                    ))
                else:
                    # General follow-up conversation
                    
                    st.session_state.messages.append(HumanMessage(content=prompt))
                    st.session_state.messages.append(AIMessage(
                        content="Thank you for sharing. Is there anything else I can help you with today?"
                    ))
        
        st.rerun()


# ===============================
# 🚀 RUN THE APP
# ===============================

if __name__ == "__main__":
    main()

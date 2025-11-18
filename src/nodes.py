from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.models import UnifiedState, Feedback
from src.helperfunctions import *
from src.tools import *
from typing import Literal
from datetime import datetime, timezone
from pydantic import BaseModel

llm = ChatOpenAI(temperature=0.5)
## Graph 1 Nodes
def start_conversation(state: UnifiedState) -> UnifiedState:
    """Starts the conversation with a welcoming message."""
    greeting = AIMessage(content="Hello, I'm here to listen and support you. This is a safe space to share what's on your mind. How are you feeling today?")
    return {
        **state,
        "messages": [greeting],
        "iterator": 0,
        "rag_context": None,
        "workflow_stage": "conversation"
    }

def track_conversation(state: UnifiedState) -> UnifiedState:
    """Tracks the number of user inputs."""
    human_message_count = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage))
    return {**state, "iterator": human_message_count}

def retrieve_context(state: UnifiedState) -> UnifiedState:
    """Let the LLM decide whether to call RAG tool based on user message."""
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return {**state, "rag_context": None}

    last_user_message = user_messages[-1].content

    try:
        system_prompt = """You are a mental health therapist assistant.
Use the RAG tool to retrieve relevant mental health information from the knowledge base
when the user mentions symptoms, feelings, or concerns that could benefit from evidence-based context."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User said: {last_user_message}\n\nWhat information from the knowledge base would be helpful?")
        ]

        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]
            if isinstance(args, dict):
                search_query = args.get("query") or args.get("input") or last_user_message
            else:
                search_query = str(args)

            retrieved_docs = rag.invoke({"query": search_query, "k": 5})
            context = format_documents(retrieved_docs)
            print(f"✓ RAG retrieved for query: {search_query}")
            return {**state, "rag_context": context}
        else:
            return {**state, "rag_context": None}

    except Exception as e:
        print(f"⚠️  RAG retrieval error: {str(e)}")
        return {**state, "rag_context": None}
    

def generate_response(state: UnifiedState) -> UnifiedState:
    """Generates an empathetic, therapeutic response using RAG context."""
    system_prompt = """You are a compassionate and professional mental health therapist.
Your role is to:
- Listen actively and empathetically to the user
- Ask thoughtful, open-ended questions to understand their feelings
- Validate their emotions and experiences
- Keep responses conversational and supportive (2-4 sentences)

Focus on understanding if they show signs of:
- Anxiety: excessive worry, nervousness, panic, physical symptoms
- Depression: persistent sadness, loss of interest, hopelessness, fatigue
- Stress: feeling overwhelmed, tension, difficulty coping with demands
"""

    if state.get("rag_context"):
        system_prompt += f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['rag_context']}\n\nUse this context to provide informed, evidence-based support while maintaining a conversational tone."

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    return {**state, "messages": [response]}

def classify_disorder(state: UnifiedState) -> UnifiedState:
    """Analyzes conversation and classifies the disorder using RAG for ground truth."""
    conversation_summary = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Therapist'}: {msg.content}"
        for msg in state["messages"]
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    try:
        search_query = f"diagnostic criteria for stress and anxeity and depression mental health assessment"

        diagnostic_prompt = f"""Use the RAG tool to retrieve mental health diagnostic criteria.
Search for: {search_query}

This will help classify the following conversation:
{conversation_summary[:500]}"""

        messages = [
            SystemMessage(content="You are analyzing a mental health conversation for classification. Call the RAG tool with a search query string."),
            HumanMessage(content=diagnostic_prompt)
        ]

        response = llm_with_tools.invoke(messages)

        diagnostic_context = ""
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]
            if isinstance(args, dict):
                query = args.get("query") or args.get("input") or search_query
            else:
                query = str(args)

            classification_docs = rag.invoke({"query": query, "k": 5})
            diagnostic_context = format_documents(classification_docs)
            print(f"✓ Classification RAG retrieved for: {query}")

    except Exception as e:
        print(f"⚠️  RAG retrieval error during classification: {str(e)}")
        diagnostic_context = ""

    classification_prompt = f"""Based on the entire conversation history, analyze the user's mental health concerns.

{diagnostic_context}

the user chat history is
{conversation_summary}

Look for these key symptoms:
**Anxiety:** Excessive worry, restlessness, racing thoughts, panic attacks, avoidance behaviors, physical tension
**Depression:** Persistent sadness, loss of interest/pleasure, hopelessness, fatigue, sleep changes, appetite changes, worthlessness
**Stress:** Feeling overwhelmed, inability to cope, irritability, concentration problems, burnout

Classify the PRIMARY concern as anxiety, depression, or stress based on which symptom cluster dominates."""

    messages = state["messages"] + [HumanMessage(content=classification_prompt)]
    result: Feedback = llm_structured.invoke(messages)

    type_mapping = {
        "anxiety": "GAD",
        "depression": "PHQ",
        "stress": "PSS"
    }

    try:
        supabase.table("student_questionnaire_results").insert({
            "student_id": state["student_id"],
            "type": type_mapping.get(result.disorder.lower(), "PSS")
        }).execute()
        print("✅ Inserted classification result into Supabase.")
    except Exception as e:
        print(f"⚠️ Supabase insert failed: {e}")

    return {
        **state,
        "disorder": result.disorder,
        "condition": result.disorder,
        #"messages": [AIMessage(content=f"Based on our conversation and clinical evidence, I've identified your primary concern as {result.disorder}.\n\n{result.reasoning}\n\nTo better understand your situation, I'd like to ask you a few questions. This will help me provide more personalized support.")],
        "workflow_stage": "classified"
    }




## Graph 2 Nodes
# TRANSITION 1: Classification → Questionnaire
# ============================================

def transition_to_questionnaire(state: UnifiedState) -> UnifiedState:
    """
    BRIDGE NODE 1: Connects classification to questionnaire.
    Maps disorder to questionnaire type and preserves messages.
    """
    disorder = state.get("disorder", "stress")
    student_id = state.get("student_id")

    print(f"\n🔄 Transitioning from classification to questionnaire...")
    print(f"   Disorder identified: {disorder}")
    print(f"   Student ID: {student_id}")
    print(f"   Messages in state: {len(state.get('messages', []))}")
    print(f"   Starting questionnaire assessment...\n")

    return {
        **state,
        "disorder": disorder,
        "workflow_stage": "questionnaire"
    }

# MODIFIED GRAPH 2 FUNCTIONS - Using UnifiedState with Messages
# ============================================

def create_questionnaire(state: UnifiedState) -> UnifiedState:
    """Initialize or resume questionnaire - supports PSS, PHQ-9, and GAD-7"""
    disorder = state.get('disorder', 'stress')
    student_id = state.get('student_id')

    print("\n" + "="*50)
    print("INITIALIZING QUESTIONNAIRE")
    print("="*50)
    print(f"Disorder: {disorder}")
    print(f"Student ID: {student_id}")

    # Get questionnaire config
    if disorder not in QUESTIONNAIRES:
        error_text = f"Unknown disorder type: {disorder}"
        return {
            **state,
            'messages': state.get('messages', []) + [AIMessage(content=error_text)],
            'next_node': 'end'
        }

    config = QUESTIONNAIRES[disorder]
    questionnaire_type = config['type']
    questionnaire_name = config['name']
    questions = config['questions']

    print(f"Questionnaire Type: {questionnaire_type}")
    print(f"Questionnaire Name: {questionnaire_name}")
    print(f"Number of Questions: {len(questions)}\n")

    try:
        # Reword all questions
        reword_questionnaire = {}
        print('Rewording questions for natural conversation...\n')

        for question_id, question in questions.items():
            reword_question = questionnaire_reword_chain.invoke({"question": question}).content
            question_key = f'{questionnaire_type}{question_id}'
            reword_questionnaire[question_key] = reword_question

        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if record exists
        exists = supabase.table('student_questionnaire_results').select('*').eq('student_id', student_id).execute()

        if not exists.data:
            # Create new record
            new_record = {
                'student_id': student_id,
                'timestamp': timestamp,
                'type': questionnaire_type
            }

            # Initialize all question fields to None
            for i in range(1, len(questions) + 1):
                new_record[f'{questionnaire_type}{i}'] = None

            supabase.table('student_questionnaire_results').insert(new_record).execute()
            print(f"✓ New record created for student {student_id}\n")

            first_question_key = f'{questionnaire_type}1'
            response_text = f"{reword_questionnaire[first_question_key]}"
            #I'm here to help you sort through your thoughts and find a path that feels right for you, you can share as much or as little as you like. We'll go at your pace.
            return {
                **state,
                'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                'current_question_id': first_question_key,
                'reword_questionnaire': reword_questionnaire,
                'questionnaire_config': config,
                'next_node': 'ask_question'
            }
        else:
            # Record exists - find first unanswered
            print(f"✓ Found existing record for student {student_id}")
            record = exists.data[0]

            first_unanswered = None
            answered_count = 0

            for i in range(1, len(questions) + 1):
                question_key = f'{questionnaire_type}{i}'
                if record.get(question_key) is not None:
                    answered_count += 1
                elif first_unanswered is None:
                    first_unanswered = question_key

            if first_unanswered is None:
                print("✓ All questions already answered!\n")
                response_text = "You don't have to have it all figured out right now, we can work through it together..."

                return {
                    **state,
                    'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                    'reword_questionnaire': reword_questionnaire,
                    'questionnaire_config': config,
                    'next_node': 'total_score_label'
                }
            else:
                print(f"✓ Resuming from question {first_unanswered} ({answered_count}/{len(questions)} completed)\n")
                response_text = f"{reword_questionnaire[first_unanswered]}"
                #I'm here to listen. Whatever you're comfortable sharing, we can work through it together.
                return {
                    **state,
                    'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                    'current_question_id': first_unanswered,
                    'reword_questionnaire': reword_questionnaire,
                    'questionnaire_config': config,
                    'next_node': 'ask_question'
                }

    except Exception as e:
        error_text = f"❌ Error creating questionnaire: {e}"
        return {
            **state,
            'messages': state.get('messages', []) + [AIMessage(content=error_text)],
            'next_node': 'end'
        }




def ask_question_node(state: UnifiedState) -> UnifiedState:
    """Display question and signal that we need user input"""
    return {
        **state,
    }


def score_user_answer(state: UnifiedState) -> UnifiedState:
    """
    Hybrid scoring: Try keyword matching first, fall back to LLM if needed.
    Works for PSS, PHQ-9, and GAD-7.
    """
    question_id = state.get("current_question_id")
    config = state.get("questionnaire_config", QUESTIONNAIRES['stress'])

    # Get answer
    answer = state.get("user_answer", "")
    if not answer:
        human_messages = [msg for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]
        if human_messages:
            answer = human_messages[-1].content

    answer = answer.lower().strip()

    print("\n" + "-"*50)
    print(f"SCORING ANSWER FOR {question_id}")
    print("-"*50)
    print(f"Answer: '{answer}'\n")

    try:
        # Extract question number from ID (e.g., 'pss7' -> 7, 'phq3' -> 3)
        question_num = int(''.join(filter(str.isdigit, question_id)))
        scale_type = config.get('scale_type', '0-4')
        is_reverse_scoring = question_num in config.get('reverse_scoring', [])

        # STEP 1: Try keyword matching first (fastest & most reliable)
        # PSS uses 0-4 scale, PHQ/GAD use 0-3 scale
        if scale_type == '0-3':
            # PHQ-9 and GAD-7 scoring (no reverse scoring)
            keyword_patterns = {
                0: ['never', 'not at all', 'no'],
                1: ['several days', 'rarely', 'a few days'],
                2: ['more than half the days', 'half the days', 'often'],
                3: ['nearly every day', 'almost every day', 'always']
            }
        else:
            # PSS scoring (0-4 scale with possible reverse scoring)
            keyword_patterns = {
                0: ['never', 'not at all', 'no'],
                1: ['almost never', 'rarely', 'seldom', 'hardly'],
                2: ['sometimes', 'occasionally', 'once in a while'],
                3: ['fairly often', 'often', 'frequently', 'regularly'],
                4: ['very often', 'always', 'constantly', 'all the time']
            }

        matched_score = None
        for base_score, keywords in keyword_patterns.items():
            if any(keyword in answer for keyword in keywords):
                matched_score = base_score
                print(f"✓ Keyword match found: '{answer}' → base score {base_score}")
                break

        if matched_score is not None:
            # Apply reverse scoring ONLY for PSS (if applicable)
            final_score = (4 - matched_score) if is_reverse_scoring else matched_score

            # Cap score at 3 for PHQ/GAD scales
            if scale_type == '0-3' and final_score > 3:
                final_score = 3

            print(f"✓ Scoring type: {'REVERSE' if is_reverse_scoring else 'DIRECT'}")
            print(f"✓ Scale: {scale_type}")
            print(f"✓ Final score: {final_score}\n")

            return {
                **state,
                "score": final_score,
                "next_node": "save_score"
            }

        # STEP 2: No keyword match - use LLM
        print(f"⚠️ No exact keyword match, using LLM...")

        Score_instructions = f"""
Score this response for a mental health questionnaire.

Questionnaire type: {config['name']}
Scale: {scale_type}
Question {question_num} uses {"REVERSE" if is_reverse_scoring else "DIRECT"} scoring.

Scoring guidelines:
- For PSS (0-4 scale): "never"=0, "almost never"=1, "sometimes"=2, "fairly often"=3, "very often"=4
  - Apply reverse scoring for questions 7-10 only (flip the score: 0→4, 1→3, 2→2, 3→1, 4→0)
- For PHQ-9/GAD-7 (0-3 scale): "not at all"=0, "several days"=1, "more than half the days"=2, "nearly every day"=3
  - NO reverse scoring

Response: "{answer}"

Return the numeric score and brief reasoning.
"""

        class ScoreResponse(BaseModel):
            score: int
            reasoning: str

        response_score = llm.with_structured_output(ScoreResponse).invoke([
            SystemMessage(content=Score_instructions),
            HumanMessage(content=answer)
        ])

        score = response_score.score if hasattr(response_score, 'score') else 2

        # Validate score range
        max_score_value = 3 if scale_type == '0-3' else 4
        if score > max_score_value:
            score = max_score_value
        if score < 0:
            score = 0

        print(f"✓ LLM score: {score}")
        print(f"   Scale: {scale_type}")
        print(f"   Reasoning: {response_score.reasoning if hasattr(response_score, 'reasoning') else 'N/A'}\n")

        return {
            **state,
            "score": score,
            "next_node": "save_score"
        }

    except Exception as e:
        print(f"❌ Error scoring: {e}\n")
        return {
            **state,
            "messages": state.get('messages', []) + [AIMessage(content="Could you rephrase that?")],
            "score": 2,  # Safe default
            "next_node": "save_score"
        }



def save_answer_score(state: UnifiedState) -> UnifiedState:
    """Save score and prepare next question - works for all questionnaire types"""
    student_id = state.get("student_id")
    question_id = state.get("current_question_id")
    score = state.get("score")
    reword_questionnaire = state.get("reword_questionnaire", {})
    config = state.get("questionnaire_config", QUESTIONNAIRES['stress'])
    disorder = state.get("disorder", "stress")  # Get disorder from state

    print("\n" + "-"*50)
    print(f"SAVING SCORE FOR {question_id}")
    print("-"*50)

    try:
        supabase.table("student_questionnaire_results").update(
            {question_id: score}
        ).eq("student_id", student_id).execute()

        print(f"✓ Score {score} saved successfully\n")

        exists = supabase.table("student_questionnaire_results").select("*").eq("student_id", student_id).execute()

        if not exists.data:
            error_text = "Error: Record not found"
            return {
                **state,
                "messages": state.get('messages', []) + [AIMessage(content=error_text)],
                "next_node": "end"
            }

        record = exists.data[0]
        questionnaire_type = config['type']
        total_questions = len(config['questions'])
        current_num = int(''.join(filter(str.isdigit, question_id)))

        next_unanswered = None
        answered_count = 0

        for i in range(1, total_questions + 1):
            key = f'{questionnaire_type}{i}'
            if record.get(key) is not None:
                answered_count += 1
            elif next_unanswered is None and i > current_num:
                next_unanswered = key

        print(f"Progress: {answered_count}/{total_questions} questions completed")

        if next_unanswered:
            acknowledgments = [
                #"would you mind to share a bit about",
                #"I appreciate you telling me that. Do you wanna tell me more about",
                "I am wondering",
                "It takes strength to be so open. So,",
                "thank you for sharing. Just to understand you better,",
                "You are allowed to feel what you feel."
            ]
            import random
            ack = random.choice(acknowledgments)

            print(f"Moving to next question: {next_unanswered}\n")

            response_text = f"{ack} {reword_questionnaire[next_unanswered]}".strip()

            return {
                **state,
                "score": score,
                "messages": state.get('messages', []) + [AIMessage(content=response_text)],
                "current_question_id": next_unanswered,
                "reword_questionnaire": reword_questionnaire,
                "questionnaire_config": config,
                "disorder": disorder,  # Preserve disorder
                "next_node": "ask_question",
            }
        else:
            print("✓ All questions completed!\n")
            response_text = "Thank you for letting me in. I can only imagine how that feels..."

            return {
                **state,
                "score": score,
                "messages": state.get('messages', []) + [AIMessage(content=response_text)],
                "current_question_id": None,
                "questionnaire_config": config,
                "disorder": disorder,  # Preserve disorder
                "next_node": "total_score_label",
            }

    except Exception as e:
        print(f"❌ Error saving: {e}\n")
        error_text = f"Error saving your answer: {e}"

        return {
            **state,
            "messages": state.get('messages', []) + [AIMessage(content=error_text)],
            "next_node": "end"
        }





def total_score_label(state: UnifiedState) -> UnifiedState:
    """Calculate total score and provide assessment - works for all questionnaire types"""
    print("\n" + "="*50)
    print("CALCULATING FINAL RESULTS")
    print("="*50)

    student_id = state.get('student_id')
    disorder = state.get('disorder', 'stress')

    # Get config - first try from state, then fall back to disorder mapping
    config = state.get('questionnaire_config')
    if config is None:
        print(f"⚠️ No questionnaire_config in state, using disorder: {disorder}")
        config = QUESTIONNAIRES.get(disorder, QUESTIONNAIRES['stress'])

    questionnaire_type = config['type']
    questionnaire_name = config['name']
    max_score = config['max_score']
    score_ranges = config['score_ranges']

    print(f"Using config for: {questionnaire_name} (type: {questionnaire_type})")
    print(f"Expected questions: {len(config['questions'])}")

    scores = []

    try:
        exists = supabase.table('student_questionnaire_results').select('*').eq('student_id', student_id).execute()

        if exists.data:
            record = exists.data[0]
            print(f"\n📊 Collecting scores from database:")
            for i in range(1, len(config['questions']) + 1):
                key = f'{questionnaire_type}{i}'
                value = record.get(key)
                print(f"  {key}: {value}")
                if value is not None:
                    scores.append(value)

        total_score = sum([x for x in scores if isinstance(x, int)])

        print(f"\n✓ Individual scores: {scores}")
        print(f"✓ Total score: {total_score}/{max_score}\n")

        # Determine score label based on ranges
        score_label = 'Unknown'
        severity = 'unknown'
        for min_score, max_score_range, label in score_ranges:
            if min_score <= total_score <= max_score_range:
                score_label = label
                severity = label.lower()
                break

        print(f"Assessment: {score_label}\n")

        # Update database with results
        update_data = {
            f'{questionnaire_type}_total_score': total_score,
            f'{questionnaire_type}_score_label': score_label
        }

        print(f"Updating database with: {update_data}")

        update_result = supabase.table('student_questionnaire_results').update(
            update_data
        ).eq('student_id', student_id).execute()

        print(f"✓ Database update result: {update_result.data}")

        # Verify the update
        verify = supabase.table('student_questionnaire_results').select('*').eq('student_id', student_id).execute()
        if verify.data:
            print(f"✓ Verification - Total score in DB: {verify.data[0].get(f'{questionnaire_type}_total_score')}")
            print(f"✓ Verification - Score label in DB: {verify.data[0].get(f'{questionnaire_type}_score_label')}")

        response_text = """ I am really glad you shared that with me, it takes courage to open up about how you are feeling. 
        You are not facing this alone; I am here with you, and I am ready to support you however I can."""

        return {
            **state,
            'total_score': total_score,
            'score_label': score_label,
            'severity': severity,
            'disorder': disorder,  # Preserve disorder
            'questionnaire_config': config,  # Preserve config
            'messages': state.get('messages', []) + [AIMessage(content=response_text)],
            'next_node': 'transition_to_recommendations'
        }

    except Exception as e:
        print(f"❌ Error calculating total: {e}\n")
        import traceback
        traceback.print_exc()
        error_text = f'Error calculating results: {e}'

        return {
            **state,
            'messages': state.get('messages', []) + [AIMessage(content=error_text)],
            'next_node': 'end'
        }




## Graph 3

## Transition node

def transition_to_recommendations(state: UnifiedState) -> UnifiedState:
    """
    NEW NODE: Bridges Graph 1 to Graph 2.
    Fetches actual condition/severity from database if student_id is available.
    Falls back to condition classification from conversation if DB fetch fails.
    """
    condition = state.get("condition", "stress")
    student_id = state.get("student_id")

    print(f"\n🔄 Transitioning from conversation to recommendations...")
    print(f"   condition identified from conversation: {condition}")

    # Try to fetch from database first
    if student_id:
        print(f"   Fetching assessment from database for student: {student_id}...")
        condition, severity = get_student_assessment_from_db(student_id)
        print(f"   ✓ Database assessment: {condition} ({severity})")
    else:
        # Fallback: Use conversation condition with default severity
        print(f"   No student_id provided, using conversation assessment...")
        severity_mapping = {
            "anxiety": "moderate anxiety",
            "depression": "moderate depression",
            "stress": "moderate stress"
        }
        condition = condition
        severity = severity_mapping.get(condition, "moderate stress")
        print(f"   ✓ Using mapped severity: {severity}")

    print(f"   Final assessment: {condition} at {severity} level\n")

    return {
        **state,
        "condition": condition,
        "severity": severity,
        "workflow_stage": "recommendation"
    }


def determine_route(state: UnifiedState) -> UnifiedState:
    """Determine whether student needs treatment plan or appointment."""
    severity = state["severity"].lower()
    route = SEVERITY_ROUTING.get(severity, "treatment_plan")
    print(f"✓ Route determined: {route}")
    return {**state, "route": route}


def generate_treatment_plan(state: UnifiedState) -> UnifiedState:
    """Generate self-care treatment plan for lower severity cases."""
    condition = state["condition"]
    severity = state["severity"]

    rag_context = retrieve_context_for_recommendation(condition, severity)

    system_prompt = f"""You are a compassionate mental health support assistant.

The person has been assessed with {severity} level {condition} based on our conversation.

{rag_context}

Based on the evidence-based guidelines above, provide personalized recommendations including:
- Self-care strategies specific to {condition}
- Stress management and coping techniques
- Lifestyle modifications (exercise, sleep hygiene, nutrition)
- Self-monitoring practices
- Warning signs to watch for
- When to seek additional professional support

Keep your response supportive, practical, and actionable (3 paragraphs).
Present all recommendations as clearly formatted bullet points using numbering (1, 2, 3…) or lettering (A, B, C…).
Do NOT diagnose or provide medical advice."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Provide a comprehensive self-care treatment plan for managing {condition}.")
    ]

    response = llm.invoke(messages)
    print(f"✓ Treatment plan generated")

    return {
        **state,
        "recommendation": response.content,
        "rag_context": rag_context,
        "messages": state.get("messages", []) + [AIMessage(content=response.content)] #added a line here

    }


def generate_appointment_recommendation(state: UnifiedState) -> UnifiedState:
    """Node 3b: Generate appointment recommendation for higher severity cases."""
    condition = state["condition"]
    severity = state["severity"]

    rag_context = retrieve_context_for_recommendation(condition, severity)
    nearest_slots = get_nearest_available_slot.invoke({})

    system_prompt = f"""You are a compassionate mental health support assistant with appointment booking capabilities.

The student has been assessed with {severity} level {condition}, which requires professional attention.

{rag_context}

NEAREST AVAILABLE APPOINTMENTS:
{nearest_slots}

Your response should:
1. Warmly explain why professional support is recommended at this severity level wiout mentioning the severity or any specific diagnosis (1 sentence)
2. Present the nearest available appointment slot clearly
3. Ask the student to confirm if they'd like to book this time, or see other options
4. Provide 2-3 immediate coping strategies they can use while waiting

IMPORTANT BOOKING RULES:
- DO NOT call book_appointment tool until the student explicitly confirms
- Wait for student to say "yes", "confirm", "book it", or similar confirmation
- If they want other options, they can ask and you'll show the alternative slots listed above
- Be warm, supportive, and patient

Present all recommendations as clearly formatted bullet points using numbering (1, 2, 3…) or lettering (A, B, C…).
Keep your response conversational and encouraging (2 paragraphs max)."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Help the student understand why an appointment is needed for {condition} and guide them to book one.")
    ]

    response = llm.invoke(messages)

    print(f"✓ Appointment recommendation with nearest slot generated")

    return {
        **state,
        "recommendation": response.content,
        "rag_context": rag_context,
        "messages": state.get("messages", []) + [AIMessage(content=response.content)] #added a line here

    }


def handle_appointment_interaction(state: UnifiedState) -> UnifiedState:
    """Node 4: Interactive appointment booking/management."""
    student_id = state["student_id"]
    user_message = state.get("user_message", "")
    previous_recommendation = state.get("recommendation", "")

    system_prompt = """You are an appointment booking assistant for mental health services.

Available tools:
- get_nearest_available_slot: Show nearest available appointments
- book_appointment: Book using appointment_id (ONLY after user confirms with "yes", "book it", "confirm", etc.)
- check_conflicts: Check for scheduling conflicts
- cancel_appointment: Cancel an existing appointment
- update_appointment: Reschedule an appointment (cancels old, shows new options)

CRITICAL BOOKING RULES:
1. User must explicitly confirm before booking (look for: "yes", "confirm", "book it", "okay", "sure", "yess", "yep")
2. To book, you need the appointment_id from the slot suggestion in the PREVIOUS RECOMMENDATION
3. If user asks for "other options" or "alternatives", show other available slots
4. If user says a specific time/date, find nearest slot to that time
5. Always be conversational and confirm what action you're taking

Example flows:
- User: "yes" → Extract appointment_id from PREVIOUS RECOMMENDATION → call book_appointment
- User: "show me other times" → get_nearest_available_slot with more options
- User: "I prefer Monday" → get_nearest_available_slot starting from Monday

IMPORTANT: When user confirms (yes/confirm/book it), look at the PREVIOUS RECOMMENDATION below to find the appointment_id."""

    # Build conversation history with previous recommendation
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"PREVIOUS RECOMMENDATION:\n{previous_recommendation}"),
        HumanMessage(content=f"USER'S RESPONSE: {user_message}")
    ]

    response = llm_with_tools_full.invoke(messages)

    tool_results = []
    booking_confirmed = False

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            if tool_name in ["book_appointment", "update_appointment"]:
                args["student_id"] = student_id

            if tool_name == "get_nearest_available_slot":
                result = get_nearest_available_slot.invoke(args)
            elif tool_name == "book_appointment":
                result = book_appointment.invoke(args)
                # Only mark as confirmed if booking was successful
                if "successfully booked" in str(result).lower() or "✓" in str(result):
                    booking_confirmed = True
            elif tool_name == "check_conflicts":
                result = check_conflicts.invoke(args)
            elif tool_name == "cancel_appointment":
                result = cancel_appointment.invoke(args)
            elif tool_name == "update_appointment":
                result = update_appointment.invoke(args)
            else:
                result = f"Unknown tool: {tool_name}"

            tool_results.append(result)

    response_text = response.content if hasattr(response, 'content') else str(response)
    full_response = response_text + "\n\n" + "\n\n".join(tool_results) if tool_results else response_text

    return {
        **state,
        "recommendation": full_response,
        "appointment_confirmed": booking_confirmed,
        "messages": state.get("messages", []) + [AIMessage(content=full_response)]
    }


def route_by_severity(state: UnifiedState) -> str:
    """Routes to treatment_plan or appointment based on severity."""
    route = state.get("route", "treatment_plan")
    return route
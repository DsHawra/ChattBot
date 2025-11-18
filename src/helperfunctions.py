from typing import List
from langchain_core.documents import Document
from src.tools import retrieve_treatment_info
from src.models import UnifiedState
from src.supabase import supabase
from langchain_core.prompts import ChatPromptTemplate
from src.tools import llm

SEVERITY_ROUTING = {
    "minimal depression": "treatment_plan",
    "mild depression": "treatment_plan",
    "minimal anxiety": "treatment_plan",
    "low stress": "treatment_plan",
    "moderate depression": "treatment_plan",
    "mild anxiety": "treatment_plan",
    "moderate stress": "treatment_plan",
    "moderately severe depression": "appointment",
    "moderate anxiety": "appointment",
    "high stress": "appointment",
    "severe depression": "appointment",
    "moderate to severe anxiety": "appointment"
}

def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into a readable context string."""
    if not docs:
        return "No relevant context found."
    context = "Retrieved Knowledge Base Context:\n\n"
    for i, doc in enumerate(docs, 1):
        context += f"[Document {i}]\n{doc.page_content}\n\n"
    return context

def should_classify(state: UnifiedState) -> str:
    """Determines whether to continue conversation or classify disorder."""
    iterator = state.get("iterator", 0)
    return "classify" if iterator >= 5 else "continue"






## Graph 2 Functions
# Questionnaire reword prompt (same for all types)
questionnaire_reword_prompt = ChatPromptTemplate.from_template(
    """You are a warm, empathetic therapist assistant having a casual conversation with a student.

Your task: Transform this formal question into a natural, caring conversational statement.

CRITICAL RULES:
1. DO NOT ask a direct question - make it sound like you're checking in on them
2. DO NOT use phrases like "I'd like to ask you" or "Let me ask you"
3. DO NOT number the question or mention it's a questionnaire
4. Keep it short and conversational (1-2 sentences max)
5. Frame it around "the last 2 weeks" for PHQ/GAD or "the last month" for PSS
6. Make it feel like a friend checking in, not a doctor diagnosing

GOOD EXAMPLES:
 BAD: "Let me ask you - in the last month, how often have you felt stressed?"
 GOOD: "Life can get pretty overwhelming sometimes. I'm curious about how things have been for you lately - have you been feeling stressed or on edge recently?"

 BAD: "I want to know how often you've felt confident."
 GOOD: "I've noticed some people have been feeling more sure of themselves lately, while others haven't. How's that been going for you over the past month?"

Now transform this question:
Question: {question}

Your conversational version (just the reworded question, nothing else):""".strip()
)

questionnaire_reword_chain = questionnaire_reword_prompt | llm


# QUESTIONNAIRE DEFINITIONS
# ============================================

QUESTIONNAIRES = {
    'stress': {
        'type': 'pss',
        'name': 'Perceived Stress Scale',
        'questions': {
            1: 'have you been upset because of something that happened unexpectedly?',
            2: 'how often have you felt that you were unable to control the important things in your life?',
            3: 'how often have you felt nervous and stressed?',
            4: 'how often have you been angered because of things that were outside of your control?',
            5: 'how often have you felt that difficulties were piling up so high that you could not overcome them?',
            6: 'how often have you found that you could not cope with all the things that you had to do?',
            7: 'how often have you felt confident about your ability to handle your personal problems?',
            8: 'how often have you felt that things were going your way?',
            9: 'how often have you been able to control irritations in your life?',
            10:'how often have you felt that you were on top of things?'
        },
        'reverse_scoring': [7, 8, 9, 10],
        'score_ranges': [
            (0, 13, 'Low stress'),
            (14, 26, 'Moderate stress'),
            (27, 40, 'High stress')
        ],
        'max_score': 40,
        'scale_type': '0-4'  # PSS uses 0-4 scale
    },
    'depression': {
        'type': 'phq',
        'name': 'Patient Health Questionnaire (PHQ-9)',
        'questions': {
            1: 'Little interest or pleasure in doing things',
            2: 'Feeling down, depressed, or hopeless',
            3: 'Trouble falling or staying asleep, or sleeping too much',
            4: 'Feeling tired or having little energy',
            5: 'Poor appetite or overeating',
            6: 'Feeling bad about yourself - or that you are a failure or have let yourself or your family down',
            7: 'Trouble concentrating on things, such as reading the newspaper or watching television',
            8: 'Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual',
            9: 'Thoughts that you would be better off dead, or of hurting yourself in some way'
        },
        'reverse_scoring': [],  # PHQ-9 has no reverse scoring
        'score_ranges': [
            (0, 4, 'Minimal depression'),
            (5, 9, 'Mild depression'),
            (10, 14, 'Moderate depression'),
            (15, 19, 'Moderately severe depression'),
            (20, 27, 'Severe depression')
        ],
        'max_score': 27,
        'scale_type': '0-3'  # PHQ-9 uses 0-3 scale
    },
    'anxiety': {
        'type': 'gad',
        'name': 'Generalized Anxiety Disorder (GAD-7)',
        'questions': {
            1: 'Feeling nervous, anxious, or on edge',
            2: 'Not being able to stop or control worrying',
            3: 'Worrying too much about different things',
            4: 'Trouble relaxing',
            5: 'Being so restless that it\'s hard to sit still',
            6: 'Becoming easily annoyed or irritable',
            7: 'Feeling afraid as if something awful might happen'
        },
        'reverse_scoring': [],  # GAD-7 has no reverse scoring
        'score_ranges': [
            (0, 4, 'Minimal anxiety'),
            (5, 9, 'Mild anxiety'),
            (10, 14, 'Moderate anxiety'),
            (15, 21, 'Severe anxiety')
        ],
        'max_score': 21,
        'scale_type': '0-3'  # GAD-7 uses 0-3 scale
    }
}


## Graph 3 Functions
## DB Helper function

def get_student_assessment_from_db(student_id: str) -> tuple[str, str]:
    """
    Get condition and severity from Supabase based on latest questionnaire.
    Returns: (condition, severity) tuple
    """
    if supabase is None:
        print("⚠️  Supabase not initialized. Using defaults.")
        return ("stress", "moderate stress")

    try:
        response = supabase.table("student_questionnaire_results") \
            .select("type, pss_score_label, phq_score_label, gad_score_label") \
            .eq("student_id", student_id) \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            print(f"⚠️  No results found for student {student_id}")
            return ("stress", "moderate stress")

        result = response.data[0]
        questionnaire_type = result["type"].upper()

        type_mapping = {
            "PSS": ("stress", result.get("pss_score_label", "moderate stress")),
            "PHQ": ("depression", result.get("phq_score_label", "moderate depression")),
            "GAD": ("anxiety", result.get("gad_score_label", "mild anxiety"))
        }

        condition, severity = type_mapping.get(
            questionnaire_type,
            ("stress", "moderate stress")
        )

        return (condition, severity or "moderate stress")

    except Exception as e:
        print(f"⚠️  Error retrieving assessment: {str(e)}")
        return ("stress", "moderate stress")

def retrieve_context_for_recommendation(condition: str, severity: str) -> str:
    """
    Helper function: Retrieve RAG context for generating recommendations.
    """
    try:
        retrieved_docs = retrieve_treatment_info.invoke({
            "condition": condition,
            "severity": severity,
            "k": 5
        })

        context = format_documents(retrieved_docs)
        print(f"✓ RAG context retrieved ({len(retrieved_docs)} documents)")
        return context

    except Exception as e:
        print(f"⚠️  RAG retrieval error: {str(e)}")
        return ""

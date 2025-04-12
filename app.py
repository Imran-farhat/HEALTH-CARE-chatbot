from flask import Flask, request, jsonify, render_template, session
import os
import random
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
import string

# Download NLTK data with explicit error handling
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Import resources to verify they're available
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    print("✓ All NLTK packages successfully loaded")
except Exception as e:
    print(f"Error loading NLTK resources: {e}")
    print("Trying alternate initialization method...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Failed to download NLTK resources: {e}")

# Initialize NLP tools with fallback mechanisms
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not initialize all NLP tools. Using simplified processing: {e}")
    # Fallback to simpler text processing if NLTK fails
    lemmatizer = None
    stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'is', 'are', 'was', 'were'])

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # For session management

# Store conversation history and patient profiles in memory
conversation_history = {}
patient_profiles = {}
consultations = {}
conversation_context = {}  # Store conversation context for each session

# Common follow-up questions to enhance conversation
FOLLOW_UP_QUESTIONS = {
    "symptoms": [
        "How long have you been experiencing these symptoms?",
        "Is there anything that makes your symptoms better or worse?",
        "Have you experienced these symptoms before?",
        "Are these symptoms affecting your daily activities?"
    ],
    "general": [
        "Is there anything else I can help you with today?",
        "Do you have any other questions about your health?",
        "Was there something specific about this topic you'd like to know more about?"
    ],
    "medication": [
        "Are you currently taking any medications for this condition?",
        "Have you noticed any side effects from your medications?"
    ],
    "lifestyle": [
        "Have you made any lifestyle changes to help manage your health?",
        "How would you describe your diet and exercise routine?"
    ]
}

# Common clarification questions when input is unclear
CLARIFICATION_QUESTIONS = [
    "I'm not sure I understood completely. Could you tell me more about your concern?",
    "I'd like to help you better. Can you provide a bit more detail about what you're experiencing?",
    "To give you the best advice, could you elaborate on your question?",
    "I want to make sure I understand correctly. Could you rephrase that for me?"
]

def extract_keywords(text):
    """Extract important keywords from text with robust error handling"""
    try:
        # Fallback to basic tokenization if NLTK fails
        try:
            # Try NLTK tokenization first
            tokens = word_tokenize(text.lower())
        except Exception:
            # Fallback to basic string splitting
            tokens = text.lower().split()
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords and lemmatize if possible
        if lemmatizer:
            keywords = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        else:
            keywords = [word for word in tokens if word not in stop_words]
        
        return keywords
    except Exception as e:
        print(f"Warning: Keyword extraction failed: {e}")
        # Return basic words as fallback
        return [word for word in text.lower().split() if len(word) > 3]

def get_conversation_context(session_id):
    """Get the current conversation context for a session"""
    if session_id not in conversation_context:
        conversation_context[session_id] = {
            'last_topic': None,
            'mentioned_symptoms': [],
            'mentioned_conditions': [],
            'follow_up_asked': False,
            'clarification_asked': False,
            'message_count': 0,
            'unanswered_questions': []
        }
    return conversation_context[session_id]

def update_conversation_context(session_id, user_message, bot_response):
    """Update the conversation context based on the current exchange"""
    context = get_conversation_context(session_id)
    
    # Increment message count
    context['message_count'] += 1
    
    # Extract symptoms mentioned in the user message
    for symptom in SYMPTOM_TO_SPECIALIST.keys():
        if symptom in user_message.lower() and symptom not in context['mentioned_symptoms']:
            context['mentioned_symptoms'].append(symptom)
    
    # Extract conditions mentioned
    for disease in DISEASES.keys():
        if disease in user_message.lower() and disease not in context['mentioned_conditions']:
            context['mentioned_conditions'].append(disease)
    
    # Track if a question was asked by the user
    if '?' in user_message:
        question_match = re.search(r'([^.!?]+\?)', user_message)
        if question_match:
            question = question_match.group(1).strip()
            if question and len(question) > 10:
                context['unanswered_questions'].append(question)
    
    # Set the last topic based on response content
    if "symptom" in bot_response.lower():
        context['last_topic'] = "symptoms"
    elif "medication" in bot_response.lower():
        context['last_topic'] = "medication"
    elif "diet" in bot_response.lower() or "exercise" in bot_response.lower():
        context['last_topic'] = "lifestyle"
    elif any(disease in bot_response.lower() for disease in DISEASES.keys()):
        context['last_topic'] = "condition"
    
    # Reset follow-up and clarification flags
    context['follow_up_asked'] = "follow up" in bot_response.lower()
    context['clarification_asked'] = any(q in bot_response for q in CLARIFICATION_QUESTIONS)
    
    # Save the updated context
    conversation_context[session_id] = context

def add_conversation_elements(response, session_id):
    """Add personalized follow-up questions or references to previous messages"""
    context = get_conversation_context(session_id)
    
    # If we have very little context, don't add any elements
    if context['message_count'] <= 1:
        return response
    
    # Add a follow-up question based on the topic if one hasn't been asked
    if not context['follow_up_asked'] and context['last_topic'] and random.random() < 0.7:
        follow_ups = FOLLOW_UP_QUESTIONS.get(context['last_topic'], FOLLOW_UP_QUESTIONS['general'])
        if follow_ups:
            response += f"\n\n{random.choice(follow_ups)}"
            context['follow_up_asked'] = True
    
    # Reference previously mentioned symptoms if relevant
    if context['mentioned_symptoms'] and "symptom" in response.lower() and random.random() < 0.5:
        symptom = random.choice(context['mentioned_symptoms'])
        response += f"\n\nI notice you mentioned {symptom} earlier. Is that still bothering you?"
    
    # If there's an unanswered question, try to address it
    if context['unanswered_questions'] and random.random() < 0.4:
        question = context['unanswered_questions'].pop(0)
        response += f"\n\nGoing back to your question about '{question.strip()}' - would you like more information on that?"
    
    return response

def process_message(user_message, profile=None):
    """
    Process user messages with enhanced NLP techniques and return natural responses.
    """
    # Get session ID
    session_id = session.get('session_id', str(random.randint(1000, 9999)))
    session['session_id'] = session_id
    
    # Get conversation context
    context = get_conversation_context(session_id)
    
    # Clean the user message
    user_message = user_message.strip()
    
    # Check for very short or vague messages
    if len(user_message.split()) < 2 and not any(user_message.lower() == g for g in ["hi", "hello", "hey", "thanks", "ok"]):
        clarification = random.choice(CLARIFICATION_QUESTIONS)
        context['clarification_asked'] = True
        return clarification
    
    # Extract keywords for better understanding
    keywords = extract_keywords(user_message)
    
    # Check for simple greetings with more varied responses
    greetings = ["hi", "hello", "hey", "greetings", "howdy", "good morning", "good afternoon", "good evening"]
    greeting_pattern = r'\b(' + '|'.join(greetings) + r')\b'
    if re.search(greeting_pattern, user_message.lower()):
        greeting_responses = [
            "Hello! How can I assist with your health questions today?",
            "Hi there! How are you feeling today?",
            "Hello! What health concerns can I help you with?",
            "Hey! How can I support your healthcare journey today?"
        ]
        
        if profile and 'name' in profile:
            personalized_responses = [
                f"Hello {profile['name']}! How are you feeling today?",
                f"Hi {profile['name']}! What health concerns can I help you with?",
                f"Good to see you, {profile['name']}! How can I assist you today?",
                f"Hello {profile['name']}! How can I support your health journey today?"
            ]
            return random.choice(personalized_responses)
        else:
            return random.choice(greeting_responses)
    
    # Check for illness and general symptoms without specifics
    illness_phrases = ["sick", "ill", "unwell", "not feeling well", "feeling bad", "under the weather"]
    if any(phrase in user_message.lower() for phrase in illness_phrases):
        if not any(keyword in SYMPTOM_TO_SPECIALIST.keys() for keyword in keywords):
            # User feels sick but hasn't specified symptoms
            return ("I'm sorry to hear you're not feeling well. To help you better, could you describe your symptoms? " 
                   "For example, do you have a fever, cough, headache, or any other specific symptoms? "
                   "The more details you provide, the better I can assist you.")
    
    # If multiple messages exchanged and the user sends a short follow-up
    if context['message_count'] > 2 and len(user_message.split()) < 5:
        if context['mentioned_symptoms']:
            recent_symptom = context['mentioned_symptoms'][-1]
            if "yes" in user_message.lower() or "still" in user_message.lower():
                return f"I see that you're still concerned about {recent_symptom}. It might be helpful to consult with a {SYMPTOM_TO_SPECIALIST.get(recent_symptom, 'healthcare provider')}. Would you like me to help you schedule an appointment?"
        
        if context['mentioned_conditions']:
            recent_condition = context['mentioned_conditions'][-1]
            if "more" in user_message.lower() or "tell" in user_message.lower() or "info" in user_message.lower():
                disease_info = DISEASES.get(recent_condition, {})
                if disease_info:
                    return f"Here's some more information about {recent_condition}:\n\n" + \
                           f"Managing {recent_condition} typically involves working closely with a {disease_info.get('specialist', 'specialist')}. " + \
                           f"Besides medical treatment, lifestyle changes can help, such as: {disease_info.get('tips', 'consulting your doctor regularly')}.\n\n" + \
                           "Would you like to know about treatment options or have any specific questions?"
    
    # Process through existing logic but enhance with conversation elements
    base_response = suggest_doctor(user_message, profile)
    enhanced_response = add_conversation_elements(base_response, session_id)
    
    # Update conversation context with this exchange
    update_conversation_context(session_id, user_message, enhanced_response)
    
    return enhanced_response

def get_season():
    """
    Get the current season based on the current month
    """
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return "spring"
    elif 6 <= current_month <= 8:
        return "summer"
    elif 9 <= current_month <= 11:
        return "fall"
    else:
        return "winter"

# Symptom-to-specialist mapping
SYMPTOM_TO_SPECIALIST = {
    "fever": "General Physician",
    "cough": "Pulmonologist",
    "chest pain": "Cardiologist",
    "skin rash": "Dermatologist",
    "joint pain": "Orthopedic Specialist",
    "headache": "Neurologist",
    "stomach pain": "Gastroenterologist",
    "anxiety": "Psychiatrist",
    "depression": "Psychiatrist",
    "eye": "Ophthalmologist",
    "ear": "ENT Specialist",
    "throat": "ENT Specialist",
    "nose": "ENT Specialist",
    "dental": "Dentist",
    "tooth": "Dentist",
    "pregnancy": "Gynecologist",
    "menstrual": "Gynecologist",
    "urinary": "Urologist",
    "diabetes": "Endocrinologist",
    "thyroid": "Endocrinologist"
}

# Disease to specialist mapping with detailed information
DISEASES = {
    "hypertension": {
        "specialist": "Cardiologist",
        "description": "High blood pressure that can lead to heart disease if untreated.",
        "symptoms": "Headaches, shortness of breath, nosebleeds, but often asymptomatic.",
        "tips": "Reduce salt intake, exercise regularly, maintain a healthy weight, and limit alcohol.",
        "doctors": [
            {"name": "Dr. Sarah Chen", "expertise": "Cardiovascular Health", "years_exp": 15},
            {"name": "Dr. Michael Rodriguez", "expertise": "Hypertension Management", "years_exp": 12}
        ]
    },
    "diabetes": {
        "specialist": "Endocrinologist",
        "description": "A condition affecting how your body processes blood sugar.",
        "symptoms": "Increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue.",
        "tips": "Monitor blood sugar regularly, follow a balanced diet, exercise daily, take medications as prescribed.",
        "doctors": [
            {"name": "Dr. Emily Wilson", "expertise": "Diabetes Management", "years_exp": 18},
            {"name": "Dr. James Taylor", "expertise": "Endocrine Disorders", "years_exp": 14}
        ]
    },
    "asthma": {
        "specialist": "Pulmonologist",
        "description": "A condition that affects the airways and causes breathing difficulties.",
        "symptoms": "Wheezing, coughing, chest tightness, shortness of breath.",
        "tips": "Identify and avoid triggers, use inhalers as prescribed, follow your asthma action plan.",
        "doctors": [
            {"name": "Dr. Robert Johnson", "expertise": "Respiratory Diseases", "years_exp": 20},
            {"name": "Dr. Lisa Parker", "expertise": "Asthma Management", "years_exp": 15}
        ]
    },
    "arthritis": {
        "specialist": "Rheumatologist",
        "description": "Inflammation of one or more joints causing pain and stiffness.",
        "symptoms": "Joint pain, stiffness, swelling, and decreased range of motion.",
        "tips": "Stay active with low-impact exercises, maintain a healthy weight, use hot and cold therapy.",
        "doctors": [
            {"name": "Dr. David Kim", "expertise": "Joint Disorders", "years_exp": 16},
            {"name": "Dr. Maria Gonzalez", "expertise": "Rheumatology", "years_exp": 22}
        ]
    },
    "depression": {
        "specialist": "Psychiatrist",
        "description": "A mental health disorder characterized by persistently depressed mood.",
        "symptoms": "Persistent sadness, loss of interest, changes in sleep, fatigue, feelings of worthlessness.",
        "tips": "Seek professional help, stay connected with supportive people, exercise regularly, establish routines.",
        "doctors": [
            {"name": "Dr. Jennifer Adams", "expertise": "Mood Disorders", "years_exp": 17},
            {"name": "Dr. Thomas Wright", "expertise": "Clinical Depression", "years_exp": 19}
        ]
    },
    "migraine": {
        "specialist": "Neurologist",
        "description": "A severe headache disorder that can cause intense throbbing pain.",
        "symptoms": "Severe headache, nausea, light sensitivity, sound sensitivity, vision changes.",
        "tips": "Identify and avoid triggers, practice stress management, maintain regular sleep and meal schedules.",
        "doctors": [
            {"name": "Dr. Nicole Barnes", "expertise": "Headache Medicine", "years_exp": 14},
            {"name": "Dr. Andrew Peters", "expertise": "Neurology", "years_exp": 18}
        ]
    },
    "eczema": {
        "specialist": "Dermatologist",
        "description": "A condition that makes your skin red and itchy.",
        "symptoms": "Dry, itchy, red skin, rash, scaly patches, oozing or crusting.",
        "tips": "Moisturize regularly, use gentle soaps, identify and avoid triggers, manage stress.",
        "doctors": [
            {"name": "Dr. Olivia Martinez", "expertise": "Skin Disorders", "years_exp": 13},
            {"name": "Dr. William Lee", "expertise": "Dermatology", "years_exp": 16}
        ]
    },
    "gerd": {
        "specialist": "Gastroenterologist",
        "description": "A digestive disorder that affects the lower esophageal sphincter.",
        "symptoms": "Heartburn, regurgitation, chest pain, difficulty swallowing, coughing.",
        "tips": "Eat smaller meals, avoid lying down after eating, limit trigger foods, maintain a healthy weight.",
        "doctors": [
            {"name": "Dr. Rachel Green", "expertise": "Digestive Disorders", "years_exp": 15},
            {"name": "Dr. Brian White", "expertise": "Gastroenterology", "years_exp": 17}
        ]
    },
    "thyroid_disorders": {
        "specialist": "Endocrinologist",
        "description": "Conditions that affect the thyroid gland's function, causing metabolic imbalances.",
        "symptoms": "Fatigue, weight changes, sensitivity to temperature, mood changes, irregular heartbeat.",
        "tips": "Take medications as prescribed, follow a healthy diet, get regular thyroid function tests.",
        "doctors": [
            {"name": "Dr. Sophia Rodriguez", "expertise": "Thyroid Specialists", "years_exp": 19},
            {"name": "Dr. Benjamin Chang", "expertise": "Hormonal Disorders", "years_exp": 16}
        ]
    },
    "glaucoma": {
        "specialist": "Ophthalmologist",
        "description": "An eye condition that damages the optic nerve, often caused by high pressure in the eye.",
        "symptoms": "Gradual vision loss, eye pain, blurred vision, seeing halos around lights, headaches.",
        "tips": "Get regular eye exams, use prescribed eye drops, protect eyes from injury, manage other health conditions.",
        "doctors": [
            {"name": "Dr. Jessica Wong", "expertise": "Glaucoma Specialist", "years_exp": 15},
            {"name": "Dr. Samuel Patel", "expertise": "Eye Diseases", "years_exp": 18}
        ]
    },
    "osteoporosis": {
        "specialist": "Endocrinologist or Rheumatologist",
        "description": "A condition causing bones to become weak and brittle, increasing fracture risk.",
        "symptoms": "Back pain, loss of height, stooped posture, bone fractures from minor falls.",
        "tips": "Get adequate calcium and vitamin D, exercise regularly, avoid smoking and excessive alcohol.",
        "doctors": [
            {"name": "Dr. Olivia Brown", "expertise": "Bone Health", "years_exp": 14},
            {"name": "Dr. Nathan Miller", "expertise": "Metabolic Bone Disorders", "years_exp": 17}
        ]
    },
    "fibromyalgia": {
        "specialist": "Rheumatologist",
        "description": "A disorder characterized by widespread musculoskeletal pain and fatigue.",
        "symptoms": "Chronic pain, fatigue, sleep problems, cognitive difficulties, headaches.",
        "tips": "Maintain regular exercise, practice stress reduction, follow a regular sleep routine, manage pain with appropriate treatments.",
        "doctors": [
            {"name": "Dr. Rebecca Foster", "expertise": "Chronic Pain Management", "years_exp": 15},
            {"name": "Dr. Jason Rivera", "expertise": "Fibromyalgia Specialist", "years_exp": 13}
        ]
    },
    "multiple_sclerosis": {
        "specialist": "Neurologist",
        "description": "A disease of the central nervous system that disrupts communication between brain and body.",
        "symptoms": "Fatigue, numbness, weakness, problems with balance and coordination, vision problems.",
        "tips": "Stay cool, exercise regularly, manage stress, get adequate rest, follow a balanced diet.",
        "doctors": [
            {"name": "Dr. Michelle Harris", "expertise": "Multiple Sclerosis Specialist", "years_exp": 20},
            {"name": "Dr. Kevin Walker", "expertise": "Neuroimmunology", "years_exp": 18}
        ]
    },
    "celiac_disease": {
        "specialist": "Gastroenterologist",
        "description": "An autoimmune disorder triggered by consuming gluten, affecting the small intestine.",
        "symptoms": "Diarrhea, abdominal pain, bloating, fatigue, weight loss, anemia.",
        "tips": "Follow a strict gluten-free diet, read food labels carefully, watch for cross-contamination.",
        "doctors": [
            {"name": "Dr. Amanda Lewis", "expertise": "Celiac Disease Specialist", "years_exp": 12},
            {"name": "Dr. Eric Thompson", "expertise": "Digestive Health", "years_exp": 16}
        ]
    }
}

# Medical tips database
MEDICAL_TIPS = {
    "general": [
        "Stay hydrated by drinking at least 8 glasses of water daily.",
        "Aim for 7-8 hours of quality sleep each night.",
        "Incorporate physical activity into your daily routine for at least 30 minutes.",
        "Eat a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
        "Practice stress management techniques such as meditation or deep breathing.",
    ],
    "seasonal": {
        "spring": "Stay prepared for seasonal allergies with appropriate medications and avoid peak pollen times.",
        "summer": "Protect your skin from sun damage with SPF 30+ sunscreen and stay hydrated in hot weather.",
        "fall": "Get your annual flu shot and strengthen your immune system with vitamin-rich foods.",
        "winter": "Wash hands frequently to prevent colds and flu, and combat seasonal depression with light therapy."
    },
    "preventive": {
        "heart": "Monitor blood pressure regularly, limit sodium intake, and exercise for cardiovascular health.",
        "diabetes": "Maintain a healthy weight, limit sugary foods, and get regular blood sugar screenings.",
        "cancer": "Schedule recommended screenings, avoid tobacco, limit alcohol, and protect skin from sun damage.",
        "mental_health": "Prioritize social connections, practice mindfulness, and seek help when feeling overwhelmed."
    }
}

# Emergency guidance
EMERGENCY_GUIDANCE = {
    "chest pain": "If experiencing severe chest pain, especially with shortness of breath, sweating, or nausea, call emergency services immediately. This could be a heart attack.",
    "stroke": "If you notice facial drooping, arm weakness, or speech difficulties, seek emergency care immediately. Remember FAST: Face, Arms, Speech, Time to call for help.",
}

def suggest_doctor(symptom, profile=None):
    """
    Suggest a doctor based on symptoms and patient profile with enhanced conversation abilities.
    """
    session_id = session.get('session_id', str(random.randint(1000, 9999)))
    context = get_conversation_context(session_id)
    
    # Check for emergency conditions first
    for emergency_key, guidance in EMERGENCY_GUIDANCE.items():
        if emergency_key in symptom.lower():
            return f"⚠️ EMERGENCY ALERT: {guidance}"
    
    # Check if user is asking about a specific disease
    for disease_name, disease_info in DISEASES.items():
        if disease_name in symptom.lower() or (disease_info.get("description") and disease_info["description"].lower() in symptom.lower()):
            doctors = disease_info["doctors"]
            doctor_info = random.choice(doctors)
            
            # Add disease to conversation context
            if disease_name not in context['mentioned_conditions']:
                context['mentioned_conditions'].append(disease_name)
            
            context['last_topic'] = "condition"
            
            # Vary the response format to sound more natural
            intro_phrases = [
                f"Let me tell you about {disease_name}.",
                f"Here's the information about {disease_name} you asked for.",
                f"I'd be happy to explain {disease_name} to you.",
                f"About {disease_name}:"
            ]
            
            response = (
                f"📋 *{disease_name.upper()}*\n"
                f"{random.choice(intro_phrases)} This condition is typically treated by a {disease_info['specialist']}.\n\n"
                f"*Description*: {disease_info['description']}\n"
                f"*Common symptoms*: {disease_info['symptoms']}\n\n"
                f"*Health tips*: {disease_info['tips']}\n\n"
                f"*Recommended specialist*: {doctor_info['name']} with {doctor_info['years_exp']} years of experience in {doctor_info['expertise']}."
            )
            
            if profile and 'name' in profile:
                # Choose different personalized intros
                personal_intros = [
                    f"Hello {profile['name']}, here's what you should know about {disease_name}:",
                    f"{profile['name']}, I've put together some information about {disease_name} for you:",
                    f"Thanks for asking about {disease_name}, {profile['name']}. Here's what I found:"
                ]
                response = f"{random.choice(personal_intros)}\n\n" + response.split("\n\n", 1)[1]
                
                # Add consultation booking option with varied phrasing
                booking_phrases = [
                    f"Would you like to schedule a consultation with a {disease_info['specialist']}?",
                    f"I can help you book an appointment with a {disease_info['specialist']} if you'd like.",
                    f"Let me know if you'd like to see a {disease_info['specialist']} about this.",
                    f"Would you like me to set up an appointment with one of our {disease_info['specialist']}s?"
                ]
                response += f"\n\n{random.choice(booking_phrases)} Just say 'book appointment' or 'schedule consultation'."
            
            return response
    
    # Check if user is asking for health tips with more natural language
    if any(word in symptom.lower() for word in ["tip", "advice", "recommend", "suggestion", "help", "guide"]):
        if "heart" in symptom.lower():
            return f"💙 *Heart Health Tips*:\n{MEDICAL_TIPS['preventive']['heart']}"
        elif "diabetes" in symptom.lower():
            return f"💉 *Diabetes Management Tips*:\n{MEDICAL_TIPS['preventive']['diabetes']}"
        elif "cancer" in symptom.lower():
            return f"🎗️ *Cancer Prevention Tips*:\n{MEDICAL_TIPS['preventive']['cancer']}"
        elif "mental" in symptom.lower() or "stress" in symptom.lower() or "anxiety" in symptom.lower() or "depression" in symptom.lower():
            return f"🧠 *Mental Health Tips*:\n{MEDICAL_TIPS['preventive']['mental_health']}"
        else:
            # Provide general health tips with varied language
            tips = MEDICAL_TIPS["general"]
            selected_tips = random.sample(tips, min(3, len(tips)))
            tips_text = "\n• ".join(selected_tips)
            
            # Add seasonal tip based on current month with conversational intro
            season = get_season()
            seasonal_tip = MEDICAL_TIPS["seasonal"][season]
            
            intros = [
                "Here are some health tips that might help you:",
                "These health recommendations could be beneficial for you:",
                "Consider these health suggestions:",
                "Here are some general wellness tips I'd recommend:"
            ]
            
            seasonal_intros = [
                f"Since it's {season} now, here's a timely tip:",
                f"For this {season} season specifically:",
                f"This is especially important during {season}:",
                f"A special recommendation for the current {season} season:"
            ]
            
            return f"💡 *Health Tips*\n\n{random.choice(intros)}\n\n• {tips_text}\n\n🌡️ *{random.choice(seasonal_intros)}*\n{seasonal_tip}"
    
    # Check if user wants to book a consultation with more conversational responses
    booking_keywords = ["book", "schedule", "appointment", "consultation", "consult", "see a doctor"]
    if any(keyword in symptom.lower() for keyword in booking_keywords):
        if not profile or 'name' not in profile:
            return "To book a consultation, I'll need some basic information about you first. Could you click on the 'Patient Profile' button and fill in your details? This helps us match you with the right healthcare provider."
        
        # Create a more dynamic appointment scheduling experience
        days_ahead = random.randint(2, 5)
        appt_date = datetime.now() + timedelta(days=days_ahead)
        time_options = ["9:00 AM", "11:30 AM", "2:15 PM", "4:00 PM"]
        appt_time = random.choice(time_options)
        
        # Look for doctor specialty in the message
        specialist = None
        
        # First check if any condition was mentioned recently in the conversation
        if context['mentioned_conditions']:
            recent_condition = context['mentioned_conditions'][-1]
            condition_info = DISEASES.get(recent_condition)
            if condition_info:
                specialist = condition_info["specialist"]
        
        # If no condition, check for symptoms
        if not specialist and context['mentioned_symptoms']:
            recent_symptom = context['mentioned_symptoms'][-1]
            specialist = SYMPTOM_TO_SPECIALIST.get(recent_symptom)
        
        # Finally look for keywords in current message
        if not specialist:
            for key, value in SYMPTOM_TO_SPECIALIST.items():
                if key in symptom.lower():
                    specialist = value
                    break
        
        # Default to General Physician if no specialty is determined
        if not specialist:
            specialist = "General Physician"
        
        # Find a doctor for this specialty
        doctor = None
        for disease, info in DISEASES.items():
            if info["specialist"] == specialist and info["doctors"]:
                doctor = random.choice(info["doctors"])
                break
        
        if not doctor:
            doctor = {"name": f"Dr. {random.choice(['Smith', 'Jones', 'Patel', 'Johnson', 'Williams'])}", "expertise": specialist}
        
        # Store consultation in memory
        session_id = session.get('session_id', str(random.randint(1000, 9999)))
        if session_id not in consultations:
            consultations[session_id] = []
        
        consultation_id = len(consultations[session_id]) + 1
        
        new_consultation = {
            "id": consultation_id,
            "patient": profile.get('name', 'Unknown'),
            "doctor": doctor["name"],
            "specialty": specialist,
            "date": appt_date.strftime("%A, %B %d, %Y"),
            "time": appt_time,
            "status": "Scheduled"
        }
        
        consultations[session_id].append(new_consultation)
        
        # Create more varied and conversational booking confirmations
        booking_responses = [
            f"I've scheduled your appointment with {doctor['name']}, who specializes in {specialist} care.",
            f"Great news! I've booked you with {doctor['name']}, a {specialist}.",
            f"Your appointment is confirmed with {doctor['name']} ({specialist}).",
            f"I've arranged for you to see {doctor['name']}, an experienced {specialist}."
        ]
        
        arrival_notes = [
            "Please arrive 15 minutes before your scheduled time.",
            "We recommend arriving 15 minutes early to complete any paperwork.",
            "It's a good idea to come 15 minutes ahead of your appointment time.",
            "Please come 15 minutes early to check in."
        ]
        
        return (
            f"✅ *Consultation Booked*\n\n"
            f"{random.choice(booking_responses)}\n\n"
            f"*Date*: {new_consultation['date']}\n"
            f"*Time*: {new_consultation['time']}\n"
            f"*Consultation ID*: #{consultation_id}\n\n"
            f"{random.choice(arrival_notes)} You can view or manage your appointments by typing 'my consultations'."
        )
    
    # Check if user is asking about their consultations with more natural responses
    consultation_phrases = ["my consultation", "my appointment", "my booking", "my doctor", "my visit", "check appointment"]
    if any(phrase in symptom.lower() for phrase in consultation_phrases):
        session_id = session.get('session_id', None)
        if not session_id or session_id not in consultations or not consultations[session_id]:
            no_consult_responses = [
                "I don't see any scheduled consultations for you at the moment. Would you like to book one?",
                "You don't have any upcoming appointments in our system. Would you like me to help you schedule one?",
                "It looks like you don't have any consultations scheduled. I'd be happy to help you book one now.",
                "There are no appointments in your calendar. Would you like to schedule a consultation?"
            ]
            return random.choice(no_consult_responses)
        
        user_consultations = consultations[session_id]
        
        if len(user_consultations) == 1:
            c = user_consultations[0]
            single_consult_intros = [
                "Here are the details of your upcoming appointment:",
                "I found your scheduled consultation:",
                "Here's your appointment information:",
                "I have your consultation details right here:"
            ]
            
            management_options = [
                "If you need to make changes, you can type 'reschedule consultation' or 'cancel consultation'.",
                "Need to make changes? Just let me know if you want to reschedule or cancel.",
                "You can reschedule or cancel this appointment by typing 'reschedule consultation' or 'cancel consultation'.",
                "To modify this appointment, just say 'reschedule consultation' or 'cancel consultation'."
            ]
            
            return (
                f"🗓️ *Your Consultation*\n\n"
                f"{random.choice(single_consult_intros)}\n\n"
                f"*Doctor*: {c['doctor']}\n"
                f"*Specialty*: {c['specialty']}\n"
                f"*Date*: {c['date']}\n"
                f"*Time*: {c['time']}\n"
                f"*Status*: {c['status']}\n"
                f"*Consultation ID*: #{c['id']}\n\n"
                f"{random.choice(management_options)}"
            )
        else:
            multiple_consult_intros = [
                "Here are all your scheduled consultations:",
                "I found the following appointments for you:",
                "These are your upcoming consultations:",
                "Here's a list of your scheduled appointments:"
            ]
            
            response = f"🗓️ *Your Consultations*\n\n{random.choice(multiple_consult_intros)}\n\n"
            for i, c in enumerate(user_consultations, 1):
                response += (
                    f"*{i}.* {c['doctor']} ({c['specialty']})\n"
                    f"   Date: {c['date']} at {c['time']}\n"
                    f"   Status: {c['status']}\n"
                    f"   ID: #{c['id']}\n\n"
                )
            response += "To manage a specific consultation, type 'reschedule #ID' or 'cancel #ID' with the ID number."
            return response
    
    # Check for cancel/reschedule requests with improved handling
    cancel_match = re.search(r'cancel\s+(?:consultation|appointment)?\s*#?(\d+)', symptom.lower())
    reschedule_match = re.search(r'reschedule\s+(?:consultation|appointment)?\s*#?(\d+)', symptom.lower())
    
    if cancel_match:
        consultation_id = int(cancel_match.group(1))
        session_id = session.get('session_id', None)
        if not session_id or session_id not in consultations:
            return "I couldn't find any consultations to cancel. If you'd like to book a new appointment, just let me know."
        
        for c in consultations[session_id]:
            if c['id'] == consultation_id:
                c['status'] = "Cancelled"
                
                cancel_responses = [
                    f"I've cancelled your appointment with {c['doctor']} as requested.",
                    f"Your consultation with {c['doctor']} has been successfully cancelled.",
                    f"The appointment with {c['doctor']} has been removed from your schedule.",
                    f"Your consultation with {c['doctor']} is now cancelled."
                ]
                
                followup_questions = [
                    "Would you like to schedule a new appointment?",
                    "Do you want to book a different time instead?",
                    "Can I help you find another specialist?",
                    "Is there anything else I can help you with today?"
                ]
                
                return f"✅ {random.choice(cancel_responses)} This was scheduled for {c['date']} at {c['time']}. {random.choice(followup_questions)}"
                
        return f"I couldn't find consultation #{consultation_id}. Please check the ID number and try again, or type 'my consultations' to see all your appointments."
    
    if reschedule_match:
        consultation_id = int(reschedule_match.group(1))
        session_id = session.get('session_id', None)
        if not session_id or session_id not in consultations:
            return "I couldn't find any consultations to reschedule. Would you like to book a new appointment instead?"
        
        for c in consultations[session_id]:
            if c['id'] == consultation_id:
                # Create a new date with some variation
                days_ahead = random.randint(3, 7)
                new_date = datetime.now() + timedelta(days=days_ahead)
                time_options = ["10:00 AM", "1:30 PM", "3:45 PM", "5:00 PM"]
                new_time = random.choice(time_options)
                
                old_date = c['date']
                old_time = c['time']
                
                c['date'] = new_date.strftime("%A, %B %d, %Y")
                c['time'] = new_time
                
                reschedule_responses = [
                    f"I've rescheduled your appointment with {c['doctor']}.",
                    f"Your consultation with {c['doctor']} has been successfully rescheduled.",
                    f"I've updated your appointment with {c['doctor']}.",
                    f"Your consultation time with {c['doctor']} has been changed."
                ]
                
                return (
                    f"✅ *Consultation Rescheduled*\n\n"
                    f"{random.choice(reschedule_responses)}\n\n"
                    f"*Original*: {old_date} at {old_time}\n"
                    f"*New Date*: {c['date']}\n"
                    f"*New Time*: {c['time']}\n\n"
                    f"Your consultation ID remains #{c['id']}. Is this new time convenient for you?"
                )
                
        return f"I couldn't find consultation #{consultation_id}. Please verify the ID and try again, or type 'my consultations' to see all your appointments."
    
    # Find matches in known symptoms with more conversational responses
    for key, specialist in SYMPTOM_TO_SPECIALIST.items():
        if key in symptom.lower():
            # Add this symptom to conversation context
            if key not in context['mentioned_symptoms']:
                context['mentioned_symptoms'].append(key)
            
            symptom_responses = [
                f"Based on your mention of {key}, I'd recommend consulting a {specialist}.",
                f"For symptoms related to {key}, a {specialist} would be the appropriate specialist to see.",
                f"Given that you're experiencing {key}, you should consider visiting a {specialist}.",
                f"A {specialist} would be the right doctor to help with your {key}."
            ]
            
            response = random.choice(symptom_responses)
            
            # Add personalized recommendation if profile is available
            if profile and 'name' in profile:
                age_info = f", age {profile['age']}" if 'age' in profile else ""
                
                personal_intros = [
                    f"{profile['name']}{age_info}, {response.lower()}",
                    f"Hello {profile['name']}{age_info}. {response}",
                    f"Thanks for sharing that, {profile['name']}{age_info}. {response}",
                    f"{profile['name']}{age_info}, I've analyzed your symptoms. {response}"
                ]
                
                response = random.choice(personal_intros)

                # Add advice based on medical history if available
                if profile.get('medicalHistory') and profile['medicalHistory'] != 'None':
                    history_notes = [
                        f" Given your history of {profile['medicalHistory']}, it's especially important to mention this when consulting.",
                        f" Be sure to inform the specialist about your {profile['medicalHistory']} history.",
                        f" With your {profile['medicalHistory']} history, this deserves prompt attention.",
                        f" Your {profile['medicalHistory']} should be discussed with the specialist."
                    ]
                    response += random.choice(history_notes)
                
                # Add medication warning if applicable
                if profile.get('currentMedications') and profile['currentMedications'] != 'None':
                    med_notes = [
                        f" Please also inform the doctor about your current medications: {profile['currentMedications']}.",
                        f" Don't forget to mention that you're taking {profile['currentMedications']} during your consultation.",
                        f" The specialist should know about your medication regimen, including {profile['currentMedications']}.",
                        f" Remember to bring a list of your current medications ({profile['currentMedications']}) to your appointment."
                    ]
                    response += random.choice(med_notes)
                    
                # Add booking option with varied phrasing
                booking_options = [
                    f"\n\nWould you like me to schedule an appointment with a {specialist} for you?",
                    f"\n\nI can help you book a consultation with a {specialist} right now if you'd like.",
                    f"\n\nShall I set up an appointment with a {specialist} for you?",
                    f"\n\nWould you like me to arrange a consultation with a {specialist}?"
                ]
                response += random.choice(booking_options) + " Just say 'book consultation' to proceed."
            
            return response
            
    # Generic response if no specific match - make more conversational
    generic_responses = [
        "I'm not seeing any specific symptoms that point to a particular specialist. Could you tell me more about what you're experiencing?",
        "I need a bit more information to suggest the right specialist. Can you describe your symptoms in more detail?",
        "To help you better, I'd need to know more about your specific health concerns. Could you elaborate on your symptoms?",
        "I'd like to connect you with the right specialist, but I need more details about your symptoms. Could you share more information?"
    ]
    
    if context['message_count'] <= 1:
        # First message tends to be vaguer, so provide general guidance
        intro_responses = [
            "I'm here to help with your health questions! To get started, you can ask me about specific symptoms, diseases, or general health advice.",
            "Welcome! I can assist with medical information, specialist recommendations, or even booking appointments. What health topic can I help with?",
            "Hello! I'm your healthcare assistant. You can ask me about medical conditions, symptoms, or health tips. What would you like to know?",
            "I'm ready to help with your healthcare needs. Feel free to ask about specific conditions, symptoms, or medical advice. What can I help with today?"
        ]
        return random.choice(intro_responses)
    
    if profile and 'name' in profile:
        personalized_generic = [
            f"{profile['name']}, I'm not quite sure what specific health concern you're asking about.",
            f"Hello {profile['name']}, I'd like to help but need more specific information about your symptoms.",
            f"{profile['name']}, to provide the best guidance, could you share more details about your health concern?",
            f"I want to give you the best advice, {profile['name']}, but need more specific symptoms to recommend a specialist."
        ]
        return f"{random.choice(personalized_generic)} You can ask about specific conditions like diabetes or hypertension, describe symptoms like headaches or joint pain, or request general health tips."
    else:
        return f"{random.choice(generic_responses)}\n\nYou can also ask about specific health conditions like diabetes or hypertension, or request general health tips."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/style.css')
def serve_css():
    return app.send_static_file('style.css')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    profile = data.get('profile', {})
    
    try:
        # Generate a response with improved NLP
        bot_response = process_message(user_message, profile)
    except Exception as e:
        print(f"Error processing message: {e}")
        bot_response = "I apologize, but I encountered an error processing your request. Could you try rephrasing your question?"
    
    # Save conversation history (in memory)
    session_id = session.get('session_id', str(random.randint(1000, 9999)))
    session['session_id'] = session_id
    
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        'user': user_message,
        'bot': bot_response,
        'timestamp': datetime.now().isoformat(),
        'profile': profile
    })

    return jsonify({'response': bot_response})

@app.route('/save_profile', methods=['POST'])
def save_profile():
    profile_data = request.json
    
    # Create session ID if not exists
    session_id = session.get('session_id', str(random.randint(1000, 9999)))
    session['session_id'] = session_id
    
    # Save profile to memory
    patient_profiles[session_id] = profile_data
    
    return jsonify({'status': 'success'})

@app.route('/get_conversation_history', methods=['GET'])
def get_conversation_history():
    session_id = session.get('session_id')
    if not session_id or session_id not in conversation_history:
        return jsonify([])
    
    return jsonify(conversation_history[session_id])

@app.route('/get_consultations', methods=['GET'])
def get_consultations():
    """Return the consultations for the current session"""
    session_id = session.get('session_id', str(random.randint(1000, 9999)))
    session['session_id'] = session_id
    
    if session_id not in consultations:
        consultations[session_id] = []
    
    return jsonify(consultations[session_id])

# Print banner at startup
print("\n==================================================")
print("HEALTHCARE CHATBOT SERVER STARTING")
print("==================================================")
print("✓ Server starting on http://localhost:5000")
print("✓ Open your browser and navigate to: http://localhost:5000")
print("==================================================\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

"""
Streamlit Frontend for Quiz Generator Model
"""
import streamlit as st
import os
from inference import QuizGenerator
import time
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import textwrap

# Page configuration
st.set_page_config(
    page_title="Quiz Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with prominent colors
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 6px solid #FF6B35;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #F7931E 0%, #FF6B35 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #FF6B35;
    }
    /* Style for success messages */
    .stSuccess {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    /* Style for error messages */
    .stError {
        background-color: #f44336 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    /* Style for warning messages */
    .stWarning {
        background-color: #FF9800 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    /* Style for info messages */
    .stInfo {
        background-color: #2196F3 !important;
        color: white !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    [data-testid="stSidebar"] .stButton>button {
        background: #FF6B35;
        color: white;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background: #F7931E;
    }
    /* Selectbox and input styling */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #004E89 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    /* Main content area */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    /* Headers */
    h1, h2, h3 {
        color: #004E89 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
SUBJECT_OPTIONS = [
    "Applied Physics",
    "Automata",
    "COAL",
    "Calculus",
    "Cloud Computing",
    "Computer Networks",
    "Data Structures and Algorithms",
    "Digital Image Processing",
    "ICT",
    "Machine Learning and Operations",
    "NLP",
    "Object Oriented Programming",
    "Operating System",
    "Programming Fundamentals",
    "Software Design and Analysis",
    "Software for machine and devices",
    "Theory of Computation and Automata",
]



if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = None

def find_latest_checkpoint(base_path):
    """Find the latest checkpoint folder in the model directory"""
    if not os.path.exists(base_path):
        return None
    
    # Check if base_path itself contains model files
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path
    
    # Look for checkpoint folders
    checkpoints = []
    for item in os.listdir(base_path):
        checkpoint_path = os.path.join(base_path, item)
        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
            # Check if it contains model files
            if os.path.exists(os.path.join(checkpoint_path, "config.json")):
                try:
                    checkpoint_num = int(item.split("-")[1])
                    checkpoints.append((checkpoint_num, checkpoint_path))
                except (ValueError, IndexError):
                    continue
    
    if checkpoints:
        # Return the latest checkpoint (highest number)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    
    return None

@st.cache_resource
def load_model(model_path):
    """Load the quiz generator model (cached)"""
    try:
        # Try to find latest checkpoint if model_path is a directory
        if os.path.isdir(model_path):
            checkpoint_path = find_latest_checkpoint(model_path)
            if checkpoint_path:
                model_path = checkpoint_path
        
        generator = QuizGenerator(model_path)
        return generator, True
    except Exception as e:
        return None, str(e)

def _wrap_text_for_pdf(text, width=80):
    """Ensure text always fits PDF width by breaking long tokens."""
    if not text:
        return ""
    wrapped_lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return "\n".join(wrapped_lines) if wrapped_lines else text

def _safe_multicell(pdf, text, line_height=8):
    """Render text safely regardless of token length."""
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin
    if effective_width <= 0:
        effective_width = pdf.w

    lines = _wrap_text_for_pdf(text).split("\n")
    for line in lines:
        if not line:
            pdf.ln(line_height)
            continue

        buffer = ""
        for char in line:
            tentative = buffer + char
            if pdf.get_string_width(tentative) <= effective_width or not buffer:
                buffer = tentative
            else:
                pdf.cell(0, line_height, buffer, ln=True)
                buffer = char

        if buffer:
            pdf.cell(0, line_height, buffer, ln=True)

def create_pdf_from_questions(questions, metadata):
    """Build a simple PDF for the generated quiz questions."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Quiz Generator Output", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 12)
    meta_lines = [
        f"Subject: {metadata.get('subject', '-')}",
        f"Topic: {metadata.get('topic', '-')}",
        f"Difficulty: {metadata.get('difficulty', '-')}",
        f"Type: {metadata.get('question_type', '-')}",
        f"Generated: {metadata.get('generated_at', '-')}",
        f"Count: {len(questions)}",
    ]
    for line in meta_lines:
        _safe_multicell(pdf, line)
    pdf.ln(5)

    for idx, question in enumerate(questions, start=1):
        pdf.set_font("Helvetica", "B", 12)
        _safe_multicell(pdf, f"Question {idx}")
        pdf.set_font("Helvetica", "", 12)
        _safe_multicell(pdf, question)
        pdf.ln(4)

    raw_output = pdf.output(dest="S")
    if isinstance(raw_output, bytearray):
        raw_output = bytes(raw_output)
    elif isinstance(raw_output, str):
        raw_output = raw_output.encode("latin-1")

    buffer = BytesIO()
    buffer.write(raw_output)
    buffer.seek(0)
    return buffer

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 Quiz Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Quiz Question Generation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path
        model_path = st.text_input(
            "Model Path",
            value="./results/t5-quiz-generator",
            help="Path to the fine-tuned model directory (will automatically find latest checkpoint if in checkpoint folder)"
        )
        
        # Load model button
        if st.button("🔄 Load Model", use_container_width=True):
            if os.path.exists(model_path):
                with st.spinner("Loading model... This may take a moment..."):
                    generator, status = load_model(model_path)
                    if status == True:
                        st.session_state.generator = generator
                        st.session_state.model_loaded = True
                        st.success(" Model loaded successfully!")
                    else:
                        st.error(f"Error loading model: {status}")
                        st.session_state.model_loaded = False
            else:
                st.error(f"Model path not found: {model_path}")
                st.info("💡 Make sure you've trained the model first using train.py")
                st.session_state.model_loaded = False
        
        st.divider()
        
        # Model status
        if st.session_state.model_loaded:
            st.success("Model Ready")
        else:
            st.warning("⚠️ Model Not Loaded")
        
        st.divider()
        
        # Information
        st.header("ℹ About")
        subject_list = "\n".join(f"- {subject}" for subject in SUBJECT_OPTIONS)
        st.markdown(f"""
        This tool generates quiz questions using a fine-tuned T5 model.
        
        **Supported Subjects:**
        {subject_list}
        
        **Question Types:**
        - MCQ
        - Short Answer
        - Long Answer
        - Programming
        """)
    
    # Main content area
    if not st.session_state.model_loaded:
        st.warning(" Please load the model first using the sidebar.")
        
        # Show instructions
        with st.expander("How to Use", expanded=True):
            st.markdown("""
            ### Steps to get started:
            
            1. **Train the model** (if not already done):
               ```bash
               python train.py
               ```
            
            2. **Load the model** using the sidebar button
            
            3. **Fill in the form** below with your preferences
            
            4. **Generate questions** and see the results!
            
            ### Tips:
            - Start with generating 1-2 questions to test
            - CPU inference is slower (10-30 seconds per question)
            - For faster generation, use GPU if available
            """)
    else:
        # Input form
        st.header(" Generate Quiz Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.selectbox(
                "📚 Subject",
                options=SUBJECT_OPTIONS,
                help="Select the subject for the quiz question"
            )
            
            topic = st.text_input(
                "📖 Topic",
                placeholder="e.g., Tokenization, Derivatives, Mechanics...",
                help="Enter the specific topic within the subject"
            )
            
            difficulty = st.selectbox(
                "📊 Difficulty Level",
                options=["easy", "medium", "hard"],
                index=1,
                help="Select the difficulty level"
            )
        
        with col2:
            # Determine available question types based on subject
            available_types = ["MCQ", "Short Answer", "Long Answer"]
            if subject in CODING_SUBJECTS:
                available_types.append("programming")
            
            question_type = st.selectbox(
                "📝 Question Type",
                options=available_types,
                help="Select the type of question to generate"
            )
            
            num_questions = st.number_input(
                " Number of Questions",
                min_value=1,
                max_value=10,
                value=1,
                help="How many questions to generate (CPU may be slow for multiple questions)"
            )
            
            # Hidden Configuration (Hardcoded for best results)
            # CHANGE TEMPERATURE HERE to control creativity (0.1 = deterministic, 1.0+ = creative/random)
            temperature = 0.9 
            use_sampling = True
            max_length = 128
        
        # Generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.button(
                "Generate Questions",
                use_container_width=True,
                type="primary"
            )
        
        # Generate questions
        if generate_button:
            if not topic.strip():
                st.error(" Please enter a topic!")
            else:
                try:
                    with st.spinner(f"🤖 Generating {num_questions} question(s)... This may take a while on CPU."):
                        timestamp = datetime.now()
                        metadata = {
                            "subject": subject,
                            "topic": topic,
                            "difficulty": difficulty,
                            "question_type": question_type,
                            "generated_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "timestamp_slug": timestamp.strftime("%Y%m%d_%H%M%S")
                        }
                        if num_questions == 1:
                            # Generate single question
                            question = st.session_state.generator.generate_question(
                                subject=subject,
                                topic=topic,
                                difficulty=difficulty,
                                question_type=question_type,
                                temperature=temperature,
                                max_length=max_length,
                                do_sample=use_sampling
                            )
                            
                            # Display result
                            st.success(" Question generated successfully!")
                            st.markdown("---")
                            st.markdown(f"### 📝 Generated Question")
                            st.markdown(f'<div class="question-box">{question}</div>', unsafe_allow_html=True)
                            
                            # Metadata
                            with st.expander("📋 Question Details"):
                                st.write(f"**Subject:** {subject}")
                                st.write(f"**Topic:** {topic}")
                                st.write(f"**Difficulty:** {difficulty}")
                                st.write(f"**Type:** {question_type}")
                                st.write(f"**Length:** {len(question)} characters")

                            st.session_state.generated_questions = {
                                "questions": [question],
                                "metadata": metadata
                            }
                        else:
                            # Generate multiple questions
                            questions = st.session_state.generator.generate_multiple_questions(
                                subject=subject,
                                topic=topic,
                                difficulty=difficulty,
                                question_type=question_type,
                                num_questions=num_questions,
                                temperature=temperature,
                                max_length=max_length,
                                do_sample=use_sampling
                            )
                            
                            # Display results
                            st.success(f" Generated {len(questions)} question(s) successfully!")
                            st.markdown("---")
                            
                            for i, question in enumerate(questions, 1):
                                st.markdown(f"###  Question {i}")
                                st.markdown(f'<div class="question-box">{question}</div>', unsafe_allow_html=True)
                                st.markdown("---")
                            
                            # Metadata
                            with st.expander("Generation Details"):
                                st.write(f"**Subject:** {subject}")
                                st.write(f"**Topic:** {topic}")
                                st.write(f"**Difficulty:** {difficulty}")
                                st.write(f"**Type:** {question_type}")
                                st.write(f"**Number Generated:** {len(questions)}/{num_questions}")

                            st.session_state.generated_questions = {
                                "questions": questions,
                                "metadata": metadata
                            }
                    
                except Exception as e:
                    st.session_state.generated_questions = None
                    st.error(f" Error generating questions: {str(e)}")
                    st.exception(e)
        
        if st.session_state.generated_questions:
            st.markdown("---")
            st.header("⬇️ Download Questions")
            quiz_data = st.session_state.generated_questions
            subject_slug = quiz_data["metadata"].get("subject", "quiz").replace(" ", "_").lower()
            timestamp_slug = quiz_data["metadata"].get("timestamp_slug", "output")
            pdf_buffer = create_pdf_from_questions(quiz_data["questions"], quiz_data["metadata"])
            st.download_button(
                label="Download as PDF",
                data=pdf_buffer,
                file_name=f"{subject_slug}_{timestamp_slug}.pdf",
                mime="application/pdf"
            )

        # Examples section
        st.markdown("---")
        st.header("💡 Example Prompts")
        
        examples = [
            {
                "Subject": "NLP",
                "Topic": "Tokenization",
                "Difficulty": "medium",
                "Type": "MCQ",
                "Description": "Generate an MCQ about text tokenization"
            },
            {
                "Subject": "Calculus",
                "Topic": "Derivatives",
                "Difficulty": "hard",
                "Type": "Short Answer",
                "Description": "Generate a hard short answer question about derivatives"
            },
            {
                "Subject": "Applied Physics",
                "Topic": "Mechanics",
                "Difficulty": "easy",
                "Type": "MCQ",
                "Description": "Generate an easy MCQ about mechanics"
            },
            {
                "Subject": "ICT",
                "Topic": "Network Security",
                "Difficulty": "medium",
                "Type": "Long Answer",
                "Description": "Generate a comprehensive long answer question"
            }
        ]
        
        for i, example in enumerate(examples):
            with st.expander(f" Example {i+1}: {example['Description']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Subject:** {example['Subject']}")
                    st.write(f"**Topic:** {example['Topic']}")
                with col2:
                    st.write(f"**Difficulty:** {example['Difficulty']}")
                    st.write(f"**Type:** {example['Type']}")
                
                if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                    st.session_state.example_subject = example['Subject']
                    st.session_state.example_topic = example['Topic']
                    st.session_state.example_difficulty = example['Difficulty']
                    st.session_state.example_type = example['Type']
                    st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 2rem;'>
                <p>Made with  using Streamlit and T5 Transformer</p>
                <p>Fine-tuned Quiz Generator Model</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()


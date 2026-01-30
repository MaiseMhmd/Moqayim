import streamlit as st
import re
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from collections import Counter
import math
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import requests  # ADD THIS LINE

# Load environment variables
load_dotenv()

# Try to import optional dependencies
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


# Add this BEFORE init_session_state()
def get_session_from_api(session_id):
    """Fetch session from Flask LTI server"""
    # REPLACE THIS URL with your actual Render URL once deployed
    FLASK_API_URL = os.environ.get('FLASK_API_URL', 'https://moqayim.onrender.com')
    
    try:
        response = requests.get(
            f"{FLASK_API_URL}/api/session/{session_id}",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Session not found: {session_id}")
            return None
    except Exception as e:
        print(f"Error fetching session: {str(e)}")
        return None


def init_session_state():
    """Initialize session state variables"""
    
    # Check for LTI session parameter
    query_params = st.query_params
    if 'session' in query_params:
        session_id = query_params['session']
        role = query_params.get('role', ['student'])[0] if isinstance(query_params.get('role'), list) else query_params.get('role', 'student')
        
        # Load LTI session data from Flask API
        lti_data = get_session_from_api(session_id)
        if lti_data:
            st.session_state.lti_session = lti_data
            st.session_state.user_role = role
            
            # Show info banner
            st.info(f"🔗 Connected via LTI | Role: {role} | Course: {lti_data['course_id']}")
    
    # ... rest of your existing init code

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RubricCriterion:
    """Single grading criterion with marks allocation"""
    name: str
    marks: int
    key_points: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

@dataclass
class AssessmentConfig:
    """Teacher-defined assessment configuration"""
    question: str
    model_answer: str
    rubric: List[RubricCriterion]
    language: Literal["en", "ar"] = "en"

@dataclass
class StudentSubmission:
    """Student's submitted answer"""
    answer: str
    language: Literal["en", "ar"] = "en"

@dataclass
class CriterionResult:
    """Grading result for one criterion"""
    criterion_name: str
    status: Literal["met", "partial", "not_met"]
    marks_awarded: int
    marks_total: int
    justification: str

@dataclass
class GradingReport:
    """Complete grading report"""
    criterion_results: List[CriterionResult]
    total_score: int
    max_score: int
    feedback: List[str]

# ============================================================================
# GROQ OCR PROCESSING
# ============================================================================

class GroqOCR:
    """Extract text from images using Groq's Llama Maverick vision model"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client with API key"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Install with: pip install groq")
        
        self.client = Groq(api_key=api_key) if api_key else Groq()
    
    @staticmethod
    def image_to_base64_url(image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL"""
        buffered = io.BytesIO()
        # Convert to RGB if necessary (removes alpha channel)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        # Use PNG for better text preservation
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def extract_text(self, image: Image.Image, language: str = "en") -> str:
        """Extract text from image using Groq Llama Maverick"""
        try:
            # Check image size limits
            width, height = image.size
            total_pixels = width * height
            if total_pixels > 33177600:  # 33 megapixels limit
                # Resize image if too large
                import math
                scale = math.sqrt(33177600 / total_pixels)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert image to base64 data URL
            image_url = self.image_to_base64_url(image)
            
            # Check base64 size (4MB limit)
            base64_size = len(image_url.encode('utf-8'))
            if base64_size > 4 * 1024 * 1024:  # 4MB
                raise Exception(f"Image too large after encoding: {base64_size} bytes")
            
            # Prepare language-specific prompt
            if language == "ar":
                prompt = "استخرج كل النص الموجود في هذه الصورة بدقة عالية جداً. يشمل ذلك النص المكتوب بخط اليد والنص المطبوع والكتابة اليدوية غير الواضحة. اقرأ بعناية حرفاً بحرف وكلمة بكلمة. إذا كان النص مكتوباً بخط اليد، حاول التعرف على الحروف والكلمات حتى لو كانت مشوهة أو غير منتظمة. أعد النص المستخرج فقط بدون أي تعليقات أو تنسيق إضافي أو شرح."
            else:
                prompt = "Extract all text from this image with very high accuracy. This includes handwritten text, printed text, and unclear handwriting. Read carefully letter by letter and word by word. If the text is handwritten, try to recognize the letters and words even if they are distorted or irregular. Return only the extracted text without any comments, additional formatting, or explanations."
            
            # Call Groq API
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Lower temperature for more accurate OCR
                max_completion_tokens=2048,  # Increase token limit for longer text
                top_p=1,
                stream=False,
                stop=None
            )
            
            # Extract and return the text
            extracted_text = completion.choices[0].message.content
            if not extracted_text or extracted_text.strip() == "":
                raise Exception("No text was extracted from the image. The image may not contain readable text or the quality may be too poor.")
            return extracted_text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise Exception("API rate limit exceeded. Please try again later.")
            elif "size" in error_msg.lower() or "large" in error_msg.lower():
                raise Exception("Image file is too large. Please use a smaller image or reduce the resolution.")
            else:
                raise Exception(f"OCR failed: {error_msg}")

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

class DocumentProcessor:
    """Extract text from PDFs and images using Groq OCR"""
    
    @staticmethod
    def extract_text_from_image(image: Image.Image, language: str = "en", api_key: Optional[str] = None) -> str:
        """Extract text from a PIL Image using Groq"""
        if not GROQ_AVAILABLE:
            raise Exception("Groq library is not installed. Please install it with: pip install groq")
        
        try:
            ocr = GroqOCR(api_key=api_key)
            return ocr.extract_text(image, language)
        except Exception as e:
            raise Exception(f"Image OCR failed: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf_text(pdf_file) -> str:
        """Extract text from PDF using PyPDF2 (for text-based PDFs)"""
        if not PYPDF2_AVAILABLE:
            return ""
        
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            return "\n\n".join(text_parts)
        except Exception as e:
            return ""
    
    @staticmethod
    def extract_text_from_pdf_ocr(pdf_file, language: str = "en", api_key: Optional[str] = None) -> str:
        """Extract text from PDF using Groq OCR (for scanned PDFs)"""
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("pdf2image is not installed. Install with: pip install pdf2image")
        
        if not GROQ_AVAILABLE:
            raise Exception("Groq library is not installed. Install with: pip install groq")
        
        try:
            pdf_file.seek(0)
            # Convert PDF to images with higher DPI for better accuracy
            images = pdf2image.convert_from_bytes(pdf_file.read(), dpi=300)
            
            # Initialize Groq OCR
            ocr = GroqOCR(api_key=api_key)
            
            # Extract text from each page
            full_text = []
            for i, image in enumerate(images):
                page_text = ocr.extract_text(image, language)
                if page_text:
                    full_text.append(f"--- Page {i+1} ---\n{page_text}")
            
            return "\n\n".join(full_text)
        except Exception as e:
            raise Exception(f"PDF OCR failed: {str(e)}")
    
    @staticmethod
    def process_uploaded_file(uploaded_file, language: str = "en", api_key: Optional[str] = None) -> str:
        """Process uploaded file (PDF or image) using Groq OCR"""
        if uploaded_file is None:
            return ""
        
        file_type = uploaded_file.type
        
        try:
            if file_type == "application/pdf":
                # First try to extract text directly (for text-based PDFs)
                text = DocumentProcessor.extract_text_from_pdf_text(uploaded_file)
                
                # If no text extracted, it's a scanned PDF - use Groq OCR
                if not text or len(text.strip()) < 50:
                    st.info("📄 Processing scanned PDF with Groq OCR...")
                    text = DocumentProcessor.extract_text_from_pdf_ocr(uploaded_file, language, api_key)
                
                return text
                
            elif file_type.startswith("image/"):
                image = Image.open(uploaded_file)
                st.info("🔍 Extracting text with Groq OCR...")
                return DocumentProcessor.extract_text_from_image(image, language, api_key)
            else:
                st.error("Unsupported file type. Please upload PDF or image files.")
                return ""
        except Exception as e:
            st.error(f"File processing error: {str(e)}")
            return ""
    
    @staticmethod
    def smart_split_qa(extracted_text: str) -> tuple[str, str]:
        """
        Intelligently split extracted text into question and answer.
        Looks for common markers and patterns.
        """
        text = extracted_text.strip()
        
        # Remove page markers
        text = re.sub(r'---\s*Page\s+\d+\s*---', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Common patterns for question/answer separation (case insensitive)
        patterns = [
            # Pattern 1: Question: ... Answer:
            (r'(?i)question[:\s]+(.*?)(?:answer[:\s]+|model\s+answer[:\s]+|solution[:\s]+)(.*)', 1, 2),
            # Pattern 2: Q: ... A:
            (r'(?i)q[:\.\s]+(.*?)(?:a[:\.\s]+|ans[:\.\s]+)(.*)', 1, 2),
            # Pattern 3: Just "Answer:" or "Solution:"
            (r'(?i)(.*?)(?:answer[:\s]+|solution[:\s]+|model\s+answer[:\s]+)(.*)', 1, 2),
        ]
        
        for pattern, q_group, a_group in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                question = match.group(q_group).strip()
                answer = match.group(a_group).strip()
                
                # Only accept if both parts are substantial
                if len(question) > 10 and len(answer) > 10:
                    return question, answer
        
        # If no pattern found, try to split at roughly midpoint
        lines = text.split('\n')
        if len(lines) > 3:
            mid = len(lines) // 2
            question = '\n'.join(lines[:mid]).strip()
            answer = '\n'.join(lines[mid:]).strip()
        else:
            # If very few lines, split by sentence count
            sentences = re.split(r'[.!?]+', text)
            mid = len(sentences) // 2
            question = '.'.join(sentences[:mid]).strip()
            answer = '.'.join(sentences[mid:]).strip()
        
        return question, answer

# ============================================================================
# GRADING ENGINE
# ============================================================================

class GradingEngine:
    """Deterministic rubric-based grading engine"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split normalized text into tokens"""
        return GradingEngine.normalize_text(text).split()
    
    @staticmethod
    def calculate_overlap(student_tokens: List[str], reference_tokens: List[str]) -> float:
        """Calculate token overlap ratio"""
        if not reference_tokens:
            return 0.0
        student_set = set(student_tokens)
        reference_set = set(reference_tokens)
        intersection = len(student_set & reference_set)
        return intersection / len(reference_set)
    
    @staticmethod
    def calculate_cosine_similarity(student_tokens: List[str], reference_tokens: List[str]) -> float:
        """Calculate TF-based cosine similarity"""
        if not student_tokens or not reference_tokens:
            return 0.0
        
        student_counts = Counter(student_tokens)
        reference_counts = Counter(reference_tokens)
        
        all_terms = set(student_counts.keys()) | set(reference_counts.keys())
        
        dot_product = sum(student_counts[term] * reference_counts[term] for term in all_terms)
        
        student_mag = math.sqrt(sum(count ** 2 for count in student_counts.values()))
        reference_mag = math.sqrt(sum(count ** 2 for count in reference_counts.values()))
        
        if student_mag == 0 or reference_mag == 0:
            return 0.0
        
        return dot_product / (student_mag * reference_mag)
    
    @staticmethod
    def check_key_points(student_text: str, key_points: List[str]) -> tuple[int, int, List[str]]:
        """Check how many key points are present"""
        if not key_points:
            return 0, 0, []
        
        student_norm = GradingEngine.normalize_text(student_text)
        found_points = []
        
        for point in key_points:
            point_norm = GradingEngine.normalize_text(point)
            if point_norm in student_norm or any(term in student_norm for term in point_norm.split()):
                found_points.append(point)
        
        return len(found_points), len(key_points), found_points
    
    @staticmethod
    def evaluate_criterion(
        student_answer: str,
        model_answer: str,
        criterion: RubricCriterion
    ) -> CriterionResult:
        """Evaluate a single criterion"""
        
        student_tokens = GradingEngine.tokenize(student_answer)
        model_tokens = GradingEngine.tokenize(model_answer)
        
        # Calculate similarity metrics
        overlap_score = GradingEngine.calculate_overlap(student_tokens, model_tokens)
        cosine_score = GradingEngine.calculate_cosine_similarity(student_tokens, model_tokens)
        
        # Check key points
        found_count, total_count, found_points = GradingEngine.check_key_points(
            student_answer, criterion.key_points
        )
        
        # Calculate combined score
        if total_count > 0:
            key_point_score = found_count / total_count
        else:
            key_point_score = (overlap_score + cosine_score) / 2
        
        # Determine status and marks
        if key_point_score >= 0.8:
            status = "met"
            marks = criterion.marks
        elif key_point_score >= 0.4:
            status = "partial"
            marks = max(1, criterion.marks // 2) if criterion.marks > 1 else 0
        else:
            status = "not_met"
            marks = 0
        
        # Generate justification
        justification = GradingEngine.generate_justification(
            criterion, found_count, total_count, found_points, 
            key_point_score, status
        )
        
        return CriterionResult(
            criterion_name=criterion.name,
            status=status,
            marks_awarded=marks,
            marks_total=criterion.marks,
            justification=justification
        )
    
    @staticmethod
    def generate_justification(
        criterion: RubricCriterion,
        found_count: int,
        total_count: int,
        found_points: List[str],
        score: float,
        status: str
    ) -> str:
        """Generate human-readable justification"""
        
        if status == "met":
            if total_count > 0:
                return f"✓ All required elements present ({found_count}/{total_count} key points identified)."
            return f"✓ Criterion fully satisfied with strong alignment to model answer."
        
        elif status == "partial":
            if total_count > 0:
                missing = total_count - found_count
                return f"⚬ Partially satisfied: {found_count}/{total_count} key points present. Missing {missing} element(s)."
            return f"⚬ Partially satisfied: some relevant content present but incomplete coverage."
        
        else:  # not_met
            if total_count > 0:
                return f"✗ Not satisfied: {found_count}/{total_count} key points identified. Missing critical elements."
            return f"✗ Not satisfied: insufficient alignment with expected answer."
    
    @staticmethod
    def generate_feedback(
        criterion_results: List[CriterionResult],
        rubric: List[RubricCriterion]
    ) -> List[str]:
        """Generate actionable feedback based on results"""
        
        feedback = []
        
        for result in criterion_results:
            if result.status == "not_met":
                criterion = next(c for c in rubric if c.name == result.criterion_name)
                if criterion.hints:
                    feedback.append(f"Add {criterion.name.lower()}: {criterion.hints[0]}")
                else:
                    feedback.append(f"Include {criterion.name.lower()} in your answer.")
            
            elif result.status == "partial":
                criterion = next(c for c in rubric if c.name == result.criterion_name)
                if criterion.key_points:
                    feedback.append(f"Strengthen {criterion.name.lower()}: ensure all required details are covered.")
                elif criterion.hints:
                    feedback.append(f"Improve {criterion.name.lower()}: {criterion.hints[0]}")
        
        if not feedback:
            feedback.append("Excellent work! All criteria met.")
        
        return feedback[:4]
    
    @staticmethod
    def grade_submission(
        config: AssessmentConfig,
        submission: StudentSubmission
    ) -> GradingReport:
        """Grade a student submission against assessment configuration"""
        
        criterion_results = []
        
        for criterion in config.rubric:
            result = GradingEngine.evaluate_criterion(
                submission.answer,
                config.model_answer,
                criterion
            )
            criterion_results.append(result)
        
        total_score = sum(r.marks_awarded for r in criterion_results)
        max_score = sum(r.marks_total for r in criterion_results)
        
        feedback = GradingEngine.generate_feedback(criterion_results, config.rubric)
        
        return GradingReport(
            criterion_results=criterion_results,
            total_score=total_score,
            max_score=max_score,
            feedback=feedback
        )

# ============================================================================
# UI TRANSLATIONS
# ============================================================================

TRANSLATIONS = {
    "en": {
        "app_title": "Moqayim - Short Answer Grading",
        "page1_title": "Page 1: Teacher Setup",
        "page2_title": "Page 2: Student Answer",
        "page3_title": "Page 3: Grading Results",
        "question_label": "Question",
        "model_answer_label": "Model Answer",
        "rubric_label": "Rubric Criteria",
        "criterion_name": "Criterion Name",
        "criterion_marks": "Marks",
        "criterion_desc": "Description",
        "key_points": "Key Points (comma-separated, optional)",
        "hints": "Hints (comma-separated, optional)",
        "add_criterion": "Add Criterion",
        "next": "Next →",
        "submit_answer": "Submit Answer",
        "view_results": "View Results",
        "back": "← Back",
        "reset": "Reset All",
        "total_score": "Total Score",
        "feedback": "Feedback",
        "student_answer_label": "Your Answer",
        "language": "Language",
        "upload_document": "Upload Document (PDF or Image)",
        "extract_text": "Extract Text from Document",
        "extracted_text": "Extracted Text",
        "processing": "Processing document...",
        "upload_hint": "Upload a PDF or image containing the question and model answer",
        "copy_text": "Copy Extracted Text",
        "text_copied": "Text copied to clipboard!",
        "use_extracted": "Use Extracted Text"
    },
    "ar": {
        "app_title": "مُقَيِّم - تصحيح الإجابات القصيرة",
        "page1_title": "الصفحة 1: إعداد المعلم",
        "page2_title": "الصفحة 2: إجابة الطالب",
        "page3_title": "الصفحة 3: نتائج التصحيح",
        "question_label": "السؤال",
        "model_answer_label": "الإجابة النموذجية",
        "rubric_label": "معايير التقييم",
        "criterion_name": "اسم المعيار",
        "criterion_marks": "الدرجات",
        "criterion_desc": "الوصف",
        "key_points": "النقاط الرئيسية (مفصولة بفواصل، اختياري)",
        "hints": "التلميحات (مفصولة بفواصل، اختياري)",
        "add_criterion": "إضافة معيار",
        "next": "التالي ←",
        "submit_answer": "إرسال الإجابة",
        "view_results": "عرض النتائج",
        "back": "→ رجوع",
        "reset": "إعادة تعيين",
        "total_score": "الدرجة الإجمالية",
        "feedback": "التغذية الراجعة",
        "student_answer_label": "إجابتك",
        "language": "اللغة",
        "upload_document": "تحميل مستند (PDF أو صورة)",
        "extract_text": "استخراج النص من المستند",
        "extracted_text": "النص المستخرج",
        "processing": "معالجة المستند...",
        "upload_hint": "قم بتحميل PDF أو صورة تحتوي على السؤال والإجابة النموذجية",
        "copy_text": "نسخ النص المستخرج",
        "text_copied": "تم نسخ النص!",
        "use_extracted": "استخدام النص المستخرج"
    }
}

def t(key: str, default_lang: str = "en") -> str:
    """Get translation for current language"""
    try:
        lang = st.session_state.get("language", default_lang)
    except:
        lang = default_lang
    return TRANSLATIONS[lang].get(key, key)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "page": 1,
        "language": "en",
        "assessment_config": None,
        "student_submission": None,
        "grading_report": None,
        "num_criteria": 3,
        "extracted_question": "",
        "extracted_answer": "",
        "student_extracted_answer": "",
        "last_teacher_file": None,
        "last_student_file": None,
        "last_extracted_text": "",
        "last_student_extracted_text": ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_app():
    """Reset all session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def page1_teacher_setup():
    """Page 1: Teacher Setup"""
    st.header(t("page1_title"))
    
    # Get API key from environment
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Check dependencies
    if not GROQ_AVAILABLE:
        st.error("❌ Groq library is not installed.")
        st.code("pip install groq", language="bash")
        return
    
    if not api_key:
        st.warning("⚠️ Please set your GROQ_API_KEY environment variable to use OCR features.")
    
    st.divider()
    
    # Document Upload Section
    st.subheader("📄 " + t("upload_document"))
    st.caption(t("upload_hint"))
    
    uploaded_file = st.file_uploader(
        t("upload_document"),
        type=["pdf", "png", "jpg", "jpeg"],
        key="teacher_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_file and api_key:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.get("last_teacher_file") != file_key:
            with st.spinner(t("processing")):
                try:
                    extracted_text = DocumentProcessor.process_uploaded_file(
                        uploaded_file, 
                        st.session_state.language,
                        api_key
                    )
                    
                    if extracted_text:
                        question, answer = DocumentProcessor.smart_split_qa(extracted_text)
                        st.session_state.extracted_question = question
                        st.session_state.extracted_answer = answer
                        st.session_state.last_teacher_file = file_key
                        st.session_state.last_extracted_text = extracted_text
                        st.success("✓ Text extracted and fields auto-filled!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            if "last_extracted_text" in st.session_state and st.session_state.last_extracted_text:
                with st.expander("📝 " + t("extracted_text"), expanded=False):
                    st.text_area(
                        "Raw Extracted Text",
                        st.session_state.last_extracted_text,
                        height=200,
                        key="teacher_extracted_display"
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("📋 " + t("copy_text"), key="copy_teacher"):
                            st.success(t("text_copied"))
    
    st.divider()
    
    # Question input
    question = st.text_area(
        t("question_label"),
        value=st.session_state.get("extracted_question", ""),
        height=100,
        placeholder="e.g., Explain the process of photosynthesis.",
        key="question_input"
    )
    
    # Model answer
    model_answer = st.text_area(
        t("model_answer_label"),
        value=st.session_state.get("extracted_answer", ""),
        height=150,
        placeholder="e.g., Photosynthesis is the process by which plants convert light energy into chemical energy...",
        key="model_answer_input"
    )
    
    # Rubric criteria
    st.subheader(t("rubric_label"))
    
    rubric = []
    
    for i in range(st.session_state.num_criteria):
        with st.expander(f"{t('criterion_name')} {i+1}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                name = st.text_input(
                    t("criterion_name"),
                    key=f"crit_name_{i}",
                    placeholder="e.g., Definition"
                )
            with col2:
                marks = st.number_input(
                    t("criterion_marks"),
                    min_value=1,
                    max_value=10,
                    value=2,
                    key=f"crit_marks_{i}"
                )
            
            key_points_str = st.text_input(
                t("key_points"),
                key=f"crit_kp_{i}",
                placeholder="e.g., light energy, chemical energy, glucose"
            )
            
            hints_str = st.text_input(
                t("hints"),
                key=f"crit_hints_{i}",
                placeholder="e.g., Include the chemical equation"
            )
            
            if name:
                key_points = [kp.strip() for kp in key_points_str.split(",") if kp.strip()]
                hints = [h.strip() for h in hints_str.split(",") if h.strip()]
                
                rubric.append(RubricCriterion(
                    name=name,
                    marks=marks,
                    key_points=key_points,
                    hints=hints
                ))
    
    if st.button(t("add_criterion")):
        st.session_state.num_criteria += 1
        st.rerun()
    
    # Navigation
    if st.button(t("next"), type="primary"):
        if not question or not model_answer or len(rubric) == 0:
            st.error("Please fill in all required fields.")
        else:
            st.session_state.assessment_config = AssessmentConfig(
                question=question,
                model_answer=model_answer,
                rubric=rubric,
                language=st.session_state.language
            )
            st.session_state.page = 2
            st.rerun()

def page2_student_answer():
    """Page 2: Student Answer Simulation"""
    st.header(t("page2_title"))
    
    config = st.session_state.assessment_config
    
    # Display question
    st.subheader(t("question_label"))
    st.info(config.question)
    
    st.divider()
    
    # Document Upload Section for Student
    st.subheader("📄 " + t("upload_document"))
    st.caption("Upload a scanned document or image of the student's handwritten answer")
    
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.warning("⚠️ Please set your GROQ_API_KEY environment variable to use OCR features.")
    
    uploaded_student_file = st.file_uploader(
        t("upload_document"),
        type=["pdf", "png", "jpg", "jpeg"],
        key="student_upload",
        label_visibility="collapsed"
    )
    
    if uploaded_student_file and api_key:
        file_key = f"{uploaded_student_file.name}_{uploaded_student_file.size}"
        
        if st.session_state.get("last_student_file") != file_key:
            with st.spinner(t("processing")):
                try:
                    extracted_student_text = DocumentProcessor.process_uploaded_file(
                        uploaded_student_file, 
                        st.session_state.language,
                        api_key
                    )
                    
                    if extracted_student_text:
                        st.session_state.student_extracted_answer = extracted_student_text
                        st.session_state.last_student_file = file_key
                        st.session_state.last_student_extracted_text = extracted_student_text
                        st.success("✓ Student answer extracted and auto-filled!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            if "last_student_extracted_text" in st.session_state and st.session_state.last_student_extracted_text:
                with st.expander("📝 " + t("extracted_text"), expanded=False):
                    st.text_area(
                        "Raw Extracted Text",
                        st.session_state.last_student_extracted_text,
                        height=200,
                        key="student_extracted_display"
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("📋 " + t("copy_text"), key="copy_student"):
                            st.success(t("text_copied"))
    
    st.divider()
    
    # Student answer input
    student_answer = st.text_area(
        t("student_answer_label"),
        value=st.session_state.get("student_extracted_answer", ""),
        height=200,
        placeholder="Type your answer here...",
        key="student_answer_input"
    )
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("back")):
            st.session_state.page = 1
            st.rerun()
    
    with col2:
        if st.button(t("submit_answer"), type="primary"):
            if not student_answer.strip():
                st.error("Please provide an answer.")
            else:
                submission = StudentSubmission(
                    answer=student_answer,
                    language=st.session_state.language
                )
                
                # Grade the submission
                report = GradingEngine.grade_submission(config, submission)
                
                st.session_state.student_submission = submission
                st.session_state.grading_report = report
                st.session_state.page = 3
                st.rerun()

def page3_results():
    """Page 3: Grading Results"""
    st.header(t("page3_title"))
    
    report = st.session_state.grading_report
    config = st.session_state.assessment_config
    submission = st.session_state.student_submission
    
    # Score summary
    percentage = (report.total_score / report.max_score * 100) if report.max_score > 0 else 0
    
    st.metric(
        label=t("total_score"),
        value=f"{report.total_score}/{report.max_score}",
        delta=f"{percentage:.1f}%"
    )
    
    # Criterion breakdown
    st.subheader("Criterion Breakdown")
    
    for result in report.criterion_results:
        status_color = {
            "met": "🟢",
            "partial": "🟡",
            "not_met": "🔴"
        }[result.status]
        
        with st.expander(
            f"{status_color} {result.criterion_name} - {result.marks_awarded}/{result.marks_total}",
            expanded=True
        ):
            st.write(result.justification)
    
    # Feedback
    st.subheader(t("feedback"))
    for i, fb in enumerate(report.feedback, 1):
        st.write(f"{i}. {fb}")
    
    # Show student answer
    with st.expander("View Student Answer"):
        st.write(submission.answer)
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("back")):
            st.session_state.page = 2
            st.rerun()
    
    with col2:
        if st.button(t("reset"), type="secondary"):
            reset_app()
            st.rerun()

def main():
    """Main application"""
    st.set_page_config(
        page_title="Moqayim Grading",
        page_icon="📝",
        layout="wide"
    )
    
    init_session_state()
    
    # Header with language toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title(t("app_title"))
    
    with col2:
        lang = st.selectbox(
            t("language"),
            options=["en", "ar"],
            format_func=lambda x: "English" if x == "en" else "العربية",
            key="language_selector",
            on_change=lambda: st.session_state.update({"language": st.session_state.language_selector})
        )
    
    with col3:
        if st.button(t("reset"), key="header_reset"):
            reset_app()
            st.rerun()
    
    st.divider()
    
    # Page routing
    if st.session_state.page == 1:
        page1_teacher_setup()
    elif st.session_state.page == 2:
        page2_student_answer()
    elif st.session_state.page == 3:
        page3_results()

if __name__ == "__main__":

    main()

import streamlit as st
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from collections import Counter
import math
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv

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

from database import get_session

def init_session_state():
    """Initialize session state variables"""
    
    # Check for LTI session parameter
    query_params = st.query_params
    if 'session' in query_params:
        session_id = query_params['session']
        role = query_params.get('role', 'student')
        
        # Load LTI session data
        lti_data = get_session(session_id)
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
    """Single grading criterion with marks allocation and its own model answer"""
    name: str
    marks: int
    model_answer: str = ""
    key_points: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)

@dataclass
class AssessmentConfig:
    """Teacher-defined assessment configuration"""
    question: str
    rubric: List[RubricCriterion]
    language: Literal["en", "ar"] = "en"

@dataclass
class CriterionSubmission:
    """Student's answer for one criterion"""
    criterion_name: str
    answer: str

@dataclass
class StudentSubmission:
    """Student's submitted answers for all criteria"""
    criterion_answers: List[CriterionSubmission]
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

# ============================================================================
# GRADING ENGINE
# ============================================================================

class GroqGradingEngine:
    """LLM-powered grading engine using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not installed. Run: pip install groq")
        
        self.client = Groq(api_key=api_key) if api_key else Groq()
        self.model = "llama-3.3-70b-versatile"  # Best for reasoning tasks
    
    def grade_criterion(
        self,
        question: str,
        criterion: RubricCriterion,
        student_answer: str,
        language: str = "en"
    ) -> CriterionResult:
        """
        Grade a single criterion using Groq LLM.
        This is called ONCE for EACH criterion independently.
        """
        
        # Build the grading prompt
        prompt = self._build_grading_prompt(
            question=question,
            criterion=criterion,
            student_answer=student_answer,
            language=language
        )
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(language)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent grading
                max_tokens=500,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Parse the JSON response
            result_json = json.loads(response.choices[0].message.content)
            
            # Extract grading information
            marks_awarded = int(result_json.get("marks_awarded", 0))
            marks_awarded = max(0, min(marks_awarded, criterion.marks))  # Clamp to valid range
            
            justification = result_json.get("justification", "No justification provided")
            
            # Determine status
            percentage = marks_awarded / criterion.marks if criterion.marks > 0 else 0
            if percentage >= 0.85:
                status = "met"
            elif percentage >= 0.40:
                status = "partial"
            else:
                status = "not_met"
            
            return CriterionResult(
                criterion_name=criterion.name,
                status=status,
                marks_awarded=marks_awarded,
                marks_total=criterion.marks,
                justification=justification
            )
            
        except Exception as e:
            # Fallback on error
            return CriterionResult(
                criterion_name=criterion.name,
                status="not_met",
                marks_awarded=0,
                marks_total=criterion.marks,
                justification=f"Error during grading: {str(e)}"
            )
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt for the LLM"""
        
        if language == "ar":
            return """أنت مصحح خبير للإجابات القصيرة. مهمتك هي تقييم إجابات الطلاب بناءً على معايير محددة.

قواعد التصحيح:
1. قيّم فهم الطالب المفاهيمي، وليس فقط وجود الكلمات المفتاحية
2. تحقق من الدقة العلمية - اكتشف الأخطاء المفاهيمية
3. اكتشف العلاقات السببية المعكوسة (مثال: "أ يسبب ب" مقابل "ب يسبب أ")
4. أعط علامات جزئية للفهم الجزئي
5. كن عادلاً ومتسقاً
6. أرجع النتائج بصيغة JSON فقط

صيغة JSON المطلوبة:
{
  "marks_awarded": <رقم من 0 إلى الحد الأقصى>,
  "justification": "<تفسير مختصر بالعربية>"
}"""
        else:
            return """You are an expert grader for short-answer questions. Your job is to evaluate student answers against specific criteria.

Grading Rules:
1. Evaluate conceptual understanding, not just keyword presence
2. Check for scientific/factual accuracy - catch conceptual errors
3. Detect reversed causation (e.g., "A causes B" vs "B causes A")
4. Award partial credit for partial understanding
5. Be fair and consistent
6. Return results in JSON format only

Required JSON format:
{
  "marks_awarded": <number from 0 to max>,
  "justification": "<brief explanation in English>"
}"""
    
    def _build_grading_prompt(
        self,
        question: str,
        criterion: RubricCriterion,
        student_answer: str,
        language: str
    ) -> str:
        """Build the grading prompt for a specific criterion"""
        
        if language == "ar":
            prompt = f"""**السؤال:**
{question}

**المعيار المراد تقييمه:**
- الاسم: {criterion.name}
- الدرجة القصوى: {criterion.marks}

**الإجابة النموذجية لهذا المعيار:**
{criterion.model_answer}

**إجابة الطالب لهذا المعيار:**
{student_answer}
"""
            
            if criterion.key_points:
                prompt += f"\n**النقاط الرئيسية المطلوبة:**\n"
                for i, point in enumerate(criterion.key_points, 1):
                    prompt += f"{i}. {point}\n"
            
            if criterion.hints:
                prompt += f"\n**إرشادات التصحيح:**\n"
                for hint in criterion.hints:
                    prompt += f"- {hint}\n"
            
            prompt += f"""
**المطلوب:**
قيّم إجابة الطالب لهذا المعيار فقط ({criterion.name}). أعط درجة من 0 إلى {criterion.marks}.

اعتبارات مهمة:
- هل المفهوم صحيح علمياً؟
- هل تم ذكر النقاط الرئيسية؟
- هل هناك أخطاء مفاهيمية؟
- هل الإجابة كاملة أم ناقصة؟

أرجع النتيجة بصيغة JSON فقط."""
        
        else:  # English
            prompt = f"""**Question:**
{question}

**Criterion to Evaluate:**
- Name: {criterion.name}
- Maximum Marks: {criterion.marks}

**Model Answer for this Criterion:**
{criterion.model_answer}

**Student Answer for this Criterion:**
{student_answer}
"""
            
            if criterion.key_points:
                prompt += f"\n**Required Key Points:**\n"
                for i, point in enumerate(criterion.key_points, 1):
                    prompt += f"{i}. {point}\n"
            
            if criterion.hints:
                prompt += f"\n**Grading Hints:**\n"
                for hint in criterion.hints:
                    prompt += f"- {hint}\n"
            
            prompt += f"""
**Task:**
Evaluate the student's answer for THIS criterion only ({criterion.name}). Award marks from 0 to {criterion.marks}.

Important considerations:
- Is the concept scientifically/factually correct?
- Are the key points addressed?
- Are there any conceptual errors?
- Is the answer complete or partial?

Return result in JSON format only."""
        
        return prompt
    
    def generate_feedback(
        self,
        criterion_results: List[CriterionResult],
        rubric: List[RubricCriterion],
        language: str = "en"
    ) -> List[str]:
        """Generate actionable feedback based on results"""
        
        feedback = []
        
        for result in criterion_results:
            criterion = next(c for c in rubric if c.name == result.criterion_name)
            
            percentage = result.marks_awarded / result.marks_total if result.marks_total > 0 else 0
            
            if percentage < 0.5:  # Less than 50%
                if language == "ar":
                    if criterion.hints:
                        feedback.append(f"حسّن {criterion.name}: {criterion.hints[0]}")
                    else:
                        feedback.append(f"أضف المزيد من التفاصيل في {criterion.name}")
                else:
                    if criterion.hints:
                        feedback.append(f"Improve {criterion.name}: {criterion.hints[0]}")
                    else:
                        feedback.append(f"Add more detail to {criterion.name}")
        
        if not feedback:
            if language == "ar":
                feedback.append("عمل ممتاز! جميع المعايير تم تحقيقها بشكل جيد.")
            else:
                feedback.append("Excellent work! All criteria well addressed.")
        
        return feedback[:5]
    
    def grade_submission(
        self,
        question: str,
        rubric: List[RubricCriterion],
        criterion_answers: List[CriterionSubmission],
        language: str = "en"
    ) -> GradingReport:
        """
        Grade a complete student submission.
        Calls the LLM ONCE for EACH criterion independently.
        """
        
        criterion_results = []
        
        # Grade each criterion independently
        for criterion in rubric:
            # Find the student's answer for this criterion
            student_answer = next(
                (ca.answer for ca in criterion_answers if ca.criterion_name == criterion.name),
                ""
            )
            
            result = self.grade_criterion(
                question=question,
                criterion=criterion,
                student_answer=student_answer,
                language=language
            )
            
            criterion_results.append(result)
        
        # Calculate totals
        total_score = sum(r.marks_awarded for r in criterion_results)
        max_score = sum(r.marks_total for r in criterion_results)
        
        # Generate feedback
        feedback = self.generate_feedback(criterion_results, rubric, language)
        
        return GradingReport(
            criterion_results=criterion_results,
            total_score=total_score,
            max_score=max_score,
            feedback=feedback
        )


class GradingEngine:
    """
    Wrapper class that uses Groq LLM for grading.
    Drop-in replacement for your existing GradingEngine.
    """
    
    @staticmethod
    def grade_submission(config: AssessmentConfig, submission: StudentSubmission) -> GradingReport:
        """
        Grade submission using Groq LLM.
        
        Args:
            config: AssessmentConfig with question and rubric
            submission: StudentSubmission with criterion answers
        
        Returns:
            GradingReport with criterion results
        """
        
        # Get API key from environment
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Initialize Groq grading engine
        grader = GroqGradingEngine(api_key=api_key)
        
        # Grade the submission
        report = grader.grade_submission(
            question=config.question,
            rubric=config.rubric,
            criterion_answers=submission.criterion_answers,
            language=config.language
        )
        
        return report

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
        "criterion_model_answer": "Model Answer for this Criterion",
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
        "use_extracted": "Use Extracted Text",
        "answer_for_criterion": "Answer for"
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
        "criterion_model_answer": "الإجابة النموذجية لهذا المعيار",
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
        "use_extracted": "استخدام النص المستخرج",
        "answer_for_criterion": "الإجابة لـ"
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
        "num_criteria": 1,  # Start with 1 criterion
        "extracted_question": "",
        "last_teacher_file": None,
        "last_student_file": None,
        "last_extracted_text": "",
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
    
    # Question input
    question = st.text_area(
        t("question_label"),
        value=st.session_state.get("extracted_question", ""),
        height=100,
        placeholder="e.g., Explain the process of photosynthesis.",
        key="question_input"
    )
    
    st.divider()
    
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
                    placeholder="e.g., Definition of Photosynthesis"
                )
            with col2:
                marks = st.number_input(
                    t("criterion_marks"),
                    min_value=1,
                    max_value=10,
                    value=2,
                    key=f"crit_marks_{i}"
                )
            
            # Model answer for this criterion
            model_answer = st.text_area(
                t("criterion_model_answer"),
                key=f"crit_model_{i}",
                height=100,
                placeholder="e.g., Photosynthesis is the process by which plants convert light energy into chemical energy..."
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
                    model_answer=model_answer,
                    key_points=key_points,
                    hints=hints
                ))
    
    if st.button(t("add_criterion")):
        st.session_state.num_criteria += 1
        st.rerun()
    
    # Navigation
    if st.button(t("next"), type="primary"):
        if not question or len(rubric) == 0:
            st.error("Please fill in the question and at least one criterion.")
        else:
            # Check that all criteria have model answers
            missing_answers = [c.name for c in rubric if not c.model_answer.strip()]
            if missing_answers:
                st.error(f"Please provide model answers for: {', '.join(missing_answers)}")
            else:
                st.session_state.assessment_config = AssessmentConfig(
                    question=question,
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
    
    # Get API key for OCR
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Collect answers for each criterion
    criterion_submissions = []
    
    for i, criterion in enumerate(config.rubric):
        st.subheader(f"📝 {criterion.name} ({criterion.marks} marks)")
        
        # Optional: Document upload for this criterion
        if api_key:
            uploaded_file = st.file_uploader(
                f"Upload answer image for {criterion.name} (optional)",
                type=["pdf", "png", "jpg", "jpeg"],
                key=f"student_upload_{i}"
            )
            
            if uploaded_file:
                file_key = f"{uploaded_file.name}_{uploaded_file.size}_{i}"
                
                if st.session_state.get(f"last_file_{i}") != file_key:
                    with st.spinner(t("processing")):
                        try:
                            extracted_text = DocumentProcessor.process_uploaded_file(
                                uploaded_file, 
                                st.session_state.language,
                                api_key
                            )
                            
                            if extracted_text:
                                st.session_state[f"extracted_answer_{i}"] = extracted_text
                                st.session_state[f"last_file_{i}"] = file_key
                                st.success("✓ Text extracted!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Answer text area
        answer = st.text_area(
            f"{t('answer_for_criterion')} {criterion.name}",
            value=st.session_state.get(f"extracted_answer_{i}", ""),
            height=150,
            placeholder=f"Type your answer for {criterion.name}...",
            key=f"student_answer_{i}"
        )
        
        if answer.strip():
            criterion_submissions.append(CriterionSubmission(
                criterion_name=criterion.name,
                answer=answer
            ))
        
        st.divider()
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(t("back")):
            st.session_state.page = 1
            st.rerun()
    
    with col2:
        if st.button(t("submit_answer"), type="primary"):
            if len(criterion_submissions) != len(config.rubric):
                st.error("Please provide answers for all criteria.")
            else:
                submission = StudentSubmission(
                    criterion_answers=criterion_submissions,
                    language=st.session_state.language
                )
                
                # Grade the submission
                with st.spinner("Grading your answers..."):
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
            st.write("**Justification:**")
            st.write(result.justification)
            
            # Show student's answer for this criterion
            student_answer = next(
                (ca.answer for ca in submission.criterion_answers if ca.criterion_name == result.criterion_name),
                ""
            )
            
            with st.expander("View your answer"):
                st.write(student_answer)
    
    # Feedback
    st.subheader(t("feedback"))
    for i, fb in enumerate(report.feedback, 1):
        st.write(f"{i}. {fb}")
    
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

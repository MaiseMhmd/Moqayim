import os
import re
import math
import io
import base64
import secrets
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
from collections import Counter
from datetime import datetime, timedelta

from flask import Flask, request, render_template_string, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

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

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# LTI Configuration
LTI_CONSUMER_KEY = os.environ.get('LTI_CONSUMER_KEY', 'moqayim_key')
LTI_SHARED_SECRET = os.environ.get('LTI_SHARED_SECRET', 'moqayim_secret')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def extract_text(self, image: Image.Image, language: str = "en") -> str:
        """Extract text from image using Groq Llama Maverick"""
        try:
            width, height = image.size
            total_pixels = width * height
            if total_pixels > 33177600:
                scale = math.sqrt(33177600 / total_pixels)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            image_url = self.image_to_base64_url(image)
            
            base64_size = len(image_url.encode('utf-8'))
            if base64_size > 4 * 1024 * 1024:
                raise Exception(f"Image too large after encoding: {base64_size} bytes")
            
            if language == "ar":
                prompt = "استخرج كل النص الموجود في هذه الصورة بدقة عالية جداً. يشمل ذلك النص المكتوب بخط اليد والنص المطبوع والكتابة اليدوية غير الواضحة. اقرأ بعناية حرفاً بحرف وكلمة بكلمة. إذا كان النص مكتوباً بخط اليد، حاول التعرف على الحروف والكلمات حتى لو كانت مشوهة أو غير منتظمة. أعد النص المستخرج فقط بدون أي تعليقات أو تنسيق إضافي أو شرح."
            else:
                prompt = "Extract all text from this image with very high accuracy. This includes handwritten text, printed text, and unclear handwriting. Read carefully letter by letter and word by word. If the text is handwritten, try to recognize the letters and words even if they are distorted or irregular. Return only the extracted text without any comments, additional formatting, or explanations."
            
            completion = self.client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                temperature=0.1,
                max_completion_tokens=2048,
                top_p=1,
                stream=False,
                stop=None
            )
            
            extracted_text = completion.choices[0].message.content
            if not extracted_text or extracted_text.strip() == "":
                raise Exception("No text was extracted from the image.")
            return extracted_text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise Exception("API rate limit exceeded. Please try again later.")
            elif "size" in error_msg.lower() or "large" in error_msg.lower():
                raise Exception("Image file is too large. Please use a smaller image.")
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
            raise Exception("Groq library is not installed.")
        
        try:
            ocr = GroqOCR(api_key=api_key)
            return ocr.extract_text(image, language)
        except Exception as e:
            raise Exception(f"Image OCR failed: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf_text(pdf_file) -> str:
        """Extract text from PDF using PyPDF2"""
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
        except Exception:
            return ""
    
    @staticmethod
    def extract_text_from_pdf_ocr(pdf_file, language: str = "en", api_key: Optional[str] = None) -> str:
        """Extract text from PDF using Groq OCR"""
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("pdf2image is not installed.")
        
        if not GROQ_AVAILABLE:
            raise Exception("Groq library is not installed.")
        
        try:
            pdf_file.seek(0)
            images = pdf2image.convert_from_bytes(pdf_file.read(), dpi=300)
            
            ocr = GroqOCR(api_key=api_key)
            
            full_text = []
            for i, image in enumerate(images):
                page_text = ocr.extract_text(image, language)
                if page_text:
                    full_text.append(f"--- Page {i+1} ---\n{page_text}")
            
            return "\n\n".join(full_text)
        except Exception as e:
            raise Exception(f"PDF OCR failed: {str(e)}")
    
    @staticmethod
    def process_uploaded_file(file_storage, language: str = "en", api_key: Optional[str] = None) -> str:
        """Process uploaded file (PDF or image) using Groq OCR"""
        if file_storage is None:
            return ""
        
        file_type = file_storage.content_type
        
        try:
            if file_type == "application/pdf":
                text = DocumentProcessor.extract_text_from_pdf_text(file_storage.stream)
                
                if not text or len(text.strip()) < 50:
                    text = DocumentProcessor.extract_text_from_pdf_ocr(file_storage.stream, language, api_key)
                
                return text
                
            elif file_type.startswith("image/"):
                image = Image.open(file_storage.stream)
                return DocumentProcessor.extract_text_from_image(image, language, api_key)
            else:
                raise Exception("Unsupported file type. Please upload PDF or image files.")
        except Exception as e:
            raise Exception(f"File processing error: {str(e)}")
    
    @staticmethod
    def smart_split_qa(extracted_text: str) -> tuple:
        """Intelligently split extracted text into question and answer"""
        text = extracted_text.strip()
        
        text = re.sub(r'---\s*Page\s+\d+\s*---', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        
        patterns = [
            (r'(?i)question[:\s]+(.*?)(?:answer[:\s]+|model\s+answer[:\s]+|solution[:\s]+)(.*)', 1, 2),
            (r'(?i)q[:\.\s]+(.*?)(?:a[:\.\s]+|ans[:\.\s]+)(.*)', 1, 2),
            (r'(?i)(.*?)(?:answer[:\s]+|solution[:\s]+|model\s+answer[:\s]+)(.*)', 1, 2),
        ]
        
        for pattern, q_group, a_group in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                question = match.group(q_group).strip()
                answer = match.group(a_group).strip()
                
                if len(question) > 10 and len(answer) > 10:
                    return question, answer
        
        lines = text.split('\n')
        if len(lines) > 3:
            mid = len(lines) // 2
            question = '\n'.join(lines[:mid]).strip()
            answer = '\n'.join(lines[mid:]).strip()
        else:
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
    def check_key_points(student_text: str, key_points: List[str]) -> tuple:
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
    def evaluate_criterion(student_answer: str, model_answer: str, criterion: RubricCriterion) -> CriterionResult:
        """Evaluate a single criterion"""
        
        student_tokens = GradingEngine.tokenize(student_answer)
        model_tokens = GradingEngine.tokenize(model_answer)
        
        overlap_score = GradingEngine.calculate_overlap(student_tokens, model_tokens)
        cosine_score = GradingEngine.calculate_cosine_similarity(student_tokens, model_tokens)
        
        found_count, total_count, found_points = GradingEngine.check_key_points(
            student_answer, criterion.key_points
        )
        
        if total_count > 0:
            key_point_score = found_count / total_count
        else:
            key_point_score = (overlap_score + cosine_score) / 2
        
        if key_point_score >= 0.8:
            status = "met"
            marks = criterion.marks
        elif key_point_score >= 0.4:
            status = "partial"
            marks = max(1, criterion.marks // 2) if criterion.marks > 1 else 0
        else:
            status = "not_met"
            marks = 0
        
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
    def generate_justification(criterion: RubricCriterion, found_count: int, total_count: int, 
                              found_points: List[str], score: float, status: str) -> str:
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
        
        else:
            if total_count > 0:
                return f"✗ Not satisfied: {found_count}/{total_count} key points identified. Missing critical elements."
            return f"✗ Not satisfied: insufficient alignment with expected answer."
    
    @staticmethod
    def generate_feedback(criterion_results: List[CriterionResult], rubric: List[RubricCriterion]) -> List[str]:
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
    def grade_submission(config: AssessmentConfig, submission: StudentSubmission) -> GradingReport:
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
# TRANSLATIONS
# ============================================================================

TRANSLATIONS = {
    "en": {
        "app_title": "Moqayim - Short Answer Grading",
        "page1_title": "Teacher Setup",
        "page2_title": "Student Answer",
        "page3_title": "Grading Results",
        "question_label": "Question",
        "model_answer_label": "Model Answer",
        "rubric_label": "Rubric Criteria",
        "criterion_name": "Criterion Name",
        "criterion_marks": "Marks",
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
        "upload_hint": "Upload a PDF or image containing the question and model answer"
    },
    "ar": {
        "app_title": "مُقَيِّم - تصحيح الإجابات القصيرة",
        "page1_title": "إعداد المعلم",
        "page2_title": "إجابة الطالب",
        "page3_title": "نتائج التصحيح",
        "question_label": "السؤال",
        "model_answer_label": "الإجابة النموذجية",
        "rubric_label": "معايير التقييم",
        "criterion_name": "اسم المعيار",
        "criterion_marks": "الدرجات",
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
        "upload_hint": "قم بتحميل PDF أو صورة تحتوي على السؤال والإجابة النموذجية"
    }
}

def t(key: str, lang: str = "en") -> str:
    """Get translation"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)

# ============================================================================
# LTI VALIDATION
# ============================================================================

def validate_lti_request(request_data):
    """Validate LTI 1.1 launch request"""
    try:
        # Simple validation - check for required parameters
        required_params = ['oauth_consumer_key', 'user_id', 'roles']
        
        for param in required_params:
            if param not in request_data:
                return False, f"Missing required parameter: {param}"
        
        # Validate consumer key
        if request_data.get('oauth_consumer_key') != LTI_CONSUMER_KEY:
            return False, "Invalid consumer key"
        
        # For prototype, skip full OAuth signature validation
        # In production, you'd validate oauth_signature using HMAC-SHA1
        
        return True, "Valid"
    
    except Exception as e:
        return False, str(e)

# ============================================================================
# HTML TEMPLATES
# ============================================================================

def render_page(content_template, lang, **kwargs):
    """Render a page with the base template"""
    from flask import render_template_string
    
    # First render the content template with all variables
    content_html = render_template_string(content_template, lang=lang, t=t, **kwargs)
    
    # Then render the base template with the rendered content
    base = """
<!DOCTYPE html>
<html lang="{{ 'ar' if lang == 'ar' else 'en' }}" dir="{{ 'rtl' if lang == 'ar' else 'ltr' }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ t('app_title', lang) }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 700;
        }
        
        .header-controls {
            display: flex;
            gap: 10px;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }
        
        input[type="text"],
        input[type="number"],
        textarea,
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 15px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="number"]:focus,
        textarea:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 120px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #e0e0e0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #d0d0d0;
        }
        
        .btn-success {
            background: #10b981;
            color: white;
        }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        
        .criterion-card {
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }
        
        .criterion-header {
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .alert-info {
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            color: #1e40af;
        }
        
        .alert-success {
            background: #d1fae5;
            border-left: 4px solid #10b981;
            color: #065f46;
        }
        
        .alert-error {
            background: #fee2e2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
        }
        
        .alert-warning {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            color: #92400e;
        }
        
        .file-upload {
            border: 2px dashed #d1d5db;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            background: #f9fafb;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .file-upload:hover {
            border-color: #667eea;
            background: #f3f4f6;
        }
        
        .file-upload input[type="file"] {
            display: none;
        }
        
        .score-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .score-card h2 {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        .score-card p {
            font-size: 18px;
            opacity: 0.9;
        }
        
        .result-card {
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }
        
        .result-card.met {
            border-left: 4px solid #10b981;
            background: #f0fdf4;
        }
        
        .result-card.partial {
            border-left: 4px solid #f59e0b;
            background: #fffbeb;
        }
        
        .result-card.not_met {
            border-left: 4px solid #ef4444;
            background: #fef2f2;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .result-title {
            font-weight: 700;
            font-size: 16px;
        }
        
        .result-score {
            font-weight: 700;
            font-size: 18px;
        }
        
        .feedback-list {
            list-style: none;
            padding: 0;
        }
        
        .feedback-list li {
            padding: 12px;
            background: #f9fafb;
            border-left: 3px solid #667eea;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        
        .divider {
            height: 2px;
            background: #e5e7eb;
            margin: 30px 0;
        }
        
        .lti-banner {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .header-controls {
                margin-top: 15px;
                width: 100%;
            }
            
            .criterion-header {
                grid-template-columns: 1fr;
            }
            
            .btn-group {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ t('app_title', lang) }}</h1>
            <div class="header-controls">
                <form method="GET" style="display: inline;">
                    <select name="lang" onchange="this.form.submit()" class="btn btn-secondary" style="padding: 8px 16px;">
                        <option value="en" {{ 'selected' if lang == 'en' else '' }}>English</option>
                        <option value="ar" {{ 'selected' if lang == 'ar' else '' }}>العربية</option>
                    </select>
                </form>
                <a href="{{ url_for('reset') }}" class="btn btn-secondary">{{ t('reset', lang) }}</a>
            </div>
        </div>
        
        <div class="content">
            {% if lti_user %}
            <div class="lti-banner">
                🔗 Connected via LTI | Role: {{ lti_role }} | User: {{ lti_user }}
            </div>
            {% endif %}
            
            {{ content | safe }}
        </div>
    </div>
</body>
</html>
"""
    
    return render_template_string(base, content=content_html, lang=lang, t=t, **kwargs)

PAGE1_CONTENT = """
<h2 style="margin-bottom: 25px;">{{ t('page1_title', lang) }}</h2>

<form method="POST" enctype="multipart/form-data">
    
    {% if not groq_available %}
    <div class="alert alert-warning">
        ⚠️ Groq library is not installed. OCR features will be disabled.
    </div>
    {% elif not groq_api_key %}
    <div class="alert alert-warning">
        ⚠️ Please set your GROQ_API_KEY environment variable to use OCR features.
    </div>
    {% endif %}
    
    <div class="divider"></div>
    
    <h3 style="margin-bottom: 15px;">📄 {{ t('upload_document', lang) }}</h3>
    <p style="color: #666; margin-bottom: 15px; font-size: 14px;">{{ t('upload_hint', lang) }}</p>
    
    <div class="form-group">
        <label class="file-upload">
            <input type="file" name="document" accept=".pdf,.png,.jpg,.jpeg" onchange="this.form.submit()">
            <div style="font-size: 48px; margin-bottom: 10px;">📤</div>
            <div style="font-weight: 600; color: #667eea;">Click to upload PDF or Image</div>
            <div style="font-size: 13px; color: #666; margin-top: 5px;">Supports PDF, PNG, JPG, JPEG</div>
        </label>
    </div>
    
    {% if extracted_text %}
    <div class="alert alert-success">
        ✓ Text extracted and fields auto-filled!
    </div>
    <details style="margin-bottom: 20px;">
        <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: #f9fafb; border-radius: 4px;">
            📝 {{ t('extracted_text', lang) }}
        </summary>
        <textarea readonly style="margin-top: 10px; background: #f9fafb; font-family: monospace; font-size: 13px;">{{ extracted_text }}</textarea>
    </details>
    {% endif %}
    
    <div class="divider"></div>
    
    <div class="form-group">
        <label>{{ t('question_label', lang) }}</label>
        <textarea name="question" placeholder="e.g., Explain the process of photosynthesis." required>{{ question or '' }}</textarea>
    </div>
    
    <div class="form-group">
        <label>{{ t('model_answer_label', lang) }}</label>
        <textarea name="model_answer" style="min-height: 150px;" placeholder="e.g., Photosynthesis is the process by which plants convert light energy into chemical energy..." required>{{ model_answer or '' }}</textarea>
    </div>
    
    <div class="divider"></div>
    
    <h3 style="margin-bottom: 20px;">{{ t('rubric_label', lang) }}</h3>
    
    {% for i in range(num_criteria) %}
    <div class="criterion-card">
        <div class="criterion-header">
            <div class="form-group" style="margin-bottom: 0;">
                <label>{{ t('criterion_name', lang) }} {{ i + 1 }}</label>
                <input type="text" name="criterion_name_{{ i }}" placeholder="e.g., Definition" required>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
                <label>{{ t('criterion_marks', lang) }}</label>
                <input type="number" name="criterion_marks_{{ i }}" min="1" max="10" value="2" required>
            </div>
        </div>
        <div class="form-group" style="margin-bottom: 10px;">
            <label>{{ t('key_points', lang) }}</label>
            <input type="text" name="criterion_kp_{{ i }}" placeholder="e.g., light energy, chemical energy, glucose">
        </div>
        <div class="form-group" style="margin-bottom: 0;">
            <label>{{ t('hints', lang) }}</label>
            <input type="text" name="criterion_hints_{{ i }}" placeholder="e.g., Include the chemical equation">
        </div>
    </div>
    {% endfor %}
    
    <button type="submit" name="add_criterion" value="1" class="btn btn-secondary">{{ t('add_criterion', lang) }}</button>
    
    <div class="btn-group">
        <button type="submit" class="btn btn-primary">{{ t('next', lang) }}</button>
    </div>
</form>
"""

PAGE2_CONTENT = """
<h2 style="margin-bottom: 25px;">{{ t('page2_title', lang) }}</h2>

<div class="alert alert-info">
    <strong>{{ t('question_label', lang) }}</strong><br>
    {{ question }}
</div>

<div class="divider"></div>

<form method="POST" enctype="multipart/form-data">
    
    <h3 style="margin-bottom: 15px;">📄 {{ t('upload_document', lang) }}</h3>
    <p style="color: #666; margin-bottom: 15px; font-size: 14px;">Upload a scanned document or image of the student's handwritten answer</p>
    
    <div class="form-group">
        <label class="file-upload">
            <input type="file" name="student_document" accept=".pdf,.png,.jpg,.jpeg" onchange="this.form.submit()">
            <div style="font-size: 48px; margin-bottom: 10px;">📤</div>
            <div style="font-weight: 600; color: #667eea;">Click to upload PDF or Image</div>
            <div style="font-size: 13px; color: #666; margin-top: 5px;">Supports PDF, PNG, JPG, JPEG</div>
        </label>
    </div>
    
    {% if student_extracted_text %}
    <div class="alert alert-success">
        ✓ Student answer extracted and auto-filled!
    </div>
    <details style="margin-bottom: 20px;">
        <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: #f9fafb; border-radius: 4px;">
            📝 {{ t('extracted_text', lang) }}
        </summary>
        <textarea readonly style="margin-top: 10px; background: #f9fafb; font-family: monospace; font-size: 13px;">{{ student_extracted_text }}</textarea>
    </details>
    {% endif %}
    
    <div class="divider"></div>
    
    <div class="form-group">
        <label>{{ t('student_answer_label', lang) }}</label>
        <textarea name="student_answer" style="min-height: 200px;" placeholder="Type your answer here..." required>{{ student_answer or '' }}</textarea>
    </div>
    
    <div class="btn-group">
        <a href="{{ url_for('page1') }}" class="btn btn-secondary">{{ t('back', lang) }}</a>
        <button type="submit" class="btn btn-primary">{{ t('submit_answer', lang) }}</button>
    </div>
</form>
"""

PAGE3_CONTENT = """
<h2 style="margin-bottom: 25px;">{{ t('page3_title', lang) }}</h2>

<div class="score-card">
    <h2>{{ total_score }}/{{ max_score }}</h2>
    <p>{{ t('total_score', lang) }} • {{ '%.1f'|format(percentage) }}%</p>
</div>

<h3 style="margin-bottom: 15px;">Criterion Breakdown</h3>

{% for result in criterion_results %}
<div class="result-card {{ result.status }}">
    <div class="result-header">
        <div class="result-title">
            {% if result.status == 'met' %}🟢
            {% elif result.status == 'partial' %}🟡
            {% else %}🔴{% endif %}
            {{ result.criterion_name }}
        </div>
        <div class="result-score">{{ result.marks_awarded }}/{{ result.marks_total }}</div>
    </div>
    <div style="color: #666; font-size: 14px;">{{ result.justification }}</div>
</div>
{% endfor %}

<div class="divider"></div>

<h3 style="margin-bottom: 15px;">{{ t('feedback', lang) }}</h3>
<ul class="feedback-list">
    {% for fb in feedback %}
    <li>{{ loop.index }}. {{ fb }}</li>
    {% endfor %}
</ul>

<details style="margin-top: 30px;">
    <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: #f9fafb; border-radius: 4px;">
        View Student Answer
    </summary>
    <div style="margin-top: 15px; padding: 15px; background: #f9fafb; border-radius: 4px; white-space: pre-wrap;">{{ student_answer }}</div>
</details>

<div class="btn-group">
    <a href="{{ url_for('page2') }}" class="btn btn-secondary">{{ t('back', lang) }}</a>
    <a href="{{ url_for('reset') }}" class="btn btn-danger">{{ t('reset', lang) }}</a>
</div>
"""

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/lti/launch', methods=['POST'])
def lti_launch():
    """Handle LTI launch from Blackboard"""
    
    # Validate LTI request
    is_valid, message = validate_lti_request(request.form)
    
    if not is_valid:
        return f"LTI Launch Failed: {message}", 403
    
    # Store LTI session data
    session.permanent = True
    session['lti_valid'] = True
    session['lti_user_id'] = request.form.get('user_id')
    session['lti_user_name'] = request.form.get('lis_person_name_full', 'User')
    session['lti_role'] = 'teacher' if 'Instructor' in request.form.get('roles', '') else 'student'
    session['lti_course_id'] = request.form.get('context_id', 'unknown')
    session['lti_course_name'] = request.form.get('context_title', 'Unknown Course')
    
    # Redirect to appropriate page based on role
    return redirect(url_for('page1'))

@app.route('/')
def index():
    """Redirect to page 1"""
    return redirect(url_for('page1'))

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    """Teacher setup page"""
    lang = request.args.get('lang', session.get('language', 'en'))
    session['language'] = lang
    
    if request.method == 'POST':
        # Handle file upload
        if 'document' in request.files and request.files['document'].filename:
            file = request.files['document']
            try:
                extracted_text = DocumentProcessor.process_uploaded_file(file, lang, GROQ_API_KEY)
                question, answer = DocumentProcessor.smart_split_qa(extracted_text)
                session['extracted_question'] = question
                session['extracted_answer'] = answer
                session['extracted_text'] = extracted_text
                return redirect(url_for('page1', lang=lang))
            except Exception as e:
                return render_page(
                    PAGE1_CONTENT,
                    lang,
                    num_criteria=session.get('num_criteria', 3),
                    error=str(e),
                    groq_available=GROQ_AVAILABLE,
                    groq_api_key=GROQ_API_KEY,
                    lti_user=session.get('lti_user_name'),
                    lti_role=session.get('lti_role'),
                    url_for=url_for
                )
        
        # Handle add criterion
        if request.form.get('add_criterion'):
            session['num_criteria'] = session.get('num_criteria', 3) + 1
            return redirect(url_for('page1', lang=lang))
        
        # Handle form submission
        question = request.form.get('question')
        model_answer = request.form.get('model_answer')
        num_criteria = session.get('num_criteria', 3)
        
        rubric = []
        for i in range(num_criteria):
            name = request.form.get(f'criterion_name_{i}')
            if name:
                marks = int(request.form.get(f'criterion_marks_{i}', 2))
                key_points_str = request.form.get(f'criterion_kp_{i}', '')
                hints_str = request.form.get(f'criterion_hints_{i}', '')
                
                key_points = [kp.strip() for kp in key_points_str.split(",") if kp.strip()]
                hints = [h.strip() for h in hints_str.split(",") if h.strip()]
                
                rubric.append({
                    'name': name,
                    'marks': marks,
                    'key_points': key_points,
                    'hints': hints
                })
        
        if not question or not model_answer or len(rubric) == 0:
            return render_page(
                PAGE1_CONTENT,
                lang,
                num_criteria=num_criteria,
                error="Please fill in all required fields.",
                groq_available=GROQ_AVAILABLE,
                groq_api_key=GROQ_API_KEY,
                lti_user=session.get('lti_user_name'),
                lti_role=session.get('lti_role'),
                url_for=url_for
            )
        
        session['assessment_config'] = {
            'question': question,
            'model_answer': model_answer,
            'rubric': rubric,
            'language': lang
        }
        
        return redirect(url_for('page2', lang=lang))
    
    # GET request
    return render_page(
        PAGE1_CONTENT,
        lang,
        num_criteria=session.get('num_criteria', 3),
        question=session.get('extracted_question', ''),
        model_answer=session.get('extracted_answer', ''),
        extracted_text=session.get('extracted_text', ''),
        groq_available=GROQ_AVAILABLE,
        groq_api_key=GROQ_API_KEY,
        lti_user=session.get('lti_user_name'),
        lti_role=session.get('lti_role'),
        url_for=url_for
    )

@app.route('/page2', methods=['GET', 'POST'])
def page2():
    """Student answer page"""
    lang = request.args.get('lang', session.get('language', 'en'))
    
    if 'assessment_config' not in session:
        return redirect(url_for('page1', lang=lang))
    
    config = session['assessment_config']
    
    if request.method == 'POST':
        # Handle file upload
        if 'student_document' in request.files and request.files['student_document'].filename:
            file = request.files['student_document']
            try:
                extracted_text = DocumentProcessor.process_uploaded_file(file, lang, GROQ_API_KEY)
                session['student_extracted_answer'] = extracted_text
                session['student_extracted_text'] = extracted_text
                return redirect(url_for('page2', lang=lang))
            except Exception as e:
                return render_page(
                    PAGE2_CONTENT,
                    lang,
                    question=config['question'],
                    error=str(e),
                    lti_user=session.get('lti_user_name'),
                    lti_role=session.get('lti_role'),
                    url_for=url_for
                )
        
        # Handle answer submission
        student_answer = request.form.get('student_answer')
        
        if not student_answer or not student_answer.strip():
            return render_page(
                PAGE2_CONTENT,
                lang,
                question=config['question'],
                error="Please provide an answer.",
                lti_user=session.get('lti_user_name'),
                lti_role=session.get('lti_role'),
                url_for=url_for
            )
        
        # Create assessment objects
        rubric = [
            RubricCriterion(
                name=c['name'],
                marks=c['marks'],
                key_points=c['key_points'],
                hints=c['hints']
            ) for c in config['rubric']
        ]
        
        assessment = AssessmentConfig(
            question=config['question'],
            model_answer=config['model_answer'],
            rubric=rubric,
            language=config['language']
        )
        
        submission = StudentSubmission(
            answer=student_answer,
            language=lang
        )
        
        # Grade the submission
        report = GradingEngine.grade_submission(assessment, submission)
        
        # Store results in session
        session['grading_report'] = {
            'total_score': report.total_score,
            'max_score': report.max_score,
            'percentage': (report.total_score / report.max_score * 100) if report.max_score > 0 else 0,
            'criterion_results': [
                {
                    'criterion_name': r.criterion_name,
                    'status': r.status,
                    'marks_awarded': r.marks_awarded,
                    'marks_total': r.marks_total,
                    'justification': r.justification
                } for r in report.criterion_results
            ],
            'feedback': report.feedback
        }
        session['student_answer'] = student_answer
        
        return redirect(url_for('page3', lang=lang))
    
    # GET request
    return render_page(
        PAGE2_CONTENT,
        lang,
        question=config['question'],
        student_answer=session.get('student_extracted_answer', ''),
        student_extracted_text=session.get('student_extracted_text', ''),
        lti_user=session.get('lti_user_name'),
        lti_role=session.get('lti_role'),
        url_for=url_for
    )

@app.route('/page3')
def page3():
    """Results page"""
    lang = request.args.get('lang', session.get('language', 'en'))
    
    if 'grading_report' not in session:
        return redirect(url_for('page1', lang=lang))
    
    report = session['grading_report']
    
    return render_page(
        PAGE3_CONTENT,
        lang,
        total_score=report['total_score'],
        max_score=report['max_score'],
        percentage=report['percentage'],
        criterion_results=report['criterion_results'],
        feedback=report['feedback'],
        student_answer=session.get('student_answer', ''),
        lti_user=session.get('lti_user_name'),
        lti_role=session.get('lti_role'),
        url_for=url_for
    )

@app.route('/reset')
def reset():
    """Reset session"""
    lang = session.get('language', 'en')
    session.clear()
    session['language'] = lang
    return redirect(url_for('page1', lang=lang))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

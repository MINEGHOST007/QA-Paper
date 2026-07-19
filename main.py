import streamlit as st
import requests
from PIL import Image
import io
import re
import json
import base64
import html as html_mod

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Question Paper Formatter",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Simple CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container { padding-top: 1rem; max-width: 900px; }
.stTextArea textarea { font-family: 'Consolas', monospace; font-size: 13px; }
.step-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 10px 18px; border-radius: 10px;
    margin: 20px 0 10px 0; font-size: 18px; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
_defaults = {
    "raw_text": "",
    "structured_json": "",
    "final_html": "",
    "school_name": "",
    "exam_name": "",
    "class_name": "",
    "marks": "",
    "subject": "",
    "time_allowed": "",
    "general_instructions": "",
    "show_instructions": False,
    "ocr_done": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
# OpenRouter models (free, vision-capable)
# ──────────────────────────────────────────────────────────────
FREE_VISION_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "openrouter/free",
]


# ──────────────────────────────────────────────────────────────
# API helper
# ──────────────────────────────────────────────────────────────
def call_openrouter(messages: list, api_key: str, models: list) -> str:
    """Try each model in order. Return the assistant content on first success."""
    errors = []
    for model in models:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://qpaper.streamlit.app",
                    "X-Title": "Question Paper Formatter",
                },
                data=json.dumps({"model": model, "messages": messages}),
                timeout=180,
            )
            result = resp.json()
            if "error" in result:
                e = result["error"]
                errors.append(f"{model}: {e.get('message', e) if isinstance(e, dict) else e}")
                continue
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content.strip():
                return content
            errors.append(f"{model}: Empty response")
        except requests.exceptions.Timeout:
            errors.append(f"{model}: Timeout")
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    raise RuntimeError("All models failed:\n" + "\n".join(f"  • {e}" for e in errors))


def images_to_base64_parts(images: list[Image.Image]) -> list[dict]:
    """Convert PIL images to OpenAI-compatible base64 content parts."""
    parts = []
    for img in images:
        work = img.copy()
        if max(work.size) > 2048:
            work.thumbnail((2048, 2048), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = "PNG" if work.mode == "RGBA" else "JPEG"
        work.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {}))
        b64 = base64.b64encode(buf.getvalue()).decode()
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


# ──────────────────────────────────────────────────────────────
# PASS 1 – Extract structured JSON from images (improved prompt)
# ──────────────────────────────────────────────────────────────
EXTRACT_PROMPT = r"""You are an expert OCR system specialized in reading Indian school question papers.

CRITICAL CONTEXT — WHAT IS AN INDIAN SCHOOL QUESTION PAPER:
These are exam papers used in Indian schools (CBSE, ICSE, State boards, or private schools).
A typical question paper has this structure:

1. **HEADER** at the top of the page:
   - School name (e.g., "ABHYUDAYA HIGH SCHOOL") — often in ALL CAPS with NO SPACES between words
   - Exam name (e.g., "FORMATIVE ASSESSMENT - 3", "SUMMATIVE ASSESSMENT", "UNIT TEST", "HALF YEARLY EXAM", "ANNUAL EXAM")
   - Subject (e.g., "Social Studies", "Mathematics", "Science")
   - Class (e.g., "VII", "IX", "X")
   - Max Marks (e.g., "20", "80", "100") — sometimes written as "20M" or "Max. Marks: 20"
   - Time Allowed (e.g., "1 Hour", "2½ Hours", "3 Hours")

2. **GENERAL INSTRUCTIONS** (optional):
   - E.g., "Answer all questions", "Write neatly", "All questions are compulsory"

3. **SECTIONS** — The paper is divided into sections using Roman numerals (I, II, III, IV) or letters (A, B, C, D).
   Each section has:
   - A **title/heading** like "I. ANSWER THE FOLLOWING QUESTIONS" or "SECTION A"
   - A **marks breakdown** like "(4×2M=8M)" meaning 4 questions × 2 marks each = 8 marks total
   - **Questions** numbered 1, 2, 3, 4... (sometimes with asterisks like 1*, 2*)

4. **QUESTION TYPES** you will encounter:
   - **Short answer**: Simple questions worth 1-2 marks
   - **Long answer**: Questions worth 4-5+ marks, sometimes with a reading passage
   - **MCQ (Multiple Choice)**: Questions with options a, b, c, d — often the answer is to be written in a bracket [ ]
   - **Fill in the blanks**: Sentences with blanks shown as __________ 
   - **True/False**: Statements to mark as true or false
   - **Match the columns**: Two columns to be matched
   - **Map pointing**: "Draw an outline map of India and locate the given places: a) Hampi, b) River Tungabhadra, c) Kanchi, d) Bijapur"

COMMON OCR CHALLENGES — YOU MUST HANDLE THESE:
- Words are JOINED together without spaces: "WhatisSulh-i-kul?" → "What is Sulh-i-kul?"
- Headers run together: "ANSWERTHEFOLLOWINGQUESTIONS" → "ANSWER THE FOLLOWING QUESTIONS"
- School names run together: "ABHYUDAYAHIGHSCHOOL" → "ABHYUDAYA HIGH SCHOOL"
- Marks notation: "4x2M=8M" or "4×2M=8M" or "(4x½M=2M)"
- Asterisks (*) used instead of periods for numbering: "1*" means question 1
- Text may be blurry, tilted, or have mixed fonts

CRITICAL RULES:
- The SAME question paper may appear TWICE in the images (printed twice on one page for cutting). Extract it only ONCE.
- Images may be in ANY ORDER — figure out the correct sequence from section numbers and question numbers.
- Extract EVERY section and EVERY question. Do NOT skip any.
- Separate ALL joined/merged words into proper readable English.
- For MCQ questions, extract ALL options (a, b, c, d).
- For map pointing, extract ALL places to locate.
- For passages/paragraphs, extract the FULL passage text.
- Extract marks as just the number (e.g., "2" not "2M"), but keep section marks info like "4x2M=8M" in the section title.

OUTPUT: A single JSON object. No markdown fences, no commentary, no explanation — ONLY valid JSON.

{
  "school_name": "ABHYUDAYA HIGH SCHOOL",
  "exam_name": "FORMATIVE ASSESSMENT - 3",
  "class": "VII",
  "subject": "Social Studies",
  "max_marks": "20",
  "time_allowed": "",
  "general_instructions": ["Answer all questions"],
  "sections": [
    {
      "title": "I. ANSWER THE FOLLOWING QUESTIONS (4x2M=8M)",
      "section_marks": "4x2M=8M",
      "questions": [
        {
          "number": "1",
          "text": "What is Sulh-i-kul?",
          "marks": "2",
          "type": "short_answer",
          "sub_parts": [],
          "options": []
        },
        {
          "number": "2",
          "text": "What kinds of arms and weapons do modern armies use?",
          "marks": "2",
          "type": "short_answer",
          "sub_parts": [],
          "options": []
        }
      ]
    },
    {
      "title": "II. ANSWER THE FOLLOWING (2x4M=8M)",
      "section_marks": "2x4M=8M",
      "questions": [
        {
          "number": "1",
          "text": "What was the relationship between the Mansabdar and Jagir?",
          "marks": "4",
          "type": "long_answer",
          "sub_parts": [],
          "options": []
        },
        {
          "number": "2",
          "text": "Read the paragraph under the title 'zabt and zamindars' and comment on it.",
          "marks": "4",
          "type": "long_answer",
          "passage": "The main source of income available to Mughal rulers was taxes on the produce of the peasantry...",
          "sub_parts": [],
          "options": []
        }
      ]
    },
    {
      "title": "III. MAP POINTING (4x½M=2M)",
      "section_marks": "4x½M=2M",
      "questions": [
        {
          "number": "1",
          "text": "Draw an outline map of India and locate the given places:",
          "marks": "2",
          "type": "map_pointing",
          "sub_parts": [
            {"label": "a", "text": "Hampi"},
            {"label": "b", "text": "River Tungabhadra"},
            {"label": "c", "text": "Kanchi"},
            {"label": "d", "text": "Bijapur"}
          ],
          "options": []
        }
      ]
    },
    {
      "title": "IV. OBJECTIVES - CHOOSE THE CORRECT OPTION (4x½M=2M)",
      "section_marks": "4x½M=2M",
      "questions": [
        {
          "number": "1",
          "text": "Vijayanagara means the city of __________.",
          "marks": "0.5",
          "type": "mcq",
          "sub_parts": [],
          "options": [
            {"label": "a", "text": "Victory"},
            {"label": "b", "text": "Viceroy"},
            {"label": "c", "text": "Valour"},
            {"label": "d", "text": "Vedas"}
          ]
        }
      ]
    }
  ]
}

Remember: Output ONLY the JSON. No text before or after. No markdown fences. Every section and every question must be included."""


def pass1_extract(images: list[Image.Image], api_key: str) -> dict:
    """Extract structured JSON from question paper images."""
    img_parts = images_to_base64_parts(images)
    content = [{"type": "text", "text": EXTRACT_PROMPT}] + img_parts
    messages = [{"role": "user", "content": content}]
    raw = call_openrouter(messages, api_key, FREE_VISION_MODELS)

    # Store raw text for debugging
    st.session_state.raw_text = raw

    # Clean up: strip markdown fences, thinking tags, etc.
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Find the JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]

    return json.loads(raw)


# ──────────────────────────────────────────────────────────────
# HTML Builder — Python generates HTML directly (NO second LLM call)
# ──────────────────────────────────────────────────────────────
def _esc(text) -> str:
    """HTML-escape text."""
    return html_mod.escape(str(text)) if text else ""


def build_question_html(q: dict) -> str:
    """Build HTML for a single question."""
    qtype = q.get("type", "short_answer")
    marks = q.get("marks", "")
    number = q.get("number", "")
    text = q.get("text", "")
    parts = []

    # Question line with marks
    marks_html = f'<span class="marks">[{_esc(marks)}]</span>' if marks else ""
    parts.append(f'<div class="question"><span class="q-num">{_esc(number)}.</span> {_esc(text)} {marks_html}</div>')

    # Passage if present
    passage = q.get("passage", "")
    if passage:
        parts.append(f'<div class="passage">&ldquo;{_esc(passage)}&rdquo;</div>')

    # MCQ options
    options = q.get("options", [])
    if options:
        cols = 2 if all(len(o.get("text", "")) < 30 for o in options) else 1
        parts.append(f'<div class="options cols-{cols}">')
        for opt in options:
            parts.append(f'<div class="option">{_esc(opt.get("label", ""))}. {_esc(opt.get("text", ""))}</div>')
        parts.append("</div>")

    # Sub-parts
    sub_parts = q.get("sub_parts", [])
    if sub_parts and qtype == "map_pointing":
        parts.append('<div class="map-items">')
        for sp in sub_parts:
            parts.append(f'<div class="map-item">{_esc(sp.get("label", ""))}. {_esc(sp.get("text", ""))}</div>')
        parts.append("</div>")
    elif sub_parts:
        parts.append('<div class="sub-parts">')
        for sp in sub_parts:
            sp_marks = sp.get("marks", "")
            sp_marks_html = f'<span class="marks">[{_esc(sp_marks)}]</span>' if sp_marks else ""
            parts.append(f'<div class="sub-part">({_esc(sp.get("label", ""))}) {_esc(sp.get("text", ""))} {sp_marks_html}</div>')
        parts.append("</div>")

    return "\n".join(parts)


def _build_header_html(paper: dict) -> str:
    """Build header + meta line HTML."""
    school = _esc(paper.get("school_name", ""))
    exam = _esc(paper.get("exam_name", ""))
    cls = _esc(paper.get("class", ""))
    subject = _esc(paper.get("subject", ""))
    max_marks = _esc(paper.get("max_marks", ""))
    time_allowed = _esc(paper.get("time_allowed", ""))

    meta_left = f"Class: {cls}" if cls else ""
    if subject:
        meta_left += f" &nbsp;|&nbsp; Subject: {subject}" if meta_left else f"Subject: {subject}"
    meta_right = ""
    if max_marks:
        meta_right = f"Max Marks: {max_marks}"
    if time_allowed:
        meta_right += f" &nbsp;|&nbsp; Time: {time_allowed}" if meta_right else f"Time: {time_allowed}"

    return f"""<div class="header-block">
    <div class="school-name">{school}</div>
    <div class="exam-name">{exam}</div>
</div>
<div class="header-meta">
    <span>{meta_left}</span>
    <span>{meta_right}</span>
</div>"""


def _build_instructions_html(paper: dict) -> str:
    """Build general instructions HTML."""
    instructions = paper.get("general_instructions", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    if instructions and any(i.strip() for i in instructions):
        instr_items = "".join(f"<li>{_esc(i)}</li>" for i in instructions if i.strip())
        return f'<div class="instructions"><b>General Instructions:</b><ol>{instr_items}</ol></div>'
    return ""


def _build_sections_html(paper: dict) -> str:
    """Build all sections HTML."""
    sections_html = ""
    for sec in paper.get("sections", []):
        sec_title = _esc(sec.get("title", ""))
        sec_marks = _esc(sec.get("section_marks", ""))
        sec_title_line = sec_title
        if sec_marks and sec_marks not in sec_title:
            sec_title_line += f' <span class="sec-marks">({sec_marks})</span>'

        questions_html = ""
        for q in sec.get("questions", []):
            questions_html += build_question_html(q) + "\n"

        sec_instr = sec.get("instructions", "")
        sec_instr_html = f'<div class="sec-instructions"><i>{_esc(sec_instr)}</i></div>' if sec_instr else ""

        sections_html += f"""<div class="section">
    <div class="section-title">{sec_title_line}</div>
    {sec_instr_html}
    <div class="section-body">{questions_html}</div>
</div>
"""
    return sections_html


def build_full_html(paper: dict, half_dup: bool, show_instructions: bool = True) -> str:
    """Build complete print-ready HTML from structured paper data."""
    subject = _esc(paper.get("subject", ""))
    cls = _esc(paper.get("class", ""))
    title = f"Question Paper - {subject} {cls}".strip(" -")

    header_html = _build_header_html(paper)
    instr_html = _build_instructions_html(paper) if show_instructions else ""
    sections_html = _build_sections_html(paper)

    if half_dup:
        # HALF-PAGE DUPLICATE MODE
        # JS pagination: measure element heights, split across pages.
        # Top half = header + instructions (page 1 only) + questions
        # Bottom half (back side) = questions ONLY (no school header)
        css = _get_css_half_dup()
        body = f"""
<!-- Hidden measuring container -->
<div id="measure-box" style="position:absolute;left:-9999px;top:0;width:186mm;font-size:11px;line-height:1.35;">
  <div id="m-header">{header_html}</div>
  <div id="m-instr">{instr_html}</div>
  <div id="m-sections">{sections_html}</div>
</div>

<!-- Pages injected by JS -->
<div id="pages-container"></div>

<script>
(function() {{
  window.addEventListener('load', function() {{ setTimeout(buildPages, 200); }});

  function buildPages() {{
    var HALF_H = 530;  // ~140mm usable content height per half
    var header = document.getElementById('m-header').innerHTML;
    var instr  = document.getElementById('m-instr').innerHTML;
    var container = document.getElementById('pages-container');

    // Collect all individual items from sections
    var secDiv = document.getElementById('m-sections');
    var items = [];
    for (var i = 0; i < secDiv.children.length; i++) {{
      var sec = secDiv.children[i];
      var titleEl = sec.querySelector('.section-title');
      var instrEl = sec.querySelector('.sec-instructions');
      var bodyEl  = sec.querySelector('.section-body');
      if (titleEl) items.push(titleEl.outerHTML);
      if (instrEl) items.push(instrEl.outerHTML);
      if (bodyEl) {{
        for (var j = 0; j < bodyEl.children.length; j++) {{
          items.push(bodyEl.children[j].outerHTML);
        }}
      }}
    }}

    // Height measurer
    var measurer = document.createElement('div');
    measurer.style.cssText = 'position:absolute;left:-9999px;top:0;width:186mm;font-size:11px;line-height:1.35;';
    document.body.appendChild(measurer);
    function measureH(s) {{ measurer.innerHTML = s; return measurer.offsetHeight; }}

    var headerH = measureH(header);
    var instrH  = measureH(instr);

    // Build pages
    var pages = [];
    var curItems = [];
    var curH = 0;
    var isFirst = true;

    function flushPage() {{
      // Page 1: TOP HALF = header + instructions + questions, BOTTOM HALF = header + questions (no instr)
      // Page 2+: TOP HALF = questions only (no header), BOTTOM HALF = questions only (no header)
      var topContent = '<div class="half-inner">';
      if (isFirst) {{
        topContent += header + instr;
      }}
      topContent += '<div class="section-body">' + curItems.join('') + '</div></div>';

      var botContent = '<div class="half-inner">';
      if (isFirst) {{
        botContent += header;
      }}
      botContent += '<div class="section-body">' + curItems.join('') + '</div></div>';

      pages.push(
        '<div class="page-sheet">' +
        '  <div class="half-top">' + topContent + '</div>' +
        '  <div class="cut-line"></div>' +
        '  <div class="half-bottom">' + botContent + '</div>' +
        '</div>'
      );
      curItems = [];
      curH = 0;
      isFirst = false;
    }}

    // Available height calculation:
    // Page 1: TOP has header+instr, BOTTOM has header only → TOP is limiting (header+instr)
    // Page 2+: Neither half has header → full HALF_H available
    var availFirst = HALF_H - headerH - instrH;
    var availNext  = HALF_H;

    for (var k = 0; k < items.length; k++) {{
      var itemH = measureH(items[k]);
      var avail = isFirst ? availFirst : availNext;
      if (curH + itemH > avail && curItems.length > 0) {{
        flushPage();
      }}
      curItems.push(items[k]);
      curH += itemH;
    }}
    if (curItems.length > 0) flushPage();

    container.innerHTML = pages.join('');
    document.body.removeChild(measurer);
    document.getElementById('measure-box').style.display = 'none';
  }}
}})();
</script>
"""
    else:
        css = _get_css_normal()
        content = f"{header_html}\n{instr_html}\n{sections_html}"
        body = f'<div class="page">{content}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _get_css_normal() -> str:
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
@page { size: A4; margin: 12mm 15mm; }
body { font-family: 'Times New Roman', Times, serif; font-size: 14px; line-height: 1.6; color: #000; background: #fff; }

.page { max-width: 210mm; margin: 0 auto; padding: 10mm; }

.header-block { text-align: center; border-bottom: 2px solid #000; padding-bottom: 8px; margin-bottom: 10px; }
.school-name { font-size: 20px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
.exam-name { font-size: 15px; margin-top: 2px; }
.header-meta { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 10px; padding: 4px 0; border-bottom: 1px solid #999; }

.instructions { font-size: 12px; margin-bottom: 12px; line-height: 1.4; }
.instructions ol { margin-left: 20px; }
.instructions li { margin-bottom: 2px; }

.section { margin-bottom: 14px; }
.section-title { font-size: 14px; font-weight: bold; text-decoration: underline; margin-bottom: 8px; }
.sec-marks { font-weight: normal; text-decoration: none; font-size: 12px; }
.sec-instructions { font-size: 12px; margin-bottom: 6px; color: #333; }

.question { margin-bottom: 8px; position: relative; padding-right: 50px; }
.q-num { font-weight: bold; }
.marks { position: absolute; right: 0; top: 0; font-size: 12px; color: #333; }

.sub-parts { margin-left: 25px; margin-bottom: 8px; }
.sub-part { margin-bottom: 3px; position: relative; padding-right: 50px; }
.sub-part .marks { font-size: 11px; }

.passage { margin: 6px 0 6px 25px; padding: 6px 10px; background: #f5f5f5; border-left: 3px solid #999; font-style: italic; font-size: 13px; line-height: 1.5; }

.options { margin: 4px 0 8px 25px; }
.options.cols-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 20px; }
.options.cols-1 .option { margin-bottom: 2px; }
.option { font-size: 13px; }

.map-items { margin: 4px 0 8px 25px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2px 15px; }
.map-item { font-size: 13px; }

@media print {
    body { background: #fff; }
    .page { padding: 0; max-width: none; }
}
"""


def _get_css_half_dup() -> str:
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
@page { size: A4 portrait; margin: 0; }
html, body { width: 210mm; margin: 0 auto; font-family: 'Times New Roman', Times, serif; font-size: 11px; line-height: 1.35; color: #000; background: #fff; }

.page-sheet {
    width: 210mm;
    height: 297mm;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.half-top, .half-bottom {
    width: 100%;
    height: 148mm;
    padding: 5mm 12mm 3mm 12mm;
    overflow: hidden;
}
.cut-line {
    width: 100%;
    height: 1mm;
    border-top: 1.5px dashed #888;
    flex-shrink: 0;
}
.half-inner { height: 100%; overflow: hidden; }

.header-block { text-align: center; border-bottom: 1.5px solid #000; padding-bottom: 4px; margin-bottom: 5px; }
.school-name { font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px; }
.exam-name { font-size: 11px; margin-top: 1px; }
.header-meta { display: flex; justify-content: space-between; font-size: 10px; font-weight: 600; margin-bottom: 5px; padding-bottom: 3px; border-bottom: 1px solid #999; }

.instructions { font-size: 9px; margin-bottom: 5px; line-height: 1.2; }
.instructions ol { margin-left: 15px; }
.instructions li { margin-bottom: 1px; }

.section { margin-bottom: 5px; }
.section-title { font-size: 11px; font-weight: bold; text-decoration: underline; margin-bottom: 3px; }
.sec-marks { font-weight: normal; text-decoration: none; font-size: 9px; }
.sec-instructions { font-size: 9px; margin-bottom: 2px; color: #333; }

.question { margin-bottom: 3px; position: relative; padding-right: 35px; }
.q-num { font-weight: bold; }
.marks { position: absolute; right: 0; top: 0; font-size: 9px; color: #333; }

.sub-parts { margin-left: 18px; margin-bottom: 3px; }
.sub-part { margin-bottom: 1px; position: relative; padding-right: 35px; font-size: 10px; }
.sub-part .marks { font-size: 9px; }

.passage { margin: 3px 0 3px 18px; padding: 3px 6px; background: #f5f5f5; border-left: 2px solid #999; font-style: italic; font-size: 9px; line-height: 1.3; }

.options { margin: 2px 0 3px 18px; }
.options.cols-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1px 12px; }
.option { font-size: 10px; }

.map-items { margin: 2px 0 3px 18px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1px 10px; }
.map-item { font-size: 10px; }

.section-body { }

@media print {
    html, body { width: 210mm; }
    .page-sheet { margin: 0; page-break-after: always; }
}
"""


# ──────────────────────────────────────────────────────────────
# Helper: merge user-entered metadata into paper JSON
# ──────────────────────────────────────────────────────────────
def merge_metadata_into_paper(paper: dict) -> dict:
    """Use user-entered values as override for paper metadata."""
    paper["school_name"] = st.session_state.school_name or paper.get("school_name", "")
    paper["exam_name"] = st.session_state.exam_name or paper.get("exam_name", "")
    paper["class"] = st.session_state.class_name or paper.get("class", "")
    paper["max_marks"] = st.session_state.marks or paper.get("max_marks", "")
    paper["subject"] = st.session_state.subject or paper.get("subject", "")
    paper["time_allowed"] = st.session_state.time_allowed or paper.get("time_allowed", "")
    if st.session_state.general_instructions.strip():
        paper["general_instructions"] = [
            line.strip() for line in st.session_state.general_instructions.split("\n") if line.strip()
        ]
    return paper


# ══════════════════════════════════════════════════════════════
#  SIMPLE VERTICAL UI — just scroll down, step by step
# ══════════════════════════════════════════════════════════════

st.title("📝 Question Paper Formatter")
st.caption("Upload photos of a question paper → AI reads it → download a clean printable paper.")

# ─── STEP 1: API Key ─────────────────────────────────────────
st.markdown('<div class="step-box">Step 1 — Paste your API key</div>', unsafe_allow_html=True)
api_key = st.text_input(
    "API Key",
    type="password",
    placeholder="Paste your free OpenRouter API key here",
    help="Get a free key at https://openrouter.ai/keys",
    label_visibility="collapsed",
)

# ─── STEP 2: Upload ──────────────────────────────────────────
st.markdown('<div class="step-box">Step 2 — Upload question paper photos</div>', unsafe_allow_html=True)
st.caption("Take clear photos of each page. Any order is fine.")
uploaded = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
if uploaded:
    img_cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with img_cols[i % len(img_cols)]:
            st.image(f, caption=f.name, width="stretch")

# ─── STEP 3: Paper details (auto-filled from OCR) ────────────
st.markdown('<div class="step-box">Step 3 — Review & fix paper details</div>', unsafe_allow_html=True)
if st.session_state.ocr_done:
    st.caption("✅ Auto-filled from your photos. Fix anything the AI got wrong.")
else:
    st.caption("These will be auto-filled after scanning. Or fill manually now.")

# Auto-expand after OCR has populated the fields
_expand_details = st.session_state.ocr_done
with st.expander("✏️ Paper details", expanded=_expand_details):
    st.session_state.school_name = st.text_input("School Name", value=st.session_state.school_name)
    st.session_state.exam_name = st.text_input("Exam Name", value=st.session_state.exam_name)
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.class_name = st.text_input("Class", value=st.session_state.class_name)
    with c2:
        st.session_state.marks = st.text_input("Max Marks", value=st.session_state.marks)
    c3, c4 = st.columns(2)
    with c3:
        st.session_state.subject = st.text_input("Subject", value=st.session_state.subject)
    with c4:
        st.session_state.time_allowed = st.text_input("Time Allowed", value=st.session_state.time_allowed)

    st.markdown("---")
    st.session_state.show_instructions = st.checkbox(
        "📋 Include General Instructions on the paper",
        value=st.session_state.show_instructions,
        help="Enable to show 'General Instructions' section at the top of the question paper.",
    )
    if st.session_state.show_instructions:
        st.session_state.general_instructions = st.text_area(
            "General Instructions (one per line)",
            value=st.session_state.general_instructions,
            height=80,
        )

# ─── STEP 4: Layout + GO button ──────────────────────────────
st.markdown('<div class="step-box">Step 4 — Choose layout & go!</div>', unsafe_allow_html=True)
half_dup = st.checkbox(
    "✂️ Half-page mode (prints 2 copies per page — cut in half)",
    value=False,
)

go_clicked = st.button(
    "🔍 Read Photos & Generate Paper",
    type="primary",
    disabled=not uploaded,
    width="stretch",
)

# Also show a regenerate button if we already have data
regen_clicked = False
if st.session_state.structured_json:
    regen_clicked = st.button(
        "🔄 Regenerate paper (use after editing details above)",
        width="stretch",
        help="Click this after changing school name, toggling instructions, or editing the JSON.",
    )

# ─── Handle GO button ────────────────────────────────────────
if go_clicked:
    if not api_key:
        st.error("⬆️ Please paste your API key in Step 1 first!")
        st.stop()
    imgs = [Image.open(f) for f in uploaded]
    with st.spinner("🔍 Reading your question paper… please wait 30–60 seconds…"):
        try:
            paper = pass1_extract(imgs, api_key)

            # Auto-fill Step 3 fields from OCR (always overwrite with OCR values
            # unless the user had manually typed something beforehand)
            if not st.session_state.school_name:
                st.session_state.school_name = paper.get("school_name", "")
            if not st.session_state.exam_name:
                st.session_state.exam_name = paper.get("exam_name", "")
            if not st.session_state.class_name:
                st.session_state.class_name = paper.get("class", "")
            if not st.session_state.marks:
                st.session_state.marks = paper.get("max_marks", "")
            if not st.session_state.subject:
                st.session_state.subject = paper.get("subject", "")
            if not st.session_state.time_allowed:
                st.session_state.time_allowed = paper.get("time_allowed", "")
            gi = paper.get("general_instructions", [])
            if not st.session_state.general_instructions.strip():
                if isinstance(gi, list):
                    st.session_state.general_instructions = "\n".join(gi)
                elif gi:
                    st.session_state.general_instructions = str(gi)
            # If OCR found instructions, turn on the toggle
            if gi and any(str(x).strip() for x in (gi if isinstance(gi, list) else [gi])):
                st.session_state.show_instructions = True

            st.session_state.ocr_done = True  # expand Step 3 on rerun

            paper = merge_metadata_into_paper(paper)
            st.session_state.structured_json = json.dumps(paper, indent=2, ensure_ascii=False)
            st.session_state.final_html = build_full_html(paper, half_dup, st.session_state.show_instructions)

            total_q = sum(len(s.get("questions", [])) for s in paper.get("sections", []))
            st.success(f"✅ Found {total_q} questions! Scroll down to see the result.")
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"AI returned bad data. Try again. ({e})")
            if st.session_state.raw_text:
                with st.expander("Raw AI response"):
                    st.code(st.session_state.raw_text, language="text")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ─── Handle Regenerate button ─────────────────────────────────
if regen_clicked:
    try:
        paper = json.loads(st.session_state.structured_json)
        paper = merge_metadata_into_paper(paper)
        st.session_state.final_html = build_full_html(paper, half_dup, st.session_state.show_instructions)
        st.rerun()
    except Exception as e:
        st.error(f"Failed: {e}")

# ══════════════════════════════════════════════════════════════
#  RESULTS — only shown after paper is generated
# ══════════════════════════════════════════════════════════════
if st.session_state.final_html:
    st.markdown("---")
    st.markdown('<div class="step-box">✅ Your Question Paper</div>', unsafe_allow_html=True)

    # Preview in iframe
    html_b64 = base64.b64encode(st.session_state.final_html.encode()).decode()
    st.markdown(
        f'<iframe src="data:text/html;base64,{html_b64}" '
        f'style="width:100%;height:900px;border:1px solid #ccc;border-radius:8px;background:#fff;" '
        f'sandbox="allow-same-origin allow-scripts"></iframe>',
        unsafe_allow_html=True,
    )

    # Download buttons
    _subj = st.session_state.subject or "paper"
    _cls = st.session_state.class_name or ""
    _fname = f"{_subj}_{_cls}".strip("_").replace(" ", "_").lower() or "question_paper"

    st.markdown("")
    printable = st.session_state.final_html.replace(
        "</body>",
        '<div id="pb" style="position:fixed;bottom:20px;right:20px;z-index:9999;">'
        '<button onclick="window.print()" style="padding:14px 28px;font-size:16px;cursor:pointer;'
        'background:#1a73e8;color:white;border:none;border-radius:8px;box-shadow:0 3px 10px rgba(0,0,0,0.3);">'
        '🖨️ Print / Save as PDF</button></div>'
        '<style>@media print{#pb{display:none!important}}</style></body>',
    )
    st.download_button(
        "⬇️ Download HTML",
        data=printable,
        file_name=f"{_fname}.html",
        mime="text/html",
        width="stretch",
    )

    st.info(
        "💡 **To save as PDF:** Open the downloaded HTML file in Chrome/Edge → "
        "press **Ctrl+P** → change destination to **Save as PDF** → click Save."
    )

    # Advanced editing — hidden unless needed
    with st.expander("🔧 Advanced: Edit extracted data (if AI made mistakes)"):
        st.caption("Edit the JSON below and click 'Regenerate' above.")
        edited = st.text_area(
            "JSON",
            value=st.session_state.structured_json,
            height=400,
            key="json_ed",
            label_visibility="collapsed",
        )
        if edited != st.session_state.structured_json:
            st.session_state.structured_json = edited

    # Debug
    if st.session_state.raw_text:
        with st.expander("🔎 Raw AI response (debug)"):
            st.code(st.session_state.raw_text, language="text")

# Moqayim — Short Answer Grading (Streamlit)

Lightweight Streamlit app for short-answer grading using OCR (Groq) and a deterministic rubric engine.

Files
- [moqayim3_api2.py](moqayim3_api2.py): main Streamlit app.
- [database.py](database.py): simple SQLite helper for LTI session demo.

Requirements
- Install from `requirements.txt`.

Local setup and run
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Create a `.env` with your Groq API key for local testing:

```text
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the app:

```powershell
streamlit run moqayim3_api2.py
```

Push to GitHub
1. Initialize repo, commit, and push:

```bash
git init
git add .
git commit -m "Initial commit: Moqayim Streamlit app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Deploy on Streamlit Cloud
1. Sign in to Streamlit Cloud and create a new app from your GitHub repository.
2. Choose the `main` branch and set the main file to `moqayim3_api2.py`.
3. In the app settings, add the environment variable `GROQ_API_KEY` (and any other secrets) via the Secrets/Environment UI rather than committing `.env` to the repo.
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` and run the app.

Notes and tips
- Keep `.env` and `lti_demo.db` out of the repo. Use Streamlit Cloud secrets for `GROQ_API_KEY`.
- If you don't use Groq or don't have the key, the app will still run but OCR features will be disabled.
- If PDF OCR is used, `pdf2image` requires poppler on host machines; Streamlit Cloud usually provides necessary support, but for local development install poppler:

Windows: install Poppler and add to PATH (e.g., via conda or download binary and add `bin` folder to PATH).

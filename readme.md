![banner](https://github.com/user-attachments/assets/a9aca2db-a195-4182-a690-0cc7de0f7480)


# BDO CyberAuditBot

BDO CyberAuditBot is an AI-powered assistant for cybersecurity compliance audits, designed for BDO Singapore. It leverages LLM and RAG to provide precise, document-based answers and generate technical pentesting reports.

---

## UI
![Screenshot1](https://github.com/user-attachments/assets/f1be8b89-b0e9-4b49-9db8-0b8e5df47917)
![Screenshot2](https://github.com/user-attachments/assets/e49e8472-b64d-4bf7-8297-3098752efa0a)


## Features

- **Conversational Chatbot:** Ask questions about cybersecurity audits, compliance, and best practices.
- **Document Upload:** Enrich the knowledge base by uploading PDF, JSON, XML, or config files.
- **Automated Reporting:** Generate and download technical pentesting reports in Word format.
- **Citations:** All answers are based strictly on your uploaded documents and indexed regulatory references.
- **Professional Guidance:** Recommendations and findings are aligned with Singapore and international standards.

---

## How to Use

### 1. Requirements

- Python 3.9 or higher
- OpenAI API key (GPT-4o)
- The following files in the same folder as `chatbot.py`:
  - `BDOagent2.png`
  - `user_avatar2.png`
  - `banner.png`
- A folder with reference PDFs (default: `cybersecurity_pdf_data/`)

### 2. Installation

1. **Clone or copy** this folder (`CHATBOT`) to your computer.
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up your API key:**
   - Copy `.env.example` to `.env` and fill in your OpenAI API key.

### 3. Running the Application

From the `CHATBOT` directory, run:

```sh
python chatbot.py
```

A web interface will open in your browser.

---

## Usage Scenarios

- **Internal Audit Support:** Get instant, document-based answers for BDO Singapore’s cybersecurity audits.
- **Compliance Checks:** Verify alignment with MAS, PDPA, ISO 27001, NIST, and other frameworks.
- **Evidence Gathering:** Upload new findings or logs to enrich the context for your queries.
- **Automated Reporting:** Generate structured pentesting reports for clients or internal review.
- **Training and Onboarding:** Use as a knowledge base for new auditors or compliance staff.

---

## Security Notes

- **Never share your `.env` file or OpenAI API key.**
- Uploaded files are indexed only for your session and not shared externally.
- All answers are strictly based on the indexed documents and uploaded files.

---

## Project Structure

```
CHATBOT/
├── chatbot.py
├── requirements.txt
├── .env.example
├── readme.md
├── BDOagent2.png
├── user_avatar2.png
├── banner.png
├── cybersecurity_pdf_data/
│   └── ... (your reference PDFs)
```

---

## Author

Carlos Egana | BDO Singapore

---

_For questions or support, contact the project Author_

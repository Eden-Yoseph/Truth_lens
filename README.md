#  TruthLens — AI-Powered Fake News Detection Web App

TruthLens is an intelligent, real-time web application that analyzes news text for credibility using both AI models and custom heuristics.  
Built in under 3 hours for the Code Crafter AI Challenge, TruthLens won 🥇 **First Place**, thanks to its hybrid approach, seamless frontend, and secure backend design.

---

##  Key Features

Paste any news paragraph for instant analysis  
Multi-model AI ensemble powered by Hugging Face APIs:
- Detects misleading or toxic language  
- Classifies sentiment and emotional bias  

 Hybrid scoring algorithm combining:
- AI inference results  
- Rule-based pattern matching  
- Heuristic credibility metrics  

Plain-English explanation of red flags or trust signals  
Responsive web interface (HTML + JS) + Flask backend (Python)  
Secure token handling and API validation  
`/check`, `/bulk`, and `/health` endpoints for testing and API integration

---

##  AI & NLP Stack

TruthLens uses a weighted ensemble of pre-trained transformer models, plus handcrafted rules, to mimic how humans assess news credibility:

| Purpose                   | Model                                               |
|---------------------------|-----------------------------------------------------|
| Toxicity / Misinformation | `martin-ha/toxic-comment-model`                    |
| Sentiment Classification  | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Fallback Dialog (Optional)| `microsoft/DialoGPT-medium`                         |

Each model is accessed via Hugging Face’s hosted inference API, with error-handling, retry logic, exponential backoff, and dynamic confidence scoring.

---

## ⚙️ Intelligent Architecture

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Model Layer**     | Dynamically queries Hugging Face transformers with configurable retry/timeout |
| **Rule-Based Analyzer** | Detects suspicious formatting, clickbait phrases, emotional triggers, etc. |
| **Credibility Engine** | Scores text from 0–100 based on both AI output and statistical indicators |
| **Explainability Module** | Converts flags into readable explanations                            |
| **Frontend**        | Beautiful UI with loading states, confidence bar, and real/fake tags        |

---

##  Endpoints

| Route             | Method | Description                          |
|-------------------|--------|--------------------------------------|
| `/`               | GET    | Web UI                              |
| `/check`          | POST   | Analyze a single news text          |
| `/analyze/bulk`   | POST   | Analyze up to 10 texts at once      |
| `/samples`        | GET    | Returns pre-defined sample inputs   |
| `/health`         | GET    | Returns service and token status    |

---

##  Tech Stack

- **Backend:** Python, Flask, Requests, Hugging Face Inference API  
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)  
- **AI/NLP:** Transformers, Sentiment & Toxicity detection, Rule-based heuristics  
- **Other Tools:** Logging, CORS, Environment Validation, Retry Logic

---

## 🏁 Project Highlights

🥇 Code Crafter AI Challenge Winner  
⏱️ Built in under 3 hours  
🧠 Combines AI + logic like an actual human fact-checker  
📈 Ready for deployment or further fine-tuning

---

## 📌 Example Use Cases

- Journalism fact-checking tools  
- Social media misinformation filters  
- AI-enhanced media literacy education  
- Research prototypes for NLP classification models

---

## 🧪 Sample Output

```json
{
  "classification": "SUSPICIOUS",
  "credibility_score": 42,
  "confidence": 80,
  "explanation": "Contains 2 clickbait phrases. Heavy use of sensational language. Overuse of exclamation marks.",
  "ai_models_used": ["toxicity_detection", "sentiment_analysis"]
}

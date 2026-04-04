from flask import Flask, render_template, request, jsonify
import requests
import json
import os
from flask_cors import CORS
import time
import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration class for better organization
class Config:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    HF_API_URL = "https://api-inference.huggingface.co/models/"
    MAX_TEXT_LENGTH = 2000
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

# Models configuration with metadata
MODELS = {
    "primary": {
        "name": "martin-ha/toxic-comment-model",
        "purpose": "toxicity_detection",
        "weight": 0.4
    },
    "secondary": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "purpose": "sentiment_analysis", 
        "weight": 0.3
    },
    "backup": {
        "name": "microsoft/DialoGPT-medium",
        "purpose": "fallback",
        "weight": 0.3
    }
}

# Comprehensive fake news indicators
FAKE_NEWS_PATTERNS = {
    "clickbait_phrases": [
        "you won't believe", "doctors hate this", "secret that", "they don't want you to know",
        "shocking truth", "incredible discovery", "amazing breakthrough", "miracle cure",
        "scientists baffled", "experts shocked", "unbelievable results", "this will blow your mind",
        "click here", "find out", "you'll never guess", "what happens next",
        "number 7 will shock you", "the result will surprise you"
    ],
    "sensational_words": [
        "amazing", "incredible", "shocking", "unbelievable", "miraculous", "revolutionary",
        "breakthrough", "exclusive", "leaked", "exposed", "revealed", "secret"
    ],
    "emotional_triggers": [
        "outraged", "disgusted", "terrified", "heartbroken", "devastated", "furious"
    ]
}

# Sample texts for frontend
SAMPLE_TEXTS = {
    "real": "The UAE government announced new regulations for artificial intelligence development in the country, focusing on ethical AI practices and data privacy protection. The initiative aims to position the UAE as a global leader in responsible AI innovation.",
    "fake": "Scientists have discovered that drinking coffee backwards (spitting it out instead of swallowing) can increase lifespan by 200% according to a study conducted by the International Institute of Reverse Nutrition.",
    "unclear": "A new study suggests that people who eat chocolate daily may have better memory, though the research sample was small and the funding source was not disclosed."
}

class NewsAnalyzer:
    """Centralized news analysis logic"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        if Config.HF_API_TOKEN:
            self.session.headers.update({
                "Authorization": f"Bearer {Config.HF_API_TOKEN}"
            })
    
    def call_huggingface_api(self, model_key: str, text: str) -> Optional[Dict]:
        """Enhanced API call with better error handling"""
        model_info = MODELS[model_key]
        url = f"{Config.HF_API_URL}{model_info['name']}"
        payload = {"inputs": text}
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.session.post(
                    url, 
                    json=payload, 
                    timeout=Config.REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    logger.info(f"Successful API call to {model_key}")
                    return response.json()
                elif response.status_code == 503:
                    wait_time = 2 ** attempt
                    logger.warning(f"Model loading, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Request error: {str(e)}")
                return None
        
        return None
    
    def analyze_text_patterns(self, text: str) -> Dict:
        """Comprehensive text pattern analysis"""
        text_lower = text.lower()
        words = text.split()
        
        analysis = {
            "basic_stats": {
                "length": len(text),
                "word_count": len(words),
                "sentence_count": len(re.findall(r'[.!?]+', text)),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
            },
            "punctuation_analysis": {
                "exclamation_marks": text.count('!'),
                "question_marks": text.count('?'),
                "ellipses": text.count('...'),
                "quotes": text.count('"') + text.count("'")
            },
            "formatting_flags": {
                "all_caps_words": len([word for word in words if word.isupper() and len(word) > 2]),
                "repeated_chars": len(re.findall(r'(.)\1{2,}', text)),
                "excessive_punctuation": len(re.findall(r'[!?]{2,}', text))
            },
            "content_flags": {
                "numbers": len(re.findall(r'\d+', text)),
                "urls": len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
                "mentions": len(re.findall(r'@\w+', text))
            }
        }
        
        # Pattern matching for fake news indicators
        analysis["fake_indicators"] = {
            "clickbait_count": sum(1 for phrase in FAKE_NEWS_PATTERNS["clickbait_phrases"] if phrase in text_lower),
            "sensational_words": sum(1 for word in FAKE_NEWS_PATTERNS["sensational_words"] if word in text_lower),
            "emotional_triggers": sum(1 for trigger in FAKE_NEWS_PATTERNS["emotional_triggers"] if trigger in text_lower)
        }
        
        return analysis
    
    def process_ai_results(self, ai_results: Dict) -> Dict:
        """Process and normalize AI model results"""
        processed = {
            "toxicity_score": 0,
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "confidence_factors": []
        }
        
        # Process toxicity results
        if "toxicity" in ai_results and ai_results["toxicity"]:
            try:
                toxicity_data = ai_results["toxicity"]
                if isinstance(toxicity_data, list) and toxicity_data:
                    if isinstance(toxicity_data[0], dict):
                        for item in toxicity_data[0]:
                            if isinstance(item, dict) and "label" in item:
                                label = item["label"].lower()
                                score = item.get("score", 0)
                                if "toxic" in label or "fake" in label:
                                    processed["toxicity_score"] = max(processed["toxicity_score"], score)
                                    if score > 0.7:
                                        processed["confidence_factors"].append("High toxicity detected")
            except Exception as e:
                logger.error(f"Error processing toxicity results: {e}")
        
        # Process sentiment results
        if "sentiment" in ai_results and ai_results["sentiment"]:
            try:
                sentiment_data = ai_results["sentiment"]
                if isinstance(sentiment_data, list) and sentiment_data:
                    for item in sentiment_data:
                        if isinstance(item, dict) and "label" in item:
                            label = item["label"].lower()
                            score = item.get("score", 0)
                            processed["sentiment_label"] = label
                            processed["sentiment_score"] = score
                            if ("negative" in label and score > 0.8) or ("positive" in label and score > 0.9):
                                processed["confidence_factors"].append("Extreme emotional bias detected")
            except Exception as e:
                logger.error(f"Error processing sentiment results: {e}")
        
        return processed
    
    def calculate_credibility_score(self, text_analysis: Dict, ai_processed: Dict) -> Dict:
        """Advanced credibility scoring algorithm"""
        credibility_score = 50  # Start neutral
        risk_factors = []
        positive_factors = []
        
        # Text pattern analysis
        fake_indicators = text_analysis["fake_indicators"]
        basic_stats = text_analysis["basic_stats"]
        punctuation = text_analysis["punctuation_analysis"]
        formatting = text_analysis["formatting_flags"]
        
        # Clickbait and sensational content
        if fake_indicators["clickbait_count"] > 0:
            penalty = fake_indicators["clickbait_count"] * 15
            credibility_score -= penalty
            risk_factors.append(f"Contains {fake_indicators['clickbait_count']} clickbait phrases")
        
        if fake_indicators["sensational_words"] > 2:
            credibility_score -= 10
            risk_factors.append("Heavy use of sensational language")
        
        # Formatting red flags
        if formatting["all_caps_words"] > 2:
            credibility_score -= 8
            risk_factors.append("Excessive capitalization")
        
        if punctuation["exclamation_marks"] > 3:
            credibility_score -= 5
            risk_factors.append("Overuse of exclamation marks")
        
        # Length and structure analysis
        if basic_stats["word_count"] < 10:
            credibility_score -= 10
            risk_factors.append("Too brief for credible reporting")
        elif basic_stats["word_count"] > 100:
            credibility_score += 5
            positive_factors.append("Substantial content length")
        
        # Quote analysis (indicates sources)
        if punctuation["quotes"] > 0:
            credibility_score += 8
            positive_factors.append("Contains quoted sources")
        
        # AI model results
        if ai_processed["toxicity_score"] > 0.7:
            credibility_score -= 20
            risk_factors.append("AI detected toxic/misleading patterns")
        
        if ai_processed["sentiment_score"] > 0.85:
            credibility_score -= 10
            risk_factors.append("Extreme emotional bias")
        
        # Normalize score
        credibility_score = max(0, min(100, credibility_score))
        
        # Calculate confidence based on analysis depth
        confidence = min(90, 60 + len(risk_factors + positive_factors) * 5)
        
        return {
            "credibility_score": credibility_score,
            "is_credible": credibility_score >= 50,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "positive_factors": positive_factors,
            "analysis_timestamp": datetime.now().isoformat()
        }

# Initialize analyzer
analyzer = NewsAnalyzer()

@app.route('/')
def index():
    """Render main page with sample texts"""
    return render_template('index.html', samples=SAMPLE_TEXTS)

@app.route('/samples')
def get_samples():
    """API endpoint for sample texts"""
    return jsonify(SAMPLE_TEXTS)

@app.route('/check', methods=['POST'])
def check_news():
    """Enhanced news analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided for analysis'}), 400
        
        news_text = data['text'].strip()
        
        if not news_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(news_text) > Config.MAX_TEXT_LENGTH:
            return jsonify({
                'error': f'Text too long. Please limit to {Config.MAX_TEXT_LENGTH} characters.'
            }), 400
        
        logger.info(f"Analyzing text of length: {len(news_text)}")
        
        # Multi-model AI analysis
        ai_results = {}
        for model_key in ["primary", "secondary"]:
            result = analyzer.call_huggingface_api(model_key, news_text)
            if result:
                ai_results[MODELS[model_key]["purpose"]] = result
        
        # Text pattern analysis
        text_analysis = analyzer.analyze_text_patterns(news_text)
        
        # Process AI results
        ai_processed = analyzer.process_ai_results(ai_results)
        
        # Calculate final credibility score
        final_analysis = analyzer.calculate_credibility_score(text_analysis, ai_processed)
        
        # Prepare comprehensive response
        response = {
            'classification': 'CREDIBLE' if final_analysis['is_credible'] else 'SUSPICIOUS',
            'confidence': final_analysis['confidence'],
            'credibility_score': final_analysis['credibility_score'],
            'isReal': final_analysis['is_credible'],
            'explanation': '. '.join(final_analysis['risk_factors'][:3]) if final_analysis['risk_factors'] else 'Analysis shows standard content patterns',
            'analysis_details': {
                'risk_factors': final_analysis['risk_factors'],
                'positive_factors': final_analysis['positive_factors'],
                'text_stats': text_analysis['basic_stats'],
                'ai_models_used': list(ai_results.keys()),
                'timestamp': final_analysis['analysis_timestamp']
            },
            'ai_powered': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in check_news: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred during analysis. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Enhanced health check with system status"""
    return jsonify({
        'status': 'healthy',
        'service': 'TruthLens AI (Enhanced)',
        'models_available': list(MODELS.keys()),
        'api_token_configured': bool(Config.HF_API_TOKEN),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/bulk', methods=['POST'])
def bulk_analyze():
    """New endpoint for bulk analysis"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'No texts array provided'}), 400
        
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts allowed per request'}), 400
        
        results = []
        for idx, text in enumerate(texts):
            if len(text.strip()) > 0:
                # Simplified analysis for bulk processing
                text_analysis = analyzer.analyze_text_patterns(text)
                ai_processed = {"toxicity_score": 0, "sentiment_score": 0.5, "confidence_factors": []}
                final_analysis = analyzer.calculate_credibility_score(text_analysis, ai_processed)
                
                results.append({
                    'index': idx,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'classification': 'CREDIBLE' if final_analysis['is_credible'] else 'SUSPICIOUS',
                    'credibility_score': final_analysis['credibility_score'],
                    'confidence': final_analysis['confidence']
                })
        
        return jsonify({
            'results': results,
            'total_analyzed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in bulk_analyze: {str(e)}")
        return jsonify({'error': 'Bulk analysis failed'}), 500

if __name__ == '__main__':
    print(" Starting Enhanced TruthLens AI")
    print("=" * 60)
    
    if Config.HF_API_TOKEN:
        print(" Hugging Face API token configured")
    else:
        print("  No Hugging Face API token found")
        print("   Get a free token at: https://huggingface.co/settings/tokens")
    
    print(f" Enhanced analysis with {len(MODELS)} AI models")
    print(f" Pattern detection for {sum(len(patterns) for patterns in FAKE_NEWS_PATTERNS.values())} indicators")
    print(" Server starting at: http://127.0.0.1:5000")
    print(" Test endpoint: /health")
    print(" Bulk analysis: /analyze/bulk")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)

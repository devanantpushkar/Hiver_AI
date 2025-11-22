from textblob import TextBlob
import pandas as pd
import re

TEST_EMAILS = [
    {"id": 1, "subject": "Unable to access shared mailbox", 
     "body": "Hi team, I'm unable to access the shared mailbox. It keeps showing a permissions error. Can you please check?", 
     "expected": "neutral"},
    {"id": 2, "subject": "Great experience!", 
     "body": "Your support team resolved my issue within 10 minutes. Absolutely fantastic service!", 
     "expected": "positive"},
    {"id": 3, "subject": "Terrible product", 
     "body": "This is the worst software I've ever used. Nothing works as advertised. Complete waste of money.", 
     "expected": "negative"},
    {"id": 4, "subject": "Query about billing", 
     "body": "I noticed a charge on my account and wanted to understand what it's for. Could you provide details?", 
     "expected": "neutral"},
    {"id": 5, "subject": "Frustrated with delays", 
     "body": "I've been waiting for 3 days for a response. This is unacceptable and very disappointing.", 
     "expected": "negative"},
    {"id": 6, "subject": "Thank you!", 
     "body": "Just wanted to say thanks for the quick fix. Really appreciate the help.", 
     "expected": "positive"},
    {"id": 7, "subject": "Feature request", 
     "body": "It would be nice if you could add dark mode. Not urgent, just a suggestion for the future.", 
     "expected": "neutral"},
    {"id": 8, "subject": "Disappointed customer", 
     "body": "I expected better from your company. The product keeps crashing and support hasn't helped at all. Very frustrated.", 
     "expected": "negative"},
    {"id": 9, "subject": "Love this tool!", 
     "body": "Been using your product for 6 months now and it's made our team so much more productive. Highly recommend!", 
     "expected": "positive"},
    {"id": 10, "subject": "Question about API", 
     "body": "I'm trying to integrate with your API. Does endpoint X support pagination? Thanks.", 
     "expected": "neutral"}
]

class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = {
            'thank', 'thanks', 'great', 'excellent', 'fantastic', 'love', 'appreciate',
            'wonderful', 'amazing', 'perfect', 'awesome', 'helpful', 'happy', 'satisfied'
        }
        self.negative_words = {
            'terrible', 'worst', 'frustrated', 'frustrating', 'disappointed', 'disappointing',
            'unacceptable', 'useless', 'horrible', 'awful', 'angry', 'upset', 'waste'
        }
        self.neutral_indicators = {
            'question', 'query', 'how', 'can you', 'could you', 'please help', 
            'i want', 'request', 'need help', 'wondering'
        }
    
    def analyze_v1(self, subject, body):
        text = f"{subject} {body}"
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = "positive"
            confidence = min(0.5 + abs(polarity), 0.9)
        elif polarity < -0.1:
            sentiment = "negative"
            confidence = min(0.5 + abs(polarity), 0.9)
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        reasoning = f"TextBlob polarity: {polarity:.2f}"
        return sentiment, confidence, reasoning
    
    def analyze_v2(self, subject, body):
        text = f"{subject} {body}".lower()
        words = set(re.findall(r'\b\w+\b', text))
        
        pos_count = len(words & self.positive_words)
        neg_count = len(words & self.negative_words)
        neutral_matches = sum(1 for phrase in self.neutral_indicators if phrase in text)
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if pos_count > neg_count and (pos_count >= 2 or polarity > 0.2):
            sentiment = "positive"
            confidence = min(0.75 + pos_count * 0.05, 0.95)
            reasoning = f"Positive keywords: {pos_count}, polarity: {polarity:.2f}"
            
        elif neg_count > pos_count and (neg_count >= 1 or polarity < -0.1):
            sentiment = "negative"
            confidence = min(0.75 + neg_count * 0.05, 0.95)
            reasoning = f"Negative keywords: {neg_count}, polarity: {polarity:.2f}"
            
        else:
            if neutral_matches >= 1 or abs(polarity) < 0.15:
                sentiment = "neutral"
                confidence = 0.70 + neutral_matches * 0.05
                reasoning = f"Neutral indicators: {neutral_matches}, polarity: {polarity:.2f}"
            else:
                sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
                confidence = 0.65
                reasoning = f"Fallback to polarity: {polarity:.2f}"
        
        return sentiment, confidence, reasoning

def evaluate(emails, analyzer_func, version):
    results = []
    for email in emails:
        sentiment, confidence, reasoning = analyzer_func(email['subject'], email['body'])
        correct = sentiment == email['expected']
        results.append({
            'id': email['id'],
            'subject': email['subject'][:40],
            'expected': email['expected'],
            'predicted': sentiment,
            'confidence': confidence,
            'correct': correct,
            'reasoning': reasoning
        })
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    
    return results, accuracy, avg_conf

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    print("="*80)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("="*80)
    
    v1_results, v1_acc, v1_conf = evaluate(TEST_EMAILS, analyzer.analyze_v1, "V1")
    v2_results, v2_acc, v2_conf = evaluate(TEST_EMAILS, analyzer.analyze_v2, "V2")
    
    print(f"\n{'Version':<10} {'Accuracy':<15} {'Avg Confidence':<15}")
    print("-"*40)
    print(f"{'V1':<10} {v1_acc:.1%} ({sum(r['correct'] for r in v1_results)}/10){'':>3} {v1_conf:.2f}")
    print(f"{'V2':<10} {v2_acc:.1%} ({sum(r['correct'] for r in v2_results)}/10)

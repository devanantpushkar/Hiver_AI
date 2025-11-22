import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq

KB_ARTICLES = [
    {
        "id": "kb_001",
        "title": "Configuring Automations in Hiver",
        "content": """To configure automations: 1) Go to Settings > Automations 2) Click Create New 
        3) Set trigger (email arrives, tag added, etc) 4) Define conditions (subject contains, 
        from domain, etc) 5) Choose actions (assign, tag, auto-reply) 6) Test and activate. 
        Automations process in 1-2 minutes. Check logs if issues occur."""
    },
    {
        "id": "kb_002",
        "title": "CSAT Not Appearing - Troubleshooting",
        "content": """If CSAT is not visible: 1) Verify CSAT enabled in Settings > Analytics > CSAT 
        2) Ensure conversations are marked 'Closed' not 'Pending' - surveys only sent on close 
        3) Check analytics permissions 4) Clear browser cache 5) Verify no filters applied. 
        CSAT has rate limiting - won't survey same customer repeatedly."""
    },
    {
        "id": "kb_003",
        "title": "Setting Up SLAs",
        "content": """Configure SLAs at Settings > SLAs. Create policy with targets for first response 
        and resolution times. Set business hours. Assign to mailboxes/tags. SLA breaches trigger 
        escalations and priority changes."""
    },
    {
        "id": "kb_004",
        "title": "Email Sync Issues",
        "content": """For sync problems: Check IMAP credentials, verify mailbox permissions, 
        reduce sync frequency if server overloaded. Max 50,000 emails per mailbox. Attachments 
        over 25MB may fail."""
    },
    {
        "id": "kb_005",
        "title": "Tag Management",
        "content": """Create tags at Settings > Tags. Choose name and color. Tags can be manual, 
        auto-applied via automations, or AI-suggested. Keep names consistent and limit to 20-30 
        tags total."""
    },
    {
        "id": "kb_006",
        "title": "Shared Mailbox Permissions",
        "content": """Permission levels: Admin (full access), Manager (analytics, users), Agent 
        (respond to emails), Viewer (read-only). Grant access at Settings > Mailboxes > Manage Access. 
        Permission denied errors mean user not added or wrong role."""
    },
    {
        "id": "kb_007",
        "title": "Analytics and Reports",
        "content": """View metrics in Analytics sidebar: response time, resolution time, email volume, 
        CSAT, SLA compliance. Filter by date, tags, agents, mailboxes. Export as CSV/PDF. 
        Stats update hourly."""
    },
    {
        "id": "kb_008",
        "title": "Email Sending Issues",
        "content": """If emails stuck in outbox: Check connection, verify SMTP settings, ensure 
        attachments under 25MB. If not delivered: check recipient address, domain not blocked, 
        SPF/DKIM configured. Ask recipients to check spam."""
    },
    {
        "id": "kb_009",
        "title": "Mail Merge",
        "content": """For mail merge: Compose > Mail Merge. Upload CSV with email,name columns. 
        Use {{name}},{{company}} tags. Preview before sending. Daily limit 2000 emails. 
        CSV errors usually from special characters or empty rows."""
    },
    {
        "id": "kb_010",
        "title": "API and Webhooks",
        "content": """Hiver API at https://api.hiver.com/v1. Use API key in X-API-Key header. 
        Rate limit 100 req/min. Webhooks available for conversation events. See docs.hiver.com/api"""
    }
]

class SimpleRAG:
    def __init__(self, groq_api_key="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"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.articles = KB_ARTICLES
        self.embeddings = None
        self.groq_client = Groq(api_key=groq_api_key or os.getenv('GROQ_API_KEY'))
        self._create_embeddings()
    
    def _create_embeddings(self):
        texts = [f"{a['title']} {a['content']}" for a in self.articles]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=False)
    
    def retrieve(self, query, top_k=3):
        query_embedding = self.encoder.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'id': self.articles[idx]['id'],
                'title': self.articles[idx]['title'],
                'content': self.articles[idx]['content'],
                'score': float(similarities[idx])
            })
        return results
    
    def generate_answer(self, query, retrieved_docs):
        context = "\n\n".join([
            f"Article: {doc['title']}\n{doc['content']}" 
            for doc in retrieved_docs
        ])
        
        prompt = f"""You are a helpful support assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Provide a clear, concise answer. If the context doesn't contain the answer, say so. Cite the article title when relevant."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            avg_score = np.mean([doc['score'] for doc in retrieved_docs])
            confidence = min(avg_score * 1.2, 0.95)
            return answer, confidence
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", 0.0
    
    def query(self, question, top_k=3):
        retrieved = self.retrieve(question, top_k)
        answer, confidence = self.generate_answer(question, retrieved)
        return {
            'query': question,
            'answer': answer,
            'retrieved': retrieved,
            'confidence': confidence
        }

if __name__ == "__main__":
    rag = SimpleRAG()
    
    queries = [
        "How do I configure automations in Hiver?",
        "Why is CSAT not appearing?"
    ]
    
    for query in queries:
        result = rag.query(query)
        print(result['answer'])
        for doc in result['retrieved']:
            print(doc['title'], doc['score'])
    
    failure_query = "What is the meaning of life?"
    result = rag.query(failure_query, top_k=3)
    print(result['answer'][:150])
    print(result['confidence'])

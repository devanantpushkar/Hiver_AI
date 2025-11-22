# *1. SimpleRAG (mini_rag.py)* 

### *Description*

A minimal Retrieval-Augmented Generation pipeline using:

* *SentenceTransformer (MiniLM-L6-v2)* for embeddings
* *Cosine similarity* for retrieval
* *Groq LLaMA-3.1-8B-Instant* for answer generation
* A small built-in knowledge base of support articles

### *Features*

* Precomputes embeddings for a knowledge base
* Retrieves top-k relevant articles
* Generates answers strictly from provided context
* Returns answer + retrieved documents + confidence score

### *Usage*

bash
python mini_rag.py


### *Example Query*


"How do I configure automations in Hiver?"


---

# *2. Email Auto-Tagger (email_tag.py)* 

### *Description*

A customer-specific email tagging system combining:

* Pattern-based detection using regular expressions
* Per-customer ML classifiers using TF-IDF + Logistic Regression
* Fallback strategies for customers with limited data

### *Dataset*

A small internal CSV (SMALL_CSV) is embedded directly for demonstration.

### *Pipeline*

1. Normalize text
2. Build dataset per customer
3. Train or fallback depending on data distribution:

   * ML classifier
   * Majority-class predictor
   * Rule-based only

### *Prediction Logic*

The function:

python
predict_tag(subject, body, customer_id)


returns:

* predicted tag
* confidence score
* source type (pattern, model, fallback, etc.)

### *Run Demo*

bash
python email_tag.py


Outputs:

* Per-customer training summary
* Classification reports
* Live predictions on example emails

---

# *3. Sentiment Analyzer (sentiment.py)* 

### *Description*

Implements two sentiment-analysis models:

#### *Version 1 – TextBlob-based*

* Uses TextBlob polarity to classify into
  positive, negative, neutral

#### *Version 2 – Hybrid Keyword + Polarity*

* Custom positive/negative word lists
* Neutral indicators
* Combines keyword counts with polarity for higher accuracy

### *Evaluation*

Runs both versions on a set of 10 test emails.

### *Usage*

bash
python sentiment.py


### *Outputs*

* Sentiment prediction per email
* Confidence score
* Reasoning
* Model-wise accuracy table

---

# *Installation*

### *Dependencies*

Install required packages:

bash
pip install sentence-transformers scikit-learn groq textblob pandas numpy
python -m textblob.download_corpora


---

# *Project Structure*


.
├── mini_rag.py          # RAG system
├── email_tag.py         # Email auto-tagging system
└── sentiment.py         # Sentiment analysis system


---

# *How to Extend*

### *For RAG*

* Replace KB_ARTICLES with your own dataset
* Add caching for embeddings
* Switch to a larger Groq model if required

### *For Email Tagging*

* Load external CSV for real datasets
* Add more pattern rules
* Enhance per-customer personalization

### *For Sentiment Analysis*

* Add domain-specific sentiment lexicons
* Replace TextBlob with transformer-based sentiment models

---

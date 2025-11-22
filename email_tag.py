#!/usr/bin/env python3

import csv
import re
from io import StringIO
from collections import defaultdict, Counter
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SMALL_CSV = '''email_id,customer_id,subject,body,tag
1,CUST_A,"Unable to access shared mailbox","Hi team, I'm unable to access the shared mailbox for our support team. It keeps showing a permissions error. Can you please check?","access_issue"
2,CUST_A,"Rules not working","We created a rule to auto-assign emails based on subject line but it stopped working since yesterday.","workflow_issue"
3,CUST_A,"Email stuck in pending","One of our emails is stuck in pending even after marking it resolved. Not sure what’s happening.","status_bug"
4,CUST_B,"Automation creating duplicate tasks","Your automation engine is creating 2 tasks for every email. This started after we edited our workflow.","automation_bug"
5,CUST_B,"Tags missing","Many of our tags are not appearing for new emails. Looks like the tagging model is not working for us.","tagging_issue"
6,CUST_B,"Billing query","We were charged incorrectly this month. Need a corrected invoice.","billing"
7,CUST_C,"CSAT not visible","CSAT scores disappeared from our dashboard today. Is there an outage?","analytics_issue"
8,CUST_C,"Delay in email loading","Opening a conversation takes 8–10 seconds. This is affecting our productivity.","performance"
9,CUST_C,"Need help setting up SLAs","We want to configure SLAs for different customer tiers. Can someone guide us?","setup_help"
10,CUST_D,"Mail merge failing","Mail merge is not sending emails even though the CSV is correct.","mail_merge_issue"
11,CUST_D,"Can't add new user","Trying to add a new team member but getting an 'authorization required' error.","user_management"
12,CUST_D,"Feature request: Dark mode","Dark mode would help during late-night support hours. Please consider this.","feature_request"
'''

def csv_to_records(csv_text):
    reader = csv.DictReader(StringIO(csv_text))
    return [r for r in reader]

def normalize_text(s):
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

PATTERNS = {
    'billing': [r'\bbill', r'\binvoice', r'\bcharged', r'\brefund'],
    'automation_bug': [r'duplicate tasks', r'creating 2 tasks', r'\bduplicate\b'],
    'access_issue': [r'permission', r'permission denied', r'unable to access', r'shared mailbox'],
    'performance': [r'\bseconds\b', r'\bslow\b', r'\bdelay\b', r'loading takes'],
    'feature_request': [r'dark mode', r'feature request', r'please consider', r'request'],
    'csat_issue': [r'csat', r'survey'],
}

COMPILED_PATTERNS = {tag: [re.compile(p) for p in pats] for tag, pats in PATTERNS.items()}

def apply_patterns(normalized_text, allowed_tags=None):
    for tag, regexes in COMPILED_PATTERNS.items():
        for rx in regexes:
            if rx.search(normalized_text):
                if allowed_tags is None or tag in allowed_tags:
                    return tag
    return None

records = csv_to_records(SMALL_CSV)
for r in records:
    r['text'] = normalize_text((r.get('subject','') or '') + ' ' + (r.get('body','') or ''))

customers = defaultdict(list)
for r in records:
    customers[r['customer_id']].append(r)

print("Customers and counts:")
for c, rows in customers.items():
    print(c, len(rows))
print()

models = {}
customer_tags = {}

for cust_id, rows in customers.items():
    texts = [r['text'] for r in rows]
    labels = [r['tag'] for r in rows]
    tag_counts = Counter(labels)
    customer_tags[cust_id] = set(labels)

    print(f"--- Customer {cust_id}: {len(rows)} emails, tags: {dict(tag_counts)} ---")

    if len(rows) < 2:
        print(f"  Too few examples ({len(rows)}); using rule-based fallback.")
        models[cust_id] = None
        continue

    if len(set(labels)) == 1:
        print("  Only one class present; storing majority-class predictor (no ML).")
        models[cust_id] = {'type': 'majority', 'label': labels[0]}
        continue

    min_label_count = min(tag_counts.values())
    try:
        if min_label_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.33, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.33, random_state=42, stratify=None
            )

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1)),
            ('clf', LogisticRegression(max_iter=400))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        print("  Trained ML model. Evaluation on held-out split:")
        print(classification_report(y_test, preds, zero_division=0))
        models[cust_id] = {'type': 'ml', 'model': pipe}
    except Exception as e:
        print("  Training failed with exception, using rule-based fallback. Exception:", e)
        models[cust_id] = None

print("\nTraining complete.\n")

def predict_tag(subject, body, customer_id):
    text = normalize_text((subject or '') + ' ' + (body or ''))
    allowed_tags = customer_tags.get(customer_id, set())

    p = apply_patterns(text, allowed_tags if allowed_tags else None)
    if p is not None:
        return p, 0.99, 'pattern'

    model_info = models.get(customer_id)
    if model_info is None:
        tags = list(allowed_tags)
        if tags:
            most = Counter([r['tag'] for r in customers.get(customer_id, [])]).most_common(1)[0][0]
            return most, 0.5, 'fallback_most_common'
        else:
            return 'unknown', 0.0, 'no_data'

    if model_info['type'] == 'majority':
        return model_info['label'], 0.6, 'majority'

    pipe = model_info['model']
    try:
        probs = pipe.predict_proba([text])[0]
        labels = pipe.classes_
        idx = int(np.argmax(probs))
        pred_label = labels[idx]
        if allowed_tags and pred_label not in allowed_tags:
            allowed_idx = [i for i,l in enumerate(labels) if l in allowed_tags]
            if allowed_idx:
                best_i = max(allowed_idx, key=lambda i: probs[i])
                return labels[best_i], float(probs[best_i]), 'model_filtered'
            else:
                most = Counter([r['tag'] for r in customers.get(customer_id, [])]).most_common(1)[0][0]
                return most, 0.4, 'fallback_mismatch'
        return pred_label, float(probs[idx]), 'model'
    except Exception as e:
        most = Counter([r['tag'] for r in customers.get(customer_id, [])]).most_common(1)[0][0]
        return most, 0.3, 'inference_error'

print("Demo predictions on the small dataset:")
for r in records:
    pred, conf, source = predict_tag(r['subject'], r['body'], r['customer_id'])
    print(f"Email {r['email_id']} | Cust {r['customer_id']} | Actual: {r['tag']} -> Pred: {pred} (conf={conf:.2f}, src={source})")

print("\nExample new email prediction:")
s = "We were charged twice on our invoice this month"
b = "Please correct the billing and send a refund."
pred, conf, src = predict_tag(s, b, "CUST_B")
print(f"Predicted: {pred} (confidence={conf}, source={src})")

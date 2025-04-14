import json

with open('gpt_output.json','r',encoding='utf-8') as f:
    best_hypotheses = json.load(f)
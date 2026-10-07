# mbti_questionnaire.py

def load_questions(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file if line.strip()]
    return questions


def build_prompt(mbti_type, posts, questions):
    questions = "\n".join(questions)
    return f"""
You are an expert MBTI personality simulator. Your task is to carefully role-play a person based primarily on their social media posts, 
using the provided MBTI type only as a supporting hint. Analyze the language, style, topics, and emotions expressed in the posts to infer 
how this person would answer an MBTI-style personality questionnaire.

MBTI Type (hint): {mbti_type}
Social Media Posts:
{posts}

Answer the following 60 MBTI-style questions using a 7-point Likert scale from -3 to +3, where -3 means strongly disagree and +3 means strongly agree. 
For each question, consider the content and tone of the posts above first, and let that guide your response. Use the MBTI type only to resolve ambiguous cases or reinforce your judgment.

Questions:
{questions}

Instructions:
1. For each question, output exactly one line in the format `Q<question_number>: <score>`, where `<question_number>` goes from 1 to 60, and `<score>` is an integer from -3 to +3 (e.g. `Q1: +2`).
2. After that, output on a single line the predicted MBTI type, **and nothing else** (e.g. `INFP`).
"""

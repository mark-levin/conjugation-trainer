from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/generate', methods=['POST'])
def generate_sentence():
    data = request.json
    verbs = data['verbs']
    tenses = data['tenses']

    completion1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are used as a conjugation trainer for the Italian language. 
             You should generate a sentence that has one of the verbs listed by the user. 
             Order of the verbs doesn't matter.
             Order of the tenses doesn't matter.
             The verb can be in one of the tenses listed by the user.
             The verb itself should be masked.
             Use either (a) proper noun(s) or (a) pronoun(s) as an object.
             You should also mention the infinitive of the verb and the required tense in brackets before the mask. 
             Don't output anything else. 

             Examples:
             Q: Verbs: potere, volere, essere | Tenses: presente, imperfetto, passato remoto
             A: Marco and Paolo (volere, passato remoto) _____ andare in vacanza l'estate scorsa.
             
             Q: Verbs: venire, andare | Tenses: presente, futuro
             A: Domani, (noi) (venire, futuro) tutti alla festa per celebrare il tuo compleanno.
             """},
            {"role": "user", "content": f"Verbs: {verbs} | Tenses: {tenses}"}
        ],
        temperature=1.5
    )

    sentence = completion1.choices[0].message.content
    return jsonify({"sentence": sentence})

@app.route('/check', methods=['POST'])
def check_answer():
    data = request.json
    sentence = data['sentence']
    answer = data['answer']

    completion2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are used as a conjugation trainer for the Italian language. 
             You are given a sentence with a masked verb conjugation and an infinitive of the verb and a required tense in brackets before the mask.
             You are also given an answer to the mask by a student.
             Your task is to check whether the answer is correct or wrong.
             You should only output 'Correct' or 'Incorrect'. Don't output anything else.
             
             Examples:
             Q: Marco and Paolo (volere, passato remoto) _____ andare in vacanza l'estate scorsa. - vollero
             A: Correct
             
             Q: Domani, (noi) (venire, futuro) tutti alla festa per celebrare il tuo compleanno. - verranno
             A: Incorrect
             """},
            {"role": "user", "content": f"{sentence} - {answer}"}
        ]
    )

    result = completion2.choices[0].message.content
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
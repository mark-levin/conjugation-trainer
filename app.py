from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import os
import random
import re
from itertools import product
import uuid

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/generate', methods=['POST'])
def generate_sentence():
    data = request.json
    verbs = [x.strip() for x in data['verbs'].split(',')]
    tenses = [x.strip() for x in data['tenses'].split(',')]
    people = ['first', 'second', 'third']
    numbers = ['singular', 'plural']

    # Ensure randomness
    combinations = list(product(verbs, tenses, people, numbers))
    selection = dict(zip(['verb', 'tense', 'person', 'number'], random.choice(combinations)))

    completion1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are used as a conjugation trainer for the Italian language. 
             You should generate a sentence that has one of the verbs listed by the user. 
             Order of the verbs doesn't matter.
             Order of the tenses doesn't matter.
             The verb can be in one of the tenses listed by the user.
             The verb itself should be masked.
             The subject of the sentence should be in the {selection['person']} person and {selection['number']} .
             Use either (a) proper noun(s) or (a) pronoun(s) as the subject.
             You should also mention the infinitive of the verb and the required tense in brackets before the mask. 
             Don't output anything else. 

             Examples:
             Q: Verb: volere | Tense: passato remoto
             A: Marco and Paolo (volere, passato remoto) _____ andare in vacanza l'estate scorsa.
             
             Q: Verb: venire | Tense: futuro
             A: Domani, (noi) (venire, futuro) tutti alla festa per celebrare il tuo compleanno.
             """},
            {"role": "user", "content": f"Verb: {selection['verb']} | Tense: {selection['tense']}"}
        ],
        temperature=1.5
    )

    sentence = completion1.choices[0].message.content
    
    # Generate the voice file
    audio_file = voice_sentence(sentence)
    
    return jsonify({"sentence": sentence, "audio_file": audio_file})

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

@app.route('/voice', methods=['POST'])
def voice_sentence(sentence):
    pattern = r'^(.*?)\s+\((.*?)\)\s+(.*)$'
    match = re.match(pattern, sentence)
    
    first_part = match.group(1).strip()
    second_part = match.group(3).strip()
    
    prompt = f'{first_part} <break time="2s" /> {second_part}'

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=prompt
    )

    unique_filename = f"speech_{uuid.uuid4()}.mp3"
    audio_path = os.path.join('static', unique_filename)
    response.stream_to_file(audio_path)

    return audio_path

if __name__ == '__main__':
    app.run(debug=True)
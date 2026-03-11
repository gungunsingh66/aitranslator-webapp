from flask import Flask,render_template, request
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

app = Flask(__name__)

#to select the model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
#to load the model
model = MBartForConditionalGeneration.from_pretrained(model_name)
#to load the tokenizer
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

languages = {
"English":"en_XX",
"Hindi":"hi_IN",
"French":"fr_XX",
"Spanish":"es_XX",
"German":"de_DE",
"Italian":"it_IT",
"Arabic":"ar_AR",
"Russian":"ru_RU",
"Chinese":"zh_CN",
"Japanese":"ja_XX"
}

def translation(data, source_lang, target_lang):
    # Set the source language of the input text
    tokenizer.src_lang = source_lang
    # Convert the input text into tokens (numbers) that the model understands
    # return_tensors="pt" converts tokens into PyTorch tensors
    encoded = tokenizer(data, return_tensors="pt")
    # Generate translated tokens using the model
    # forced_bos_token_id tells the model which language to generate
    # Example: if target_lang = "hi_IN", the output will be Hindi

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
        )
    # Convert generated tokens back into readable text
    # skip_special_tokens=True removes tokens like <s>, </s>
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # batch_decode returns a list, so we return the first translated sentence
    return result[0]

@app.route('/', methods =['POST','GET'])
def index():
    translated_text = ""
    if request.method == 'POST':
        data = request.form['data']
        source = request.form['source_lang']
        target = request.form['target_lang']

        if data:
            translated_text = translation(data, source, target)

    return render_template('index.html',languages=languages,translated_text = translated_text)

if __name__ == '__main__':
    app.run(debug = True)

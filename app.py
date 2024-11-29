from flask import Flask, request, render_template, redirect, url_for, flash
import os
import easyocr
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.secret_key = 'your_secret_key'  # Needed for flashing messages

def get_model_and_tokenizer(source_language, target_language):
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def extract_text_from_image(image_path, source_language):
    reader = easyocr.Reader([source_language, 'en'], gpu=False)
    result = reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def translate_text(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def is_language_pair_compatible(source_language, target_language):
    compatible_pairs = {
        'en': ['fr', 'de', 'es', 'it', 'nl', 'ru', 'zh', 'ja', 'ko', 'th'],
        'fr': ['en', 'de', 'es', 'it', 'nl', 'ru', 'zh', 'ja', 'ko'],
        'de': ['en', 'fr', 'es', 'it', 'nl', 'ru', 'zh', 'ja', 'ko'],
        'es': ['en', 'fr', 'de', 'it', 'nl', 'ru', 'zh', 'ja', 'ko'],
        'it': ['en', 'fr', 'de', 'es', 'nl', 'ru', 'zh', 'ja', 'ko'],
        'nl': ['en', 'fr', 'de', 'es', 'it', 'ru', 'zh', 'ja', 'ko'],
        'ru': ['en', 'fr', 'de', 'es', 'it', 'nl', 'zh', 'ja', 'ko'],
        'zh': ['en', 'fr', 'de', 'es', 'it', 'nl', 'ru', 'ja', 'ko'],
        'ja': ['en', 'fr', 'de', 'es', 'it', 'nl', 'ru', 'zh', 'ko'],
        'ko': ['en', 'fr', 'de', 'es', 'it', 'nl', 'ru', 'zh', 'ja'],
        'th': ['en']
    }
    return target_language in compatible_pairs.get(source_language, [])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        source_language = request.form['source_language']
        target_language = request.form['target_language']
        
        if not is_language_pair_compatible(source_language, target_language):
            flash(f'The language pair {source_language} to {target_language} is not compatible.')
            return redirect(request.url)
        
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            extracted_text = extract_text_from_image(file_path, source_language)
            model, tokenizer = get_model_and_tokenizer(source_language, target_language)
            translated_text = translate_text(extracted_text, model, tokenizer)
            
            return render_template('result.html', 
                                   image_path=file_path,
                                   extracted_text=extracted_text, 
                                   translated_text=translated_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
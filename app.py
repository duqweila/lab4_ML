import re

from flask import Flask, render_template, request
from translator import external_translate # Assume this is the file and function you have for translation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate():

        text = request.form.get('text', '')
        if text:
            # Add spaces between Chinese characters
            if re.search("[\u4e00-\u9FFF]", text):
                text = " ".join(text)

            translated_text = external_translate(text)
    #     else:
    #         translated_text = 'Please input text for translation.'
    # except Exception as e:
    #     translated_text = f'Error occurred during translation: {str(e)}'

        return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
from generate import TextGeneration, LSTMLanguageModel

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
9
@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        seq_len = int(request.form['len'])
        prompt = request.form['prompt']
        temp = float(request.form['temp'])
        # Generate text based on the prompt using the language model
        generator = TextGeneration()
        generated_text = generator.generate(prompt, seq_len, temp, seed=0)
        return render_template('result.html', prompt = prompt, seq_len = seq_len, temp = temp, generated_text=generated_text)
 
if __name__ == '__main__':
    app.run(debug=True)

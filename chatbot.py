from flask import Flask, render_template, request
import requests
import wikipedia

app = Flask(__name__)


def get_crop_summary(crop_name):
    try:
        summary = wikipedia.summary(crop_name, sentences=3)
        return summary
    except Exception as e:
        return f"Error fetching data: {e}"


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message'].strip()
    response = get_crop_summary(user_input)
    return render_template('chat.html', user_input=user_input, response=response)

if __name__ == "__main__":
    app.run(debug=True)

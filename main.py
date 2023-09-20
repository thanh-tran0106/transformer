# Import the required modules from the Transformers and Flask libraries.
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Start a Flask application
app = Flask(__name__)

# Load the tokenizer and the translation model for English to Finnish
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fi")

# Load the translation model itself from the model hub of Hugging Face.
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fi")

# Create an index route that can take GET and POST queries
@app.route("/", methods=['GET', 'POST'])
def index():
    # Verify that POST is the request method
    if request.method == 'POST':
       # Using the 'text_to_translate' key, retrieve the text from the form
        text_input = request.form.get('text_to_translate')

       # Verify whether text is present
        if text_input:
            # Turn the input text into PyTorch tensors after tokenizing it
            encoded_inputs = tokenizer.encode(text_input, return_tensors="pt")

            # Produce the translation based on the model
            translation_outputs = model.generate(encoded_inputs)

            # Decode the generated output to get the translated text
            result_text = tokenizer.decode(translation_outputs[0], skip_special_tokens=True)

	    # Display the index.html template and create the translated text to it
            return render_template('index.html', translated_text=result_text)

    # For GET requests (or POST requests without text), display the index.html page 
    return render_template('index.html')

#Run the application; if this script is being used as the main programme, the application will launch in debug mode
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


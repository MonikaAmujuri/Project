from flask import Flask, request, jsonify
from flask_cors import CORS
from store_pdf_to_vectors import get_text_chunks, get_vector_store, user_input
from io import BytesIO
from PyPDF2 import PdfReader
import html

app = Flask(__name__)
CORS(app)


# Endpoint to upload PDF file
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the content of the file
    file_content = file.read()

    # Pass the file content to PdfReader
    pdf_reader = PdfReader(BytesIO(file_content))

    # Extract text from the PDF file
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    print(pdf_text)
    excape_unicode_text = html.unescape(pdf_text)

    text_chunks = get_text_chunks(excape_unicode_text)

    get_vector_store(text_chunks)
    # Return the extracted text
    return jsonify({'message':'pdf text stored in FIASS index'}), 200


@app.route('/get_response_for_user_prompt', methods=['POST'])
def get_response_for_user_prompt():
    data = request.json
    user_prompt = data.get('user_prompt')
    
    result = user_input(user_prompt)
    
    # Return the response for the user prompt
    response = jsonify(html.unescape(result))
    # CORS(app)
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200


if __name__ == '__main__':
    app.run(debug=True)

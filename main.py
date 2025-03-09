from flask import Flask , send_from_directory, jsonify, request
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from PyPDF2 import PdfReader, PdfWriter
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


app = Flask(__name__,static_folder=r'.\build')
# MongoDB connection string
uri = "mongodb+srv://admin:rootpassword@gokul.3p5ox.mongodb.net/?retryWrites=true&w=majority&appName=gokul"
client = MongoClient(uri, server_api=ServerApi('1'))

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    try:
        client.admin.command('ping')
        return jsonify(
            status="success",
            message="Pinged your deployment. You successfully connected to MongoDB!"
        ), 200
    except Exception as e:
        return jsonify(
            status="error",
            message="Failed to connect to MongoDB.",
            error=str(e)
        ), 500


# Access the database and collection
db = client["covenants"]  # Replace with your database name
collection = db["test1"]  # Replace with your collection name

@app.route('/api/get-data', methods=['GET'])
def get_data():
    try:
        # Fetch all data from the collection
        data = list(collection.find({}, {"_id": 0, "issuer": 1, "date": 1, "eventType": 1, "comment":1, "isComplaint":1, "borrower":1})) 
        return jsonify(
            status="success",
            data=data
        ), 200
    except Exception as e:
        return jsonify(
            status="error",
            message="Failed to fetch data from MongoDB.",
            error=str(e)
        ), 500

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        logging.error("No file uploaded")
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    pdf_file = request.files['pdf']
    try:
        reader = PdfReader(pdf_file)
        first_page = reader.pages[0]
        text = first_page.extract_text()

        # Log the extracted text
        logging.info(f"Extracted text from first page: {text[:100]}") 

        # Simulate extracted data
        extracted_data = {
            "issuer": "Issuer Name",
            "date": "2025-02-01",
            "eventType": "Sample Event",
            "comment": "This is a sample comment.",
            "isComplaint": "yes",
            "borrower": "Borrower Name"
        }

        return jsonify({
            "status": "success",
            "message": "First page read successfully.",
            "data": extracted_data
        }), 200

    except Exception as e:
        logging.error(f"Error reading PDF: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/save-data', methods=['POST'])
def save_data():
    data = request.json
    try:
        # Save to MongoDB
        db = client["covenants"]
        collection = db["test1"]
        collection.insert_one(data)
        return jsonify({"status": "success", "message": "Data saved"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
# Route to serve the React app
@app.route('/')
@app.route('/<path:path>')
def serve_react(path='index.html'):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)


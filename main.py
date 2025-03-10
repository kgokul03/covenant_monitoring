from flask import Flask , send_from_directory, jsonify, request
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import io
import logging
import PyPDF2
import re
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import torch
import ollama
import spacy
from sentence_transformers import SentenceTransformer, util

# Set up logging
logging.basicConfig(level=logging.INFO)
CHUNK_SIZE = 600
CYAN = '\033[96m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3.2:3b'
)

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient semantic embeddings

class CreditAgreementMetadata(BaseModel):
    issuer: Optional[str] = None
    administrative_agent: Optional[str] = None
    underwriter: Optional[str] = None
    agreement_date: Optional[str] = Field(None, description="Agreement date in YYYY-MM-DD format")

class CovenantDetail(BaseModel):
    title: str
    description: str

class Covenant(BaseModel):
    items: List[CovenantDetail]

def prompts_and_execution(content, isMetaData):
    if isMetaData:
        prompt = f"""
            ### System Instruction:  
            You are an AI model tasked with extracting specific financial details from documents. **Follow these instructions precisely:**  
            - Return the output in **strict JSON format** with the specified keys.  
            - If a key's information is missing, return an **empty string ("")** instead of omitting the key.  
            - Do **not** add extra text, explanations, or additional keys
            - **Don't need your <think> tags in the answer**
            
            ### **Extraction Task:**  
            Extract the following details from the provided document:  
            
            - **issuer**: The entity issuing the agreement.  
            - **administrative_agent**: The administrative agent managing the agreement.  
            - **underwriter**: The underwriter, bookrunner, or lead arranger.  
            - **agreement_date**: The agreement date in **YYYY-MM-DD** format.  
            
            ### **Document Content:**  
            {content}
            """
    else:
        prompt = f"""
            ### System Instruction:  
            You are an helpful AI assistant tasked with extracting specific financial details from documents. **Follow these instructions precisely:**  
            - Return the output in **strict JSON format** with the **specified keys** as mentioned in extraction template 
            - No Extra information is needed 
            - no triple single quotes needed in output to represent json output 
            - add to list only if you have data for title and description. Don't add if you have empty description from the document
            
            ### **Extraction Task:**  
            Extract the list of covenants, terms & conditions and details from the provided document with the list of covenant type as key named 'title' and its summary/description of the respective covenant type as value named 'description' in around 30 words.
            
            [
                {{
                    "title": "<covenant type 1>" **In 2-5 word**
                    "description": "<description of covenant type 1>" **In 30 words**
                }},
                {{
                    "title": "<covenant type 2>" **In 2-5 words**
                    "description": "<description of covenant type 2>" **In 30 words**
                }},
                {{
                    "title": "<covenant type 3>" **In 2-5 words**
                    "description": "<description of covenant type 3>" **In 30 words**
                }},
                ... **TO BE CONTINUED**
            ]

            ### **Document Content:**  
            {content}
            """
    # Send the structured prompt to Ollama
    response = client.chat.completions.create(
        model="llama3.2:3b",  # Ensure this matches your model name
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract the text response from the model
    response_text = response.choices[0].message.content
    return response_text

def is_legal_structure(text):
    """Check if text matches the legal-style structure of Articles and Sections."""
    pattern = r"(Article\s+[IVXLCDM]+\s+.*\d+|Section\s+\d+\.\d+\s+.*\d+)"
    return re.search(pattern, text) is not None
    
def clean_text(text):
    # Remove URLs and SEC archive references
    text = re.sub(r'https?://\S+|sec\.gov/Archives/\S+', '', text)
    # Remove metadata like EX-10.1 and timestamps
    text = re.sub(r'EX-\d+\.\d+.*', '', text)
    text = re.sub(r'\d+/\d+/\d+, \d+:\d+ \w+', '', text)
    # Remove trailing page numbers (e.g., 1/274)
    text = re.sub(r'\b\d+/\d+\b', '', text)

    if is_legal_structure(text):
        return ""

    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def getDataFromPdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text_list = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        text_list.append(clean_text(page_text))
    return " ".join(text_list)  # Join all pages with a space

def getMetaDataFromPdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = pdf_reader.pages[0].extract_text() if pdf_reader.pages else None
    return clean_text(text)

def prepareChunks(text, chunk_size=CHUNK_SIZE, output_file="vault.txt", similarity_threshold=0.6):
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Use spaCy for sentence segmentation
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_chunk_embedding = None  # Store embeddings for semantic similarity checks

    for sentence in sentences:
        sentence_embedding = embedder.encode(sentence, convert_to_tensor=True)

        # If adding a sentence exceeds chunk size OR it's semantically different → Start a new chunk
        if (sum(len(s) for s in current_chunk) + len(sentence) + 1 > chunk_size or
                (current_chunk_embedding is not None and 
                 util.pytorch_cos_sim(current_chunk_embedding, sentence_embedding).item() < similarity_threshold)):
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_embedding = sentence_embedding
        else:
            current_chunk.append(sentence)
            # Update chunk embedding as average of existing embeddings
            current_chunk_embedding = sentence_embedding if current_chunk_embedding is None else \
                                      (current_chunk_embedding + sentence_embedding) / 2

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Save chunks to a file
    with open(output_file, "w", encoding="utf-8") as vault_file:
        vault_file.write("\n\n".join(chunks))  # Efficient file writing

    print(f"PDF content stored in {output_file} with semantic chunking.")
    
    return chunks

def prepareEmbeddings(chunks):
    print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
    vault_embeddings = []
    for chunk in chunks:
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
            embedding = response.get("embedding")
            if embedding:  # Ensure embedding is not None
                vault_embeddings.append(embedding)
            else:
                print(f"Skipping invalid embedding for content: {chunk.strip()}")
        except Exception as e:
            print(f"Failed to generate embedding for content: {chunk.strip()}. Error: {e}")

    if not vault_embeddings:
        print("No valid embeddings generated. Exiting...")
        exit(1)
    return vault_embeddings

def convertEmbeddingsToTensor(vault_embeddings):
    # Ensure all embeddings have the same size
    embedding_size = len(vault_embeddings[0])
    if any(len(e) != embedding_size for e in vault_embeddings):
        print("Embedding size mismatch detected. Skipping invalid embeddings...")
        vault_embeddings = [e for e in vault_embeddings if len(e) == embedding_size]

    # Convert to tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    print("Embeddings for each line in the vault:")
    print(vault_embeddings_tensor)
    return vault_embeddings_tensor

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=15):
    if not rewritten_input.strip():
        print("Rewritten input is empty. Skipping context retrieval.")
        return []

    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        print("Vault embeddings are empty. Skipping context retrieval.")
        return []

    try:
        input_embedding = ollama.embeddings(model='nomic-embed-text', prompt=rewritten_input)["embedding"]
    except Exception as e:
        print(f"Failed to generate input embedding. Error: {e}")
        return []

    if not input_embedding:
        print("Input embedding is invalid. Skipping context retrieval.")
        return []

    # Compute cosine similarity
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)

    # Adjust top_k if needed
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    # Retrieve relevant context
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def getRelevantEmbeddings(query, embeddings, content):
    relevant_context = get_relevant_context(query, embeddings, content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    return prompts_and_execution(context_str, False)

def get_covenant_metadata(pdf_path):
    first_page_content = getMetaDataFromPdf(pdf_path)
    json_response = prompts_and_execution(first_page_content, True)

    try:
        data = json.loads(json_response)  # Convert JSON string to Python dictionary
        date_obj = None
        # Normalize keys to match the Pydantic model
        mapped_details = {
            "issuer": data.get("issuer"),
            "administrative_agent": data.get("administrative_agent"),
            "underwriter": data.get("underwriter"),
            "agreement_date": data.get("agreement_date")
        }
    except Exception as exp:
        print("Error in parsing the json response: {json_response} Error: {exp}")
        return CreditAgreementMetadata()
    return CreditAgreementMetadata(**mapped_details)

def get_covenant_type_and_details(pdf_path):
    query='Extract all the covenant details, its terms and agreements defined in the document'
    text_data = getDataFromPdf(pdf_path)
    chunks = prepareChunks(text_data)
    embeddings = prepareEmbeddings(chunks)
    tensor_embedding = convertEmbeddingsToTensor(embeddings)
    response = getRelevantEmbeddings(query, tensor_embedding, chunks)
    response = response.replace("'''", "")
    parsed_response = Covenant(items=[])
    try:
        json_data = json.loads(response)  # Convert text to JSON
        parsed_response = Covenant(items=json_data)  # Validate with Pydantic
    except json.JSONDecodeError:
        print("Error: Model did not return valid JSON. Raw output:")
        print(response)
    except Exception as e:
        print(f"Failed to parse response: {e}")
    
    return parsed_response

def prepare_final_json(metadata: CreditAgreementMetadata, covenant: Covenant) -> List[dict]:
    result = []
    # for item in covenant.items:
    #     result.append({
    #         "issuer": metadata.issuer,
    #         "administrative_agent": metadata.administrative_agent,
    #         "underwriter": metadata.underwriter,
    #         "agreement_date": metadata.agreement_date,
    #         "title": item.title,
    #         "description": item.description
    #     })
    for item in covenant.items:
        result.append({
            "issuer": metadata.issuer,
            "borrower": metadata.administrative_agent,
            "date": metadata.agreement_date,
            "eventType": item.title,
            "comment": item.description,
            "isComplaint": "true"
        })
    print(result)
    return result


app = Flask(__name__,static_folder=r'.\build')
# MongoDB connection string
uri = "mongodb+srv://admin:rootpassword@gokul.3p5ox.mongodb.net/?retryWrites=true&w=majority&appName=gokul"
mongoclient = MongoClient(uri, server_api=ServerApi('1'))

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    try:
        mongoclient.admin.command('ping')
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
db = mongoclient["covenants"]  # Replace with your database name
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
    metadata = get_covenant_metadata(pdf_file)
    covenant_data = get_covenant_type_and_details(pdf_file)
    final_data = prepare_final_json(metadata, covenant_data)
    print(final_data)

    try:
        # metadata = get_covenant_metadata(pdf_file)
        # covenant_data = get_covenant_type_and_details(pdf_file)
        # final_data = prepare_final_json(metadata, covenant_data)

        return jsonify({
            "status": "success",
            "message": "First page read successfully.",
            "data": final_data
        }), 200

    except Exception as e:
        logging.error(f"Error reading PDF: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/save-data', methods=['POST'])
def save_data():
    data = request.json
    if not isinstance(data, list) or len(data) == 0: 
        return jsonify({"status": "error", "message":"invalid or empty datalist"}),400

    try:
        # Save to MongoDB
        db = mongoclient["covenants"]
        collection = db["test1"]
        collection.insert_many(data)
        return jsonify({"status": "success", "message": f"{len(data)} records saved"}), 200
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


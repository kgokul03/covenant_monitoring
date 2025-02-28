import PyPDF2
import re
import torch
import ollama
from openai import OpenAI

CHUNK_SIZE = 2000

CYAN = '\033[96m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def getDataFromPdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num_pages = len(pdf_reader.pages)
    text = ''
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        if page.extract_text():
            text += page.extract_text() + " "
    return text

def prepareChunks(text):
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into chunks by sentences, respecting a maximum chunk size
    sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check if the current sentence plus the current chunk exceeds the limit
        if len(current_chunk) + len(sentence) + 1 < CHUNK_SIZE:  # +1 for the space
            current_chunk += (sentence + " ").strip()
        else:
            # When the chunk exceeds 1000 characters, store it and start a new one
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:  # Don't forget the last chunk!
        chunks.append(current_chunk)
    with open("vault.txt", "w", encoding="utf-8") as vault_file:
        for chunk in chunks:
            # Write each chunk to its own line
            vault_file.write(chunk.strip() + "\n\n")  # Two newlines to separate chunks
    print(f"PDF content stored to vault.txt with each chunk on a separate line.")
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

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=5):
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
    
    user_input_with_context = query
    if relevant_context:
        user_input_with_context = query + "\n\nRelevant Context:\n" + context_str

    conversation_history = []
    conversation_history.append({"role": "user", "content": user_input_with_context})


    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."
    messages = [
        {"role": "system", "content": system_message}, 
        *conversation_history
    ]

    response = client.chat.completions.create(
        model='gemma:2b',
        messages=messages,
        max_tokens=3000,
    )
    print(response.choices[0].message.content)

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='gemma:2b'
)

text_data = getDataFromPdf("./Example_2.pdf")
chunks = prepareChunks(text_data)
embeddings = prepareEmbeddings(chunks)
tensor_embedding = convertEmbeddingsToTensor(embeddings)
getRelevantEmbeddings("List all the financial covenants its terms and conditions and aggrements with the tracking deadline if any and also get the issuer and borrower of the aggrement", tensor_embedding, chunks)
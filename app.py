import os
import zipfile
import numpy as np
import onnxruntime as ort
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer

app = FastAPI()

# Google Drive file IDs
MODEL_FILE_ID = "1s_9yJpwR1o-j6IzTt6J5hwV6yQBt0Y6B"
TOKENIZER_ZIP_ID = "1-t_JaB7dJiV-dzgv-QCmaiY7ojZSL_3B"

# File paths
MODEL_PATH = "t5_model.onnx"
TOKENIZER_ZIP_PATH = "t5_chatbot.zip"
TOKENIZER_DIR = "t5_chatbot"

# Ensure the directory exists before extraction
if not os.path.exists(TOKENIZER_DIR):
    os.makedirs(TOKENIZER_DIR)

# Function to download files safely
def download_file(file_id: str, output_path: str):
    """Downloads a file from Google Drive if it does not exist."""
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Failed to download {output_path}")
    else:
        print(f"{output_path} already exists, skipping download.")

# Function to extract ZIP files
def extract_zip(zip_path, extract_to):
    """Extracts a zip file and ensures correct directory structure."""
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)  # Remove zip after extraction
    else:
        print(f"Zip file {zip_path} not found, skipping extraction.")

# Download and extract model and tokenizer
download_file(MODEL_FILE_ID, MODEL_PATH)
download_file(TOKENIZER_ZIP_ID, TOKENIZER_ZIP_PATH)
extract_zip(TOKENIZER_ZIP_PATH, TOKENIZER_DIR)

# Load model and tokenizer
session = ort.InferenceSession(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_DIR)

# Request model
class ChatRequest(BaseModel):
    question: str

@app.post("/chat/")
def chat_with_bot(request: ChatRequest):
    inputs = tokenizer(request.question, return_tensors="np", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)  
    max_output_length = 50
    generated_tokens = []

    for _ in range(max_output_length):
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }
        onnx_outputs = session.run(None, onnx_inputs)
        logits = onnx_outputs[0]

        next_token_id = np.argmax(logits[:, -1, :], axis=-1)
        generated_tokens.append(next_token_id.item())

        if next_token_id == tokenizer.eos_token_id:
            break  

        decoder_input_ids = np.hstack((decoder_input_ids, np.array([[next_token_id.item()]], dtype=np.int64)))

    decoded_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {"response": decoded_response}
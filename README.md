# Tag Team Image Analysis API 🤖👁️🧠

A lightweight, asynchronous FastAPI service that acts as a homework helper and customer support agent using a "Tag Team" of AI models via the Hugging Face Inference API.

## 🌟 Features

- **Tag Team Architecture**:
  - **The Eye**: `microsoft/Florence-2-large` for Image-to-Text describing.
  - **The Brain**: `Qwen/Qwen2.5-3B-Instruct` for logical reasoning and answering.
- **FastAPI**: Asynchronous endpoints for high performance.
- **Ready for Deployment**: Configured for easy hosting on Railway.

## 🛠️ Tech Stack

- **Framework**: Python 3.9+, FastAPI
- **AI/ML**: Hugging Face Inference API (`huggingface_hub`)
- **Environment**: `python-dotenv`

## 🚀 Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/MARSWALLET/AITEST.git
   cd AITEST
   ```

2. **Create a virtual environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your **Hugging Face API Key** (`HF_API_KEY`). You can get one for free [here](https://huggingface.co/settings/tokens).

5. **Run the Server**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.
   API Documentation (Swagger UI) is at `http://127.0.0.1:8000/docs`.

## 🚄 Deploy to Railway

This project includes a `Procfile` for seamless deployment on Railway.

1. **Sign up/Login** to [Railway](https://railway.app/).
2. **New Project** -> **Deploy from GitHub repo**.
3. Select this repository: `MARSWALLET/AITEST`.
4. **Important**: Go to the **Variables** tab in your Railway service and add:
   - `HF_API_KEY`: Your Hugging Face API Token.
5. Railway will automatically detect the `Procfile` and start the service.

## 📡 API Usage

**Endpoint**: `POST /analyze`

**Request**: `multipart/form-data`
- `file`: The image file to analyze.

**Response**:
```json
{
  "vision_output": "Description of the image...",
  "final_answer": "The answered query based on the image content."
}
```

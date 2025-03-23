# DocuBrain - File Analysis

**DocuBrain** is a web application that allows users to upload various file types (e.g., text, image, video, audio, PDF, tabular) and ask questions related to their contents. The system processes the uploaded files, extracts relevant information, and generates answers based on the provided query.

## Features

- Upload files of various types (e.g., `.txt`, `.pdf`, `.jpg`, `.mp4`, `.csv`, etc.).
- Ask a question related to the contents of the uploaded file.
- The system processes the file, extracts relevant information, and provides an answer to the query.
- Supports multiple file types:
  - **Text**: `.txt`, `.md`, `.json`
  - **Image**: `.jpg`, `.jpeg`, `.png`, `.gif`
  - **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`
  - **Audio**: `.mp3`, `.wav`, `.ogg`, `.flac`
  - **PDF**: `.pdf`
  - **Tabular**: `.csv`, `.xlsx`, `.xls`
- File type-specific analysis, including visual summaries for images and videos, transcription for audio, and content extraction for text and PDF files.
- A responsive front-end built using Bootstrap for a smooth user experience.

## Requirements

To run the application locally, ensure you have the following Python dependencies:

```bash
flask
werkzeug
openai
python-dotenv
pandas
numpy
PyPDF2
opencv-python
moviepy
llama-index-core
```

You can install these dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rajveersinghcse/DocuBrain.git
   cd DocuBrain
   ```

2. **Setup environment variables**:

   Create a `.env` file in the root directory and add your **OpenAI API key**:

   ```env
   OPENAI_API_KEY=your-openai-api-key
   ```

3. **Run the application**:

   Start the Flask application:

   ```bash
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:5000/`.

## How It Works

1. **File Upload**: Upload a file by selecting it from your local system using the file input.
2. **Query Input**: Enter a query related to the contents of the file (e.g., "What is the summary of this document?").
3. **File Processing**: The file is processed based on its type:
   - **Images**: The system extracts key details and generates a visual summary.
   - **Videos**: Frames are sampled and a visual summary is generated along with a transcript.
   - **Audio**: The audio is transcribed, and a response is generated based on the transcription.
   - **Text/PDF**: The text content is processed and relevant details are returned.
   - **Tabular Files**: Information such as row count, columns, and basic statistics are analyzed.
4. **Results**: The results are displayed, including the answer to the query, relevant file details, and a summary if applicable.

## Application Structure

- **`index.html`**: The main front-end HTML template for the file upload and query input form.
- **`style.css`**: Custom styles for the user interface, including responsiveness and design tweaks.
- **`main.js`**: Handles the client-side functionality for file upload, form submission, and result display.
- **`app.py`**: The Flask server that handles routing, file upload, and interaction with the `DocuBrain` system for file processing.
- **`rag_system.py`**: The core logic for processing different file types, querying, and interacting with the OpenAI API to generate answers.
- **`requirements.txt`**: A list of dependencies for the project.

## Contributions

Feel free to fork the repository, open issues, and submit pull requests. Contributions are welcome!

## License

This project is open-source and available under the [MIT License](LICENSE).
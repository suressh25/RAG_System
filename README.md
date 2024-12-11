# Document Assistant

## Description
Document Assistant is a Streamlit application that allows users to upload various document types (PDF, Word, Excel, PowerPoint, images, and text files) and interact with the content through a question-and-answer interface. The application utilizes Langchain for processing and retrieving information from the uploaded documents.

## Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your web browser and navigate to `http://localhost:8501`.
3. Upload your document using the file uploader.
4. Enter your question in the input field and receive answers based on the content of the uploaded document.

## Requirements
- Python 3.x
- Streamlit
- Langchain
- ChromaDB
- Other dependencies listed in `requirements.txt`

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Sentiment Analyzer

## Project Overview
This project implements advanced sentiment classification using BERT (Bidirectional Encoder Representations from Transformers). The aim is to classify text as either positive or negative sentiment, leveraging Python, PyTorch, and HuggingFace’s Transformers library. The project also includes a FastAPI server for handling requests and a Google Chrome extension for user interaction.

## Project Stack
- **Programming Languages**: Python, JavaScript, HTML, CSS
- **Libraries and Frameworks**:
  - **Python**: PyTorch, Transformers, FastAPI
  - **JavaScript**: Bootstrap
- **Tools**: Google Chrome Extension API

## Key Features
- **Leveraged Pre-trained BERT Models**: Employed pre-trained BERT models and fine-tuned them on a sentiment analysis dataset to enhance model accuracy and relevance for sentiment classification.
- **Developed Custom Neural Network Architecture**: Engineered a custom neural network that incorporates BERT model's embeddings, enhancing the model's ability to understand and process contextual information.
  - Fully connected (dense) layers on top of BERT's outputs to facilitate effective sentiment classification.
  - Dropout layers to mitigate overfitting, ensuring robust performance across diverse datasets.
  - ReLU (Rectified Linear Unit) activation functions to introduce non-linearity, improving the model's learning capabilities.
- **Fine-tuned BERT for Sentiment Analysis**: Trained the model using a sentiment classification dataset, optimizing the pre-trained weights to improve performance on sentiment-specific tasks.
  - Advanced optimization techniques, including AdamW optimizer and cross-entropy loss function, were used to refine the model’s predictive accuracy.
- **Developed a FastAPI Server**: Created a FastAPI server in Python to handle real-time requests to the sentiment analysis model, facilitating the interpretation of sentences and providing immediate sentiment classification responses.
- **Engineered a Google Chrome Extension**: Developed a user-friendly Chrome extension using JavaScript, HTML, and Bootstrap.
  - Enabled users to highlight any text within the browser, with the extension dynamically sending the highlighted text to the server for sentiment analysis.
  - Implemented a feature where the extension highlights text green for positive sentiment and red for negative sentiment, maintaining the highlight for 5 seconds to provide clear visual feedback.

## Detailed Steps

### Data Preprocessing
- Used `BertTokenizerFast` from the Transformers library for tokenization.
- Converted raw text into tokens suitable for input to the BERT model.
- Managed padding and truncation to ensure uniform input size.

### Model Architecture
- Created a custom neural network architecture incorporating a pre-trained BERT model.
- Added fully connected layers on top of BERT's outputs to perform classification.
- Included dropout layers to prevent overfitting and ReLU activation for non-linearity.

### Training the Model
- Fine-tuned the BERT model on a sentiment classification dataset, adjusting the pre-trained weights to better fit the sentiment analysis task.
- Employed DataLoader from PyTorch for efficient mini-batch loading.
- Split data into training and validation sets to monitor and prevent overfitting.
- Implemented training loops to fine-tune the model, and evaluation loops to assess model performance on validation data.

### Saving and Loading the Model
- Saved the fine-tuned model's state dictionary using `torch.save`.
- Provided functionality to load the model for future inference tasks using `torch.load`.

### Inference
- Developed a prediction function to classify new sentences as positive or negative.
- Used softmax activation to convert model outputs into probability distributions over classes.
- Implemented logic to interpret and return the sentiment classification based on model predictions.

### Chrome Extension Integration
- **Content Script**: Captures highlighted text and displays a tooltip. On clicking the tooltip button, sends the text to the server and handles the response to highlight the text accordingly.
- **Bootstrap Styling**: Utilizes Bootstrap for styling the tooltip and button, ensuring a responsive and visually appealing user interface.
- **Highlighting Logic**: Highlights the selected text green for positive sentiment and red for negative sentiment, maintaining the highlight for 5 seconds.

## Conclusion
This project showcases the integration of cutting-edge NLP techniques with practical web technologies, demonstrating the power of BERT in real-world applications and providing a seamless user experience from model inference to user interaction.

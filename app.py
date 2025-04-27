import streamlit as st
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model #, QLoRAConfig

# Hugging Face API token (optional, for accessing private models or higher rate limits)
api_token = st.text_input("Enter your Hugging Face API Token (optional)", type="password")
headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

# Function to get models from Hugging Face API
def fetch_huggingface_models(task=None, limit=10):
    url = "https://huggingface.co/api/models"
    params = {"pipeline_tag": task, "limit": limit} if task else {"limit": limit}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return [model["modelId"] for model in response.json()]
    else:
        st.error("Failed to fetch models. Check your API token or try again later.")
        return []

# Allow user to select the model task/type
task = st.selectbox("Select task type", ["text-classification", "text-generation", "summarization", "question-answering", None])

# Fetch and display models
models = fetch_huggingface_models(task=task, limit=20)
model_name = st.selectbox("Choose a model from Hugging Face", models)

# Option to select LoRA or QLoRA
lora_qlora_option = st.selectbox("Select Fine-tuning Technique", ["None", "LoRA", "QLoRA"])

# Load and Fine-tune the selected model with LoRA/QLoRA
if model_name:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply LoRA or QLoRA if selected
    if lora_qlora_option == "LoRA":
        config = LoraConfig(
            r=8,   # Rank of the low-rank matrices
            lora_alpha=16,  # Scaling factor for the low-rank matrices
            lora_dropout=0.1,  # Dropout rate
            task_type="SEQ_CLASSIFICATION"  # Specify the task type (adjust if necessary)
        )
        model = get_peft_model(model, config)
        st.write(f"Using LoRA for fine-tuning with rank {config.r}")
    # elif lora_qlora_option == "QLoRA":
    #     config = QLoRAConfig(
    #         r=8,   # Rank of the low-rank matrices
    #         lora_alpha=16,  # Scaling factor for the low-rank matrices
    #         lora_dropout=0.1,  # Dropout rate
    #         task_type="SEQ_CLASSIFICATION"  # Specify the task type (adjust if necessary)
    #     )
        model = get_peft_model(model, config)
        st.write(f"Using QLoRA for fine-tuning with rank {config.r} and quantization")
    
    st.write(f"Selected model: {model_name}")
    
    # Training parameters
    epochs = st.slider("Epochs", 1, 5, 3)
    batch_size = st.slider("Batch Size", 4, 32, 16)
    learning_rate = st.slider("Learning Rate", 1e-5, 5e-5, 3e-5)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
    )
    
    # Assuming you have your dataset ready (replace with actual dataset logic)
    train_dataset = None
    eval_dataset = None
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    st.success("Fine-tuning completed!")

import os
import glob
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Step 1: Read the directories from the file
with open("data_dirs_train.txt", "r") as f:
    directories = [line.strip() for line in f if line.strip()]

# Step 2: Collect all JSONL files from these directories
all_files = []
for directory in directories:
    # Adjust the pattern if needed
    files = glob.glob(os.path.join(directory, "*.jsonl"))
    all_files.extend(files)

if not all_files:
    raise ValueError("No JSONL files found in the provided directories.")

# Step 3: Load the combined dataset
# This will create a "train" split combining all files
dataset = load_dataset("json", data_files={"train": all_files})

# Optional: Inspect a sample entry
print("Sample entry:", dataset["train"][0])

# Step 4: Load the tokenizer and model using deepseek-r1
# You can change "deepseek-r1:1.5b" to "deepseek-r1:3b" if desired.
model_name = "deepseek-r1:1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 5: Preprocess the dataset
# This function creates prompts by combining a natural language instruction (from "docstring")
# with the expected code output (from "code").
def preprocess_function(examples):
    inputs = [f"Instruction: {doc}\nCode:" for doc in examples["docstring"]]
    # Concatenate the instruction with the corresponding code
    combined = [inp + code for inp, code in zip(inputs, examples["code"])]
    model_inputs = tokenizer(combined, truncation=True, padding="max_length", max_length=512)
    return model_inputs

# Tokenize the training dataset
tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# Step 6: Prepare a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_code_gen",
    evaluation_strategy="no",  # Change if you have a validation set
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=100,
    report_to="none",  # Disable reporting to W&B or similar services
)

# Step 8: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 9: Fine-tune the model
trainer.train()

# Optional: Save the fine-tuned model and tokenizer
model.save_pretrained("./finetuned_code_gen")
tokenizer.save_pretrained("./finetuned_code_gen")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Custom CSS for a dark theme
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("GoLang Code Generation Assistant")
st.caption("Translate natural language descriptions into efficient and idiomatic Go code snippets.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant", 
        "content": "Hi! I'm your GoLang code generation assistant. Describe what you need, and I'll generate the corresponding Go code snippet."
    }]

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "assistant":
        st.markdown(f"**Assistant:** {message['content']}")
    else:
        st.markdown(f"**User:** {message['content']}")

# Load the fine-tuned model and tokenizer from the saved directory.
model_dir = "./finetuned_code_gen"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Create a text-generation pipeline with the fine-tuned model
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# User input box
user_input = st.text_input("Enter your instruction for Go code:")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Construct a prompt focused on generating Go code.
    prompt = (
        "You are a highly experienced GoLang developer and code generator. "
        "Your task is to produce efficient, idiomatic Go code that accurately fulfills the following description. "
        "Provide clear, well-commented code and include any necessary explanations within the code comments.\n\n"
        f"Description: {user_input}\n\nGo Code:\n"
    )
    
    # Generate the code snippet using the pipeline.
    with st.spinner("Generating Go code snippet..."):
        outputs = generator(prompt, max_length=512, do_sample=True, temperature=0.3)
        # Extract the generated text after the prompt
        generated_text = outputs[0]['generated_text'][len(prompt):].strip()
    
    # Append the generated code to the chat history.
    st.session_state.chat_history.append({"role": "assistant", "content": generated_text})
    
    # Rerun to update the chat display.
    st.experimental_rerun()

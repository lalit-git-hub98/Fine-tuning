import streamlit as st
import requests

from params_pg import *

if "page" not in st.session_state:
    st.session_state.page = "main"

# Function to navigate between pages
def navigate_to(page_name):
    st.session_state.page = page_name

def main_page():
    model_type_choose_list = ['Choose Model from the Application', 'Give Model Name']
    model_type_choose_selected = st.radio('Model Select', model_type_choose_list)


    if model_type_choose_selected == 'Choose Model from the Application':
        api_token = st.text_input("Enter your Hugging Face API Token (optional)", type="password")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}



        # Function to fetch model details from Hugging Face Hub
        def fetch_model_list(task_type=None, limit=10):
            params = {
                "limit": limit,
                "sort": sort_by_selected,  # You can sort by downloads, likes, or updated
            }
            if task_type:
                params["pipeline_tag"] = task_type  # Filter models by task, e.g., "text-generation"

            response = requests.get("https://huggingface.co/api/models", headers=headers, params=params)
            if response.status_code == 200:
                #st.json(response.json())
                return response.json()
            else:
                st.error("Failed to fetch models.")
                return []

        # Function to display model details and descriptions in Streamlit
        def display_model_previews(models):
            for model in models:
                model_id = model.get("modelId", "Unknown Model")
                description = model.get("description", "No description available")
                downloads = model.get("downloads", "Unknown")
                likes = model.get("likes", "Unknown")

                with st.expander(f"Model: {model_id}"):
                    st.write(f"**Description**: {description}")
                    st.write(f"**Downloads**: {downloads}")
                    st.write(f"**Likes**: {likes}")
                    if st.button("Select Model", key=model_id):
                        model_name = model_id
                        return model_name

        # Streamlit app
        st.title("Model Previews and Descriptions")
        st.subheader("Choose a model to fine-tune")

        model_select_col1, model_select_col2, model_select_col3 = st.columns(3)

        with model_select_col1:
            sort_by = ["downloads", "likes", "createdAt"]
            sort_by_selected = st.selectbox("Sort By", sort_by, placeholder="downloads")

        with model_select_col2:
            limit_set = st.number_input("Select number of models to show", min_value = 5, max_value=20)

        with model_select_col3:
            task_type_list = ["text-classification", "text-generation", "summarization", "question-answering"]
            task_type_selected = st.selectbox("Task Type", task_type_list)


        # Fetch and display models
        models = fetch_model_list(task_type=task_type_selected, limit=limit_set)


        if models:
            model_name = display_model_previews(models)
        else:
            st.write("No models found for the selected task type.")

    else:
        model_name = st.text_input('Enter Model Name')

        if model_name:
            def check_model_exists(model_name):
                url = f"https://huggingface.co/api/models/{model_name}"
                response = requests.get(url)
                return response.status_code == 200 
            
            model_name_check = False
            
            if check_model_exists(model_name):
                st.success(f"The model '{model_name}' exists on Hugging Face!")
                # Add a timer here to wait 2 seconds
                model_name_check = True
            else:
                st.error(f"The model '{model_name}' does not exist on Hugging Face.")

            if model_name_check:
                pass



# Display pages based on the current session state
if st.session_state.page == "main":
    main_page()

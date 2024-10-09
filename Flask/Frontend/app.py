import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os 
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Load environment variables from .env file
load_dotenv()

# Get the backend URL from environment variables
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')  # Default to local if not set

st.title("Chat Messages Analysis Web Application")

# sidebar menu for navigation
with st.sidebar:
    selected = option_menu(
        'Analyzer',
        ['Home', 'Analyze Languages', 'Analyze User Intents', 'Analyse Toxicity Results'],
        icons=['house', 'translate', 'clipboard-check', 'bar-chart'],
        menu_icon='chart-bar',
        default_index=0
    )

# Home Page , Upload Dataset
if selected == 'Home':
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json", "parquet"])

    if uploaded_file is not None:
        # Saving the file to session state
        st.session_state['uploaded_file'] = uploaded_file

        # Uploading the file to the backend
        with st.spinner("Uploading file..."):
            files = {'file': uploaded_file}
            response = requests.post(f"{backend_url}/upload", files=files)

        if response.status_code == 200:
            st.success("File uploaded successfully!")
            st.toast("✅ File uploaded successfully!") 
        else:
            st.error("File upload failed.")
            st.toast("❌ File upload failed.")  # Display a failure notification
            st.json(response.json())  # Show detailed response in case of failure

    # Display the "Perform Basic EDA" button only if a dataset is in the session state
    if 'uploaded_file' in st.session_state:
        # EDA button 
        if st.button('Perform EDA'):
            with st.spinner('Performing EDA...'):
                #  Request to the backend to perform EDA
                eda_response = requests.get(f"{backend_url}/eda")

                if eda_response.status_code == 200:
                    eda_data = eda_response.json()

                    # Display EDA results 
                    st.subheader("Exploratory Data Analysis (EDA)")

                    # Display rows and columns count with formatting
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", eda_data['rows'])
                    with col2:
                        st.metric("Total Columns", eda_data['columns'])

                    # Display the columns list in a well-formatted manner
                    st.write("### Columns List:")
                    st.write(eda_data['columns_list'])

                    # Display detailed information 
                    st.write("### DataFrame Information:")
                    st.text_area("Info", value=str(eda_data['info']), height=150)

                    # If you want to display a DataFrame in a table format
                    # Example: Display a sample of the dataset (if available)
                    if 'sample' in eda_data:
                        sample_data = pd.DataFrame(eda_data['sample'])  # Ensure you have 'sample' data in your response
                        st.write("### Sample Data:")
                        st.dataframe(sample_data)

                else:
                    st.error("Failed to perform EDA.")
                    st.json(eda_response.json())  # Print error details
    else:
        st.warning("Please upload a dataset to enable EDA.")

# Analyze Languages Page
if selected == 'Analyze Languages':
    st.subheader("Languages Distribution Analysis")

    # Check if a dataset has been uploaded
    if 'uploaded_file' in st.session_state:
        # Display the button only if the file is uploaded
        if st.button("Perform Analysis"):
            with st.spinner("Performing language analysis..."):
                response = requests.get(f"{backend_url}/languages_count")

                if response.status_code == 200:
                    try:
                        # JSON response containing languages count
                        analysis_info = response.json()

                        #  The title for the language distribution
                        st.write("### Language Tables")

                        # Convert the language counts to a DataFrame
                        languages_data = pd.DataFrame(list(analysis_info['languages_count'].items()), columns=['Language', 'Count'])

                        # Convert the DataFrame to a horizontal format
                        languages_data_horizontal = languages_data.set_index('Language').T

                    
                        st.write(languages_data_horizontal)

                        # Plot a professional-looking bar chart with Plotly
                        fig = px.bar(
                            languages_data, 
                            x='Language', 
                            y='Count', 
                            title='Language Distribution', 
                            labels={'Count': 'Number of Messages', 'Language': 'Language'},
                            color='Count',  # Add color based on count
                            height=500,     # Adjust the height of the chart
                            template='plotly_dark'  # Use a sleek dark theme
                        )

                        # Update layout for better readability
                        fig.update_layout(
                            xaxis_title="Language",
                            yaxis_title="Number of Messages",
                            font=dict(size=12),
                            title_font=dict(size=20),
                            xaxis_tickangle=-45  # Rotate x-axis labels for better readability
                        )

                        # Display the chart in Streamlit
                        st.plotly_chart(fig)

                    except ValueError:
                        st.error("Received unexpected response format. Could not parse JSON.")
                        st.write(response.text)  # Print raw response for debugging

                elif response.status_code == 400:
                    st.error("Failed to perform analysis.")
                    st.toast("❌ Check the column names of the dataset! Make sure 'lang' column exists.")  # Display a failure notification
                    st.write(response.json())  # Show detailed response in case of failure

                else:
                    st.error("Failed to perform analysis.")
                    st.write(response.text)  # Print raw response for debugging
    else:
        st.warning("Please upload a dataset to enable analysis.")  # Message when no dataset is uploaded

# Analyze User Intents Page
if selected == 'Analyze User Intents':
    st.subheader("User Intent Analysis")

    if 'uploaded_file' in st.session_state:
        # Display the button only if the file is uploaded
        if st.button("Perform User Intent Analysis"):
            with st.spinner("Performing user intent analysis..."):
                response = requests.get(f"{backend_url}/userintents")

                if response.status_code == 200:
                    try:
                        # the JSON response containing user intent counts
                        analysis_info = response.json()

                        # The title for the user intent distribution
                        st.write("### User Intent Distribution")

                        # Convert the user intent counts to a DataFrame
                        user_intents_data = pd.DataFrame(list(analysis_info['user_intents_count'].items()), columns=['Category', 'Count'])

                        # Print the user intent distribution as a DataFrame (tabular format)
                        user_intents_data_horizontal = user_intents_data.set_index('Category').T
                        st.write(user_intents_data_horizontal)

                        # Plot a professional-looking bar chart with Plotly
                        fig = px.bar(
                            user_intents_data, 
                            x='Category', 
                            y='Count', 
                            title='User Intent Distribution', 
                            labels={'Count': 'Number of Messages', 'Category': 'User Intent'},
                            color='Count',  # Add color based on count
                            height=500,     # Adjust the height of the chart
                            template='plotly_dark'  # Use a sleek dark theme
                        )

                        # Update layout for better readability
                        fig.update_layout(
                            xaxis_title="User Intent",
                            yaxis_title="Number of Messages",
                            font=dict(size=12),
                            title_font=dict(size=20),
                            xaxis_tickangle=-45  # Rotate x-axis labels for better readability
                        )

                       
                        st.plotly_chart(fig)

                    except ValueError:
                        st.error("Received unexpected response format. Could not parse JSON.")
                        st.write(response.text)  # Print raw response for debugging

                elif response.status_code == 400:
                    st.error("Failed to perform analysis.")
                    st.toast("❌ Check the dataset! Make sure it contains the required columns.")
                    st.write(response.json())  # Show detailed response in case of failure

                else:
                    st.error("Failed to perform analysis.")
                    st.write(response.text)  # Print raw response for debugging
    else:
        st.warning("Please upload a dataset to enable analysis.")  # Message when no dataset is uploaded

# Visualize Results Page
if selected == 'Analyse Toxicity Results':
    # Display Toxicity Visualization
  if 'uploaded_file' in st.session_state:
    if st.button("Visualize Toxicity"):
        with st.spinner("Fetching toxicity data..."):
            # Fetch toxicity data from the backend API
            response = requests.get(f"{backend_url}/toxicity")

            if response.status_code == 200:
                toxicity_data = response.json()
                toxicity_scores = toxicity_data.get('toxicity_scores', [])

                if len(toxicity_scores) == 0:
                    st.warning("No toxicity scores available in the dataset.")
                else:
                    # Create a DataFrame from toxicity scores for visualization
                    df = pd.DataFrame({'Toxicity Score': toxicity_scores})

                    st.header('Toxicity Score Visualization')

                    # Bar Chart of Toxicity Scores using Plotly Express
                    st.subheader('Bar Chart of Toxicity Scores')
                    bar_fig = px.bar(
                        df,
                        x=df.index,
                        y='Toxicity Score',
                        labels={'x': 'Sample Index', 'Toxicity Score': 'Toxicity Score'},
                        title="Bar Chart of Toxicity Scores"
                    )
                    st.plotly_chart(bar_fig)

                    # Histogram and KDE of Toxicity Scores using Plotly Express
                    st.subheader('Histogram and KDE of Toxicity Scores')
                    hist_fig = px.histogram(
                        df,
                        x='Toxicity Score',
                        nbins=5,
                        marginal="box",  # Adds a box plot above the histogram
                        title="Histogram and KDE of Toxicity Scores"
                    )
                    st.plotly_chart(hist_fig)

                # Display success message
                st.success("Toxicity data visualization completed!")
            else:
                st.error("Error fetching toxicity data")
  else:
        st.warning("Please upload a dataset to enable analysis.")  # Message when no dataset is uploaded


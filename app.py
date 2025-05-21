import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
import io

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better formatting
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: none;
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    .stChatMessage [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.5rem;
    }
    .stChatMessage [data-testid="stChatMessageContent"] pre {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stChatMessage [data-testid="stChatMessageContent"] code {
        background-color: #f0f2f6;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def plot_and_display_chart(df, prompt):
    """Function to create and display charts based on the data and user query"""
    try:
        # Set the style for seaborn
        sns.set_style("whitegrid")

        # Create a figure with a larger size
        plt.figure(figsize=(12, 6))

        # Get column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Determine plot type based on user query
        prompt_lower = prompt.lower()

        if "bar" in prompt_lower or "barplot" in prompt_lower:
            if len(categorical_cols) > 0:
                if len(numeric_cols) > 0:
                    sns.barplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
                    plt.title(f'Bar Plot of {numeric_cols[0]} by {categorical_cols[0]}')
                else:
                    sns.countplot(x=categorical_cols[0], data=df)
                    plt.title(f'Count of {categorical_cols[0]}')
            else:
                st.warning("No categorical columns found for bar plot")
                return

        elif "pie" in prompt_lower or "pie chart" in prompt_lower:
            if len(categorical_cols) > 0:
                if len(numeric_cols) > 0:
                    values = df.groupby(categorical_cols[0])[numeric_cols[0]].sum()
                else:
                    values = df[categorical_cols[0]].value_counts()

                plt.pie(values, labels=values.index, autopct='%1.1f%%')
                plt.title(f'Pie Chart of {categorical_cols[0]}')
            else:
                st.warning("No categorical columns found for pie chart")
                return

        elif "line" in prompt_lower or "lineplot" in prompt_lower:
            if len(numeric_cols) >= 2:
                sns.lineplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f'Line Plot of {numeric_cols[1]} vs {numeric_cols[0]}')
            else:
                st.warning("Need at least two numeric columns for line plot")
                return

        elif "scatter" in prompt_lower or "scatterplot" in prompt_lower:
            if len(numeric_cols) >= 2:
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f'Scatter Plot of {numeric_cols[1]} vs {numeric_cols[0]}')
            else:
                st.warning("Need at least two numeric columns for scatter plot")
                return

        elif "hist" in prompt_lower or "histogram" in prompt_lower:
            if len(numeric_cols) > 0:
                sns.histplot(data=df, x=numeric_cols[0], kde=True)
                plt.title(f'Distribution of {numeric_cols[0]}')
            else:
                st.warning("No numeric columns found for histogram")
                return

        elif "box" in prompt_lower or "boxplot" in prompt_lower:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                sns.boxplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
                plt.title(f'Box Plot of {numeric_cols[0]} by {categorical_cols[0]}')
            else:
                st.warning("Need both numeric and categorical columns for box plot")
                return

        elif "heatmap" in prompt_lower or "correlation" in prompt_lower:
            if len(numeric_cols) >= 2:
                correlation = df[numeric_cols].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            else:
                st.warning("Need at least two numeric columns for correlation heatmap")
                return

        else:
            if len(numeric_cols) >= 2:
                correlation = df[numeric_cols].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            else:
                st.warning("No specific plot type requested and insufficient data for default plot")
                return

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    except Exception as e:
        st.error(f"Error creating the plot: {str(e)}")

def format_response(response):
    """Format the response to be more readable"""
    try:
        if isinstance(response, pd.DataFrame):
            return response
        elif isinstance(response, (list, tuple)):
            return "\n".join([f"- {item}" for item in response])
        elif isinstance(response, dict):
            return "\n".join([f"**{key}**: {value}" for key, value in response.items()])
        elif response is None:
            return "No results found."
        else:
            response_str = str(response)
            response_str = response_str.replace("'", "").replace('"', '')
            return response_str
    except Exception as e:
        return f"Error formatting response: {str(e)}"

def chat_with_csv(df, prompt):
    """Function to handle chat with CSV using LangChain"""
    try:
        # Create a file-like object from the DataFrame
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-4"),
            csv_buffer,
            verbose=True,
            allow_dangerous_code=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        # Check if the prompt is about plotting
        if any(word in prompt.lower() for word in ["plot", "graph", "chart", "visualize", "show"]):
            plot_and_display_chart(df, prompt)

        # Check if the prompt is about top transactions and display the table and download button
        if "top" in prompt.lower() and "transaction" in prompt.lower():
            try:
                # Ensure 'transaction_amount' column exists before sorting
                if 'transaction_amount' in df.columns:
                    # Sort by transaction amount and get top 10
                    top_transactions = df.sort_values('transaction_amount', ascending=False).head(10)

                    # Display the table
                    st.markdown("### Top 10 Transactions")
                    st.dataframe(
                        top_transactions,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Add download button for the table
                    csv = top_transactions.to_csv(index=False)
                    st.download_button(
                        label="Download Top 10 Transactions",
                        data=csv,
                        file_name="top_10_transactions.csv",
                        mime="text/csv"
                    )
                    # Return a placeholder or None to prevent the default chat response from being displayed
                    return "Top 10 transactions displayed above."
                else:
                     st.warning("Could not find 'transaction_amount' column to display top transactions.")
                     return "Could not find 'transaction_amount' column to display top transactions."
            except Exception as e:
                st.error(f"Error displaying top transactions: {str(e)}")
                return f"Error displaying top transactions: {str(e)}"

        # If not a top transaction query, proceed with standard chat response
        result = agent.run(prompt)

        # Check if the prompt is about plotting (this was already here, keeping it)
        if any(word in prompt.lower() for word in ["plot", "graph", "chart", "visualize", "show"]):
            plot_and_display_chart(df, prompt)

        return format_response(result)
    except Exception as e:
        return f"Error processing query: {str(e)}"


def main():
    # Sidebar for file upload and API key
    with st.sidebar:
        st.header("⚙️ Settings")
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        if not openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key to use the chatbot.")
            st.info("You can get an API key from: https://platform.openai.com/api-keys")
            st.stop()
        else:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Calculate transaction amount if not present
                if 'transaction_amount' not in df.columns and 'item_price' in df.columns and 'quantity' in df.columns:
                    df['transaction_amount'] = df['item_price'] * df['quantity']

                st.session_state['df'] = df
                st.success(f"Successfully uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # Title and description with image
    st.markdown('<h1><img src="https://static.wixstatic.com/media/97b8c3_d0a1a2e3860e436fbc5712b8c33c65f9~mv2.gif" width="60" height="55" style="vertical-align: middle;"> Data Analysis Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about your data and get instant insights!")

    # Main content area
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Display data preview at the top using expander
        with st.expander("📋 Click to view Data Preview", expanded=True):
            st.dataframe(df, use_container_width=True)

        # Add a separator
        st.markdown("---")

        # Chat interface below
        st.subheader("💬 Chat with your Data")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], pd.DataFrame):
                    st.dataframe(message["content"], use_container_width=True)
                else:
                    st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    try:
                        response = chat_with_csv(df, prompt)
                        if isinstance(response, pd.DataFrame):
                            st.dataframe(response, use_container_width=True)
                        else:
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("👈 Please upload a CSV file using the sidebar to begin analysis.")

if __name__ == "__main__":
    main()

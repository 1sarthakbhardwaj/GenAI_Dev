import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Function to load the external CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state variables."""
    if 'groq_api_key' not in st.session_state:
        st.session_state['groq_api_key'] = ''
        st.session_state['authenticated'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferWindowMemory(k=5)

def main():
    # Set up page configuration
    st.set_page_config(page_title="Groq LLM SaaS Playground", page_icon="🤖", layout="wide")

    # Load external CSS
    load_css("styles.css")

    st.title("🤖 Groq LLM SaaS Playground")
    st.sidebar.title("⚙️ Application Settings")
    st.sidebar.markdown("Interact with open-source LLMs using Groq API.")

    initialize_session()

    # Step 1: Prompt for the API Key
    if not st.session_state['authenticated']:
        with st.expander("🔑 Enter Your GROQ API Key to Unlock", expanded=True):
            groq_api_key = st.text_input("GROQ API Key", type="password")
            if st.button("Unlock Application"):
                if groq_api_key:
                    st.session_state['groq_api_key'] = groq_api_key
                    st.session_state['authenticated'] = True
                    st.success("API Key authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key.")
        return

    # Step 2: Main Chatbot Interface
    st.sidebar.subheader("🤖 Chat Settings")

    # Model selection
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama3-8b-8192', 'gemma-7b-it', 'gemma2-9b-it',
         'llama3-groq-70b-8192-tool-use-preview', 'llava-v1.5-7b-4096-preview']
    )

    # Conversational memory length selection
    conversational_memory_length = st.sidebar.slider(
        'Conversational Memory Length', 1, 10, value=5
    )
    
    # Update the memory buffer window length
    st.session_state['memory'].k = conversational_memory_length

    st.markdown("## 💬 Chat with the Model")
    user_question = st.text_area("📝 Ask a question:", placeholder="Type your query here...")

    # Initialize ChatGroq instance
    try:
        groq_chat = ChatGroq(
            model_name=model,
            groq_api_key=st.session_state['groq_api_key']
        )
        conversation = ConversationChain(
            llm=groq_chat,
            memory=st.session_state['memory']
        )

        # Handle user question and display response
        if user_question:
            with st.spinner("Thinking..."):
                # Use memory to retain conversation context
                response = conversation.run(user_question)
                message = {"human": user_question, "AI": response}
                
                # Update chat history and memory
                st.session_state['chat_history'].append(message)
                st.session_state['memory'].save_context(
                    {"input": user_question},
                    {"output": response}
                )
            
            st.write("🤖 ChatBot:", response)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Display chat history
    st.markdown("### 📜 Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_history']:
            st.markdown(f"**👤 You**: {message['human']}", unsafe_allow_html=True)
            st.markdown(f"**🤖 AI**: {message['AI']}", unsafe_allow_html=True)

    # Sidebar Buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.session_state['memory'] = ConversationBufferWindowMemory(k=conversational_memory_length)
        st.rerun()

    if st.sidebar.button("Log Out"):
        st.session_state['authenticated'] = False
        st.session_state['groq_api_key'] = ''
        st.session_state['chat_history'] = []
        st.session_state['memory'] = ConversationBufferWindowMemory(k=conversational_memory_length)
        st.rerun()

if __name__ == "__main__":
    main()

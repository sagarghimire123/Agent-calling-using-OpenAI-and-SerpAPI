import streamlit as st
from langchain.agents import AgentType, load_tools, initialize_agent
from langchain_openai import OpenAI

# Initialize OpenAI client with API key
openai_api_key = "------------------------------------"
client = OpenAI(openai_api_key=openai_api_key)

# Initialize tools
serpapi_key = "--------------------"  # Replace with your SerpApi key
serpapi_tool = load_tools(["serpapi"], serpapi_api_key=serpapi_key, llm=client)
serpapi_agent = initialize_agent(serpapi_tool, client, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

wikipedia_tool = load_tools(["wikipedia"], llm=client)
wikipedia_agent = initialize_agent(wikipedia_tool, client, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def main():
    st.title("Query Interface")

    # User input
    query = st.text_input("Enter your query:")

    if query:
        source = st.radio("Choose source:", ("SerpApi", "Wikipedia"))
        
        if st.button("Submit"):
            if source == "SerpApi":
                result = serpapi_agent.run(query)
            elif source == "Wikipedia":
                result = wikipedia_agent.run(query)
            else:
                result = "Invalid source selected"

            st.subheader("Query Result")
            st.write(f"**Query:** {query}")
            st.write(f"**Result:**")
            st.write(result)

if __name__ == '__main__':
    main()


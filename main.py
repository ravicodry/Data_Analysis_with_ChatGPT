import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import matplotlib
import pandas as pd
<<<<<<< HEAD
=======

>>>>>>> df8a3d2f309ede61afe66e591f18c08e4346ab83
def main():
    matplotlib.use('TkAgg')
    load_dotenv()

    st.set_page_config(page_title="CSV App", page_icon=":chart:", layout="centered")
    st.header("Data analysis with csv")

    file=st.file_uploader("Upload a csv file", type="csv")
    
    if file is not None:
        df=pd.read_csv(file)
        st.write(df.head())
        user_question=st.text_input("What is your question?")
        llm=OpenAI(temperature=0)
        pandas_ai=PandasAI(llm)
        #agent=create_csv_agent(llm,file,verbose=True)

        if user_question is not None and user_question != "":
            #st.write(agent.run(user_question))
            st.write(pandas_ai.run(df,user_question,show_code=True))
        


if __name__ == "__main__":
    main()

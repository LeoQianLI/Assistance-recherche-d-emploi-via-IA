import sys
import os
import time
import pandas as pd
import tempfile
import streamlit as st
from fpdf import FPDF
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, MDXSearchTool, SerperDevTool, ScrapeWebsiteTool
from tools.qdrant_vector_search_tool import QdrantVectorSearchTool
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from models_st import JobRequirements, ResumeOptimization, CompanyResearch
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Initialisation des outils de recherche vectorielle et du modèle
tool = QdrantVectorSearchTool()
model = SentenceTransformer("sentence-transformers/roberta-large-nli-stsb-mean-tokens")
MAX_RESULTS = 5

def main():
    st.title("💼 Assistance à la Recherche d'Emploi ")

    # Upload du fichier PDF et extraction du texte
    uploaded_file = st.file_uploader("Téléchargez votre CV (PDF)", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extraction du texte du PDF..."):
            reader = PdfReader(uploaded_file)
            resume_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    resume_text += page_text
            if resume_text:
                st.success("Texte extrait avec succès !!")
                st.session_state.resume_text = resume_text
                if resume_text:
                    st.session_state.resume_text = resume_text

                    with st.spinner("Recherche de descriptions de poste similaires..."):
                        time.sleep(2)  # Adjust delay as needed
                        results = tool.search(resume_text)
                    st.success("Recherche terminée !")
                    if results:
                        st.subheader("Résultats de la recherche: ")
                        results_df = pd.DataFrame(results)
                        results_df['score'] = results_df['score'].apply(lambda x: round(x, 2))
                        results_df.reset_index(drop=True, inplace=True)
                        st.dataframe(results_df[['poste', 'company_name', 'score']])

                        # Let the user select a job to view details
                        options = [""] + results_df['poste'].tolist()
                        selected_job_poste = st.selectbox("Sélectionnez un poste pour voir les détails :", options)
                        if selected_job_poste:
                            selected_job = next((job for job in results if job['poste'] == selected_job_poste), None)
                            if selected_job:
                                #display_job_details(selected_job)
                                st.session_state.selected_job = selected_job
                                st.success("Pacientez pendant que nous générons le rapport...")
                                  #create a button to view the job description
                                if st.button('Job selectionné'):
                                     st.session_state['job'] = selected_job['poste']
                                     st.switch_page("pages/Job_description.py")
                    else:
                        st.write("Aucune description de poste similaire trouvée.")
            else:
                st.error("Aucun texte n'a pu être extrait du PDF.")

    if "resume_text" in st.session_state:
        resume_text = st.session_state.resume_text
        # Créer un fichier temporaire pour le contenu du CV
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        ) as temp_file:
            temp_file.write(resume_text)
            temp_file_path = temp_file.name
      # Initialiser les outils avec le fichier temporaire
        #read_resume = FileReadTool(file_path=temp_file_path)
        #sematic_search_resume = MDXSearchTool(mdx=temp_file_path)


if __name__ == "__main__":
    main()
import streamlit as st
import time
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

if 'job' not in st.session_state:
    st.session_state['job'] = 'No job selected'

col1, col2, col3 = st.columns([1, 6, 3])  # Adjusted column widths for bigger margins
with col2:
    st.title(st.session_state['job'])
    # Retrieve the selected job from the session state
    selected_job = st.session_state['selected_job']
    # Add CSS and HTML for job details
    st.markdown(
        """
        <style>
        .job-details {
            background-color: #f0f0f0;
            padding: 10px;
            width: 180%;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .job-section {
            margin-bottom: 20px;
        }
        .job-section h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.2rem;  /* Increased font size */
        }
        .job-section p {
            font-size: 1.2rem;  /* Increased font size */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="job-details">
            <div class="job-section">
                <h3>Informations sur le poste</h3>
                <p><strong>🟢 Job:</strong> {selected_job['poste']} at {selected_job['company_name']}</p>
                <p><strong>📍 Address:</strong> {selected_job['local_address']}</p>
                <p><strong>📄 Contract Type:</strong> {selected_job['contract_type']}</p>
                <p><strong>🎁 Benefits:</strong> {selected_job['benefits']}</p>
                <p><strong>🛠️ Skills:</strong> {selected_job['skills']}</p>
                <p><strong>🔧 Tools:</strong> {selected_job['tools']}</p>
                <p><strong>🌐 Remote:</strong> {selected_job['remote']}</p>
                <p><strong>🔗 Job URL:</strong> <a href="{selected_job['url']}" target="_blank">Lien vers l'annonce</a></p>
            </div>
            <div class="job-section">
                <h3>🏫Description de l'entreprise</h3>
                <p>{selected_job['company_description']}</p>
            </div>
            <div class="job-section">
                <h3>🎯Profile</h3>
                <p>{selected_job['profile']}</p>
            </div>
            <div class="job-section">
                <h3>👨‍💻Processus de recrutement</h3>
                <p>{selected_job['recruitment_process']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

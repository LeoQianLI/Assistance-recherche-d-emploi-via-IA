import tempfile
import streamlit as st
import os
import time
from crewai_tools import FileReadTool, MDXSearchTool, SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from models_st import JobRequirements, ResumeOptimization, CompanyResearch
from litellm.exceptions import RateLimitError

def get_llm(model_name, system_prompt=None):
    """Get LLM instance with fallback options"""
    try:
        # Try to get model from environment variable
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        return LLM(model=model, system_prompt=system_prompt)
    except Exception as e:
        st.warning(f"Failed to initialize {model_name}, trying fallback models...")
        # Try different fallback models in order of preference
        fallback_models = [
            "gpt-3.5-turbo",  # OpenAI (gratuit avec limite)
            "claude-2",       # Anthropic (gratuit avec limite)
            "mistral/mistral-7b-instruct",  # Mistral (gratuit)
            "gemini/gemini-1.0-pro"  # Google (gratuit avec limite)
        ]
        
        for fallback_model in fallback_models:
            try:
                st.info(f"Attempting to use {fallback_model} as fallback...")
                return LLM(model=fallback_model, system_prompt=system_prompt)
            except Exception as inner_e:
                st.warning(f"Failed to initialize {fallback_model}: {str(inner_e)}")
                continue
        
        # If all fallbacks fail, raise the original error
        raise e

def retry_with_backoff(func, max_retries=3, initial_delay=5):
    """Retry a function with exponential backoff and longer delays"""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            st.warning(f"Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            if delay > 30:  # Cap the maximum delay at 30 seconds
                delay = 30

# create a liens related to the resume_text in the file crewai_st.py

if "resume_text" in st.session_state and st.session_state.resume_text:
    resume_text = st.session_state['selected_job']
    resume_t = st.session_state.resume_text
    # Create a temporary file for CV content
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    ) as temp_file:
        temp_file.write(resume_t)
        temp_file_path = temp_file.name

    # Initialize tools with temporary file
    read_resume = FileReadTool(file_path=temp_file_path)
    sematic_search_resume = MDXSearchTool(mdx=temp_file_path)

    @CrewBase
    class Gojob:
        """Gojob crew"""

        @agent
        def resume_analyzer(self) -> Agent:
            return Agent(
                verbose=True,
                groq_llm=get_llm("resume_analyzer", "Répondez uniquement en français."),
                tools=[read_resume, sematic_search_resume],
                role="Expert en optimisation de CV",
                goal="Analyser les CV et fournir des suggestions d'optimisation structurées en français",
                backstory="""Vous êtes un spécialiste de l'optimisation de CV avec une connaissance approfondie des systèmes ATS
et des meilleures pratiques modernes en matière de CV. Vous excellez dans l'analyse des CV PDF et
la fourniture de suggestions d'amélioration concrètes. Vos recommandations se concentrent toujours
sur la lisibilité humaine et la compatibilité ATS.🟢 **TOUTES VOS RÉPONSES DOIVENT ÊTRE EN FRANÇAIS.**""",
            )

        @agent
        def job_analyzer(self) -> Agent:
            return Agent(
                verbose=True,
                tools=[ScrapeWebsiteTool(),read_resume, sematic_search_resume],
                groq_llm=get_llm("job_analyzer"),
                role="Analyste des exigences de poste",
                goal="Analyser les descriptions de poste et évaluer l'adéquation des candidats en français",
                backstory="""Vous êtes un expert en analyse du marché du travail et en évaluation des candidats. Votre force
réside dans la décomposition des exigences de poste en catégories claires et la fourniture
d'une évaluation détaillée basée sur les qualifications des candidats. Vous comprenez à la fois les
compétences techniques et générales, et pouvez évaluer avec précision les niveaux d'expérience.🟢 **TOUTES VOS RÉPONSES DOIVENT ÊTRE EN FRANÇAIS.**""",
            )

        @agent
        def company_researcher(self) -> Agent:
            return Agent(
                verbose=True,
                tools=[SerperDevTool(), read_resume, sematic_search_resume],
                groq_llm=get_llm("company_researcher"),
                role="Spécialiste en intelligence d'entreprise",
                goal="Rechercher des informations sur les entreprises et préparer des insights pour les entretiens en français",
                backstory="""Vous êtes un expert en recherche d'entreprise qui excelle dans la collecte et l'analyse
des dernières informations sur les entreprises. Vous savez comment trouver et synthétiser des données
provenant de diverses sources pour créer des profils d'entreprise complets et préparer
les candidats aux entretiens.🟢 **TOUTES VOS RÉPONSES DOIVENT ÊTRE EN FRANÇAIS.**""",
            )

        @agent
        def resume_writer(self) -> Agent:
            return Agent(
                verbose=True,
                llm=get_llm("resume_writer"),
                tools=[SerperDevTool(), read_resume, sematic_search_resume],
                role="Spécialiste en rédaction de CV en markdown",
                goal="Créer des CV formatés en markdown, optimisés pour les ATS en français",
                backstory="""Vous êtes un expert en rédaction de CV spécialisé dans la création de CV formatés en markdown.
                Vous savez comment transformer des suggestions d'optimisation structurées en documents
                formatés en markdown, optimisés pour les ATS, qui maintiennent le professionnalisme
                tout en mettant en valeur les points forts des candidats.🟢 **TOUTES VOS RÉPONSES DOIVENT ÊTRE EN FRANÇAIS.**""",
            )

        @agent
        def report_generator(self) -> Agent:
            return Agent(
                verbose=True,
                llm=get_llm("report_generator"),
                tools=[SerperDevTool(), read_resume, sematic_search_resume],
                role="Générateur de rapports de carrière et spécialiste du markdown",
                goal="Créer des rapports complets, visuellement attrayants et exploitables à partir de l'analyse des candidatures en français",
                backstory="""Vous êtes un expert en visualisation de données, rédaction technique et formatage markdown.
                Vous excellez dans la combinaison de données provenant de multiples sources JSON pour créer des rapports
                cohérents et visuellement attrayants. Votre spécialité est de transformer des analyses structurées
                en insights clairs et exploitables avec un formatage markdown approprié, des emojis et
                des éléments visuels qui rendent l'information à la fois attrayante et facilement digestible.🟢 **TOUTES VOS RÉPONSES DOIVENT ÊTRE EN FRANÇAIS.**""",
            )

        @task
        def analyze_job_task(self) -> Task:
            return Task(
                agent=self.job_analyzer(),
                output_file="output/job_analysis.json",
                output_pydantic=JobRequirements,
                description="""Analyser la description du poste et évaluer l'adéquation du candidat en fonction de son CV.
                                Sortie en JSON structuré.
                                Ajouter des emojis et des éléments visuels pour améliorer la lisibilité.
                                Si la longueur de la phrase atteint 94 caractères, veuillez passer la ligne suivante.
                                1. Extraire les exigences : compétences techniques, générales, expérience, éducation, connaissance du secteur.
                                2. Évaluer les compétences techniques et générales.
                                3. Évaluer l'expérience et l'éducation.
                                4. Calculer le score global.""",
                expected_output="Données JSON structurées contenant l'analyse du poste et les détails de l'évaluation.",
            )

        @task
        def optimize_resume_task(self) -> Task:
            return Task(
                agent=self.resume_analyzer(),
                output_file="output/resume_optimization.json",
                output_pydantic=ResumeOptimization,
                description="""Examiner le {read_resume} uploaded en fonction de l'analyse du poste et créer des suggestions d'optimisation structurées.
                                Sortie en JSON structuré.
                                Ajouter des emojis et des éléments visuels pour améliorer la lisibilité.
                                Si la longueur de la phrase atteint 94 caractères, veuillez passer la ligne suivante.
                                1. Analyser le contenu et la structure du CV.
                                2. Générer des suggestions d'amélioration.
                                3. LE CV DOIT ÊTRE GÉNÉRÉ EN FRANÇAIS SEULEMENT.""",
                expected_output="Données JSON structurées contenant des suggestions d'optimisation détaillées.",
                context= [self.analyze_job_task()]
            )

        @task
        def research_company_task(self) -> Task:
            return Task(
                agent=self.company_researcher(),
                output_file="output/company_research.json",
                output_pydantic=CompanyResearch,
                description="""Rechercher des informations sur l'entreprise et préparer une analyse complète.
                            Sortie en JSON structuré.
                            Ajouter des emojis et des éléments visuels pour améliorer la lisibilité.
                            Si la longueur de la phrase atteint 94 caractères, veuillez passer la ligne suivante.
                            1. Présentation de l'entreprise : développements récents, culture, position sur le marché.
                            2. Préparation à l'entretien : questions courantes, sujets spécifiques, projets récents.""",
                expected_output="Données JSON structurées contenant les résultats de la recherche sur l'entreprise.",
            )

        @task
        def generate_resume_task(self) -> Task:
            return Task(
                agent=self.resume_writer(),
                output_file="output/optimized_resume.md",
                description=f"""Utiliser le contenu du CV original et les suggestions d'optimisation pour créer un CV optimisé en markdown.
                                Contenu du CV original:
                                {resume_t}
                                
                                Instructions:
                                1. Intégrer le contenu original du CV
                                2. Appliquer les suggestions d'optimisation
                                3. Formater en markdown
                                4. Ajouter des emojis et des éléments visuels
                                5. LE CV DOIT ÊTRE GÉNÉRÉ EN FRANÇAIS SEULEMENT
                                6. Ne pas ajouter de blocs de code markdown comme '```'
                                7. Si la longueur de la phrase atteint 94 caractères, passer à la ligne suivante""",
                expected_output="Un document de CV au format markdown bien présenté, incorporant le contenu original et les suggestions d'optimisation.",
                context= [self.optimize_resume_task(), self.analyze_job_task(), self.research_company_task()]
            )

        @task
        def generate_report_task(self) -> Task:
            return Task(
                agent=self.report_generator(),
                output_file="output/final_report.md",
                description="""Créer un rapport de synthèse exécutif en utilisant les données des étapes précédentes.
                            Formater en markdown sans blocs de code '```'.
                            Ajouter des emojis et des éléments visuels pour améliorer la lisibilité.
                            Si la longueur de la phrase atteint 94 caractères, veuillez passer la ligne suivante.
                            1. Intégrer les données : analyse du poste, optimisation du CV, insights sur l'entreprise.
                            2. Structurer le rapport : résumé exécutif, analyse de l'adéquation, aperçu de l'optimisation, insights sur l'entreprise, prochaines étapes.
                            3. LE final_report DOIT ÊTRE GÉNÉRÉ EN FRANÇAIS SEULEMENT.""",
                expected_output=""" Un rapport markdown complet combinant toutes les analyses en un document clair et exploitable.""",
                context= [self.analyze_job_task(), self.optimize_resume_task(), self.research_company_task()]
            )

        @crew
        def crew(self) -> Crew:
            agents_instances = [
                self.resume_analyzer(),
                self.job_analyzer(),
                self.company_researcher(),
                self.resume_writer(),
                self.report_generator(),
            ]
            tasks_instances = [
                self.analyze_job_task(),
                self.optimize_resume_task(),
                self.research_company_task(),
                self.generate_resume_task(),
                self.generate_report_task(),
            ]
            return Crew(
                agents=agents_instances,
                tasks=tasks_instances,
                verbose=True,
                process=Process.sequential,
                tools=[read_resume, sematic_search_resume],
            )

    try:
        # Execute the crew with retry logic
        gojob = Gojob()
        crew_instance = gojob.crew()
        
        def execute_crew():
            return crew_instance.kickoff()
        
        # Execute with retry logic and longer initial delay
        result = retry_with_backoff(execute_crew, max_retries=3, initial_delay=5)

        # Display the final report and optimized resume
        report_file = r"C:\Users\leo12\Documents\Projet3\Assistance-recherche-d-emploi-via-IA\src\gojob\output\final_report.md"
        optimized_resume_file = r"C:\Users\leo12\Documents\Projet3\Assistance-recherche-d-emploi-via-IA\src\gojob\output\optimized_resume.md"
        
        if os.path.exists(report_file) and os.path.exists(optimized_resume_file):
            with open(report_file, "r", encoding="utf-8") as file:
                final_report = file.read()
            with open(optimized_resume_file, "r", encoding="utf-8") as file:
                optimized_resume = file.read()
            st.markdown(final_report, unsafe_allow_html=True)
            st.markdown(optimized_resume, unsafe_allow_html=True)
            st.success("Rapport généré avec succès!")
        else:
            st.error(f"Le fichier de rapport {report_file} n'existe pas.")
            
    except RateLimitError as e:
        st.error("""Nous avons atteint la limite de requêtes pour le moment. 
                   Le système va essayer d'utiliser un autre modèle d'IA.
                   Si le problème persiste, veuillez réessayer dans quelques minutes.
                   Conseil: Attendez environ 5 minutes avant de réessayer.""")
        st.error(f"Erreur détaillée: {str(e)}")
        # Try to reinitialize with a different model
        try:
            st.info("Tentative de réinitialisation avec un modèle alternatif...")
            gojob = Gojob()
            crew_instance = gojob.crew()
            result = crew_instance.kickoff()
            st.success("Opération réussie avec le modèle alternatif!")
        except Exception as fallback_error:
            st.error("Échec de l'utilisation du modèle alternatif. Veuillez réessayer plus tard.")
            st.error(f"Erreur détaillée: {str(fallback_error)}")
    except Exception as e:
        st.error("Une erreur inattendue s'est produite. Veuillez réessayer.")
        st.error(f"Erreur détaillée: {str(e)}")


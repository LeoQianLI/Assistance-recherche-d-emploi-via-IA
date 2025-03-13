import tempfile
import streamlit as st
import os
from crewai_tools import FileReadTool, MDXSearchTool, SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from models_st import JobRequirements, ResumeOptimization, CompanyResearch

# create a liens related to the resume_text in the file crewai_st.py

if "resume_text" in st.session_state and st.session_state.resume_text:
    resume_text = st.session_state['selected_job']
    resume_t = st.session_state.resume_text
    # Créer un fichier temporaire pour le contenu du CV
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    ) as temp_file:
        temp_file.write(resume_t)
        temp_file_path = temp_file.name

    # Initialiser les outils avec le fichier temporaire
    read_resume = FileReadTool(file_path=temp_file_path)
    sematic_search_resume = MDXSearchTool(mdx=temp_file_path)

    @CrewBase
    class Gojob:
        """Gojob crew"""

        @agent
        def resume_analyzer(self) -> Agent:
            return Agent(
                # config=self.agents_config["resume_analyzer"],
                verbose=True,
                groq_llm=LLM(model="gemini/gemini-2.0-flash-exp", system_prompt="Répondez uniquement en français."),
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
                # config=self.agents_config["job_analyzer"],
                verbose=True,
                tools=[ScrapeWebsiteTool(),read_resume, sematic_search_resume],
                groq_llm=LLM(model="gemini/gemini-2.0-flash-exp"),
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
                # config=self.agents_config["company_researcher"],
                verbose=True,
                tools=[SerperDevTool(), read_resume, sematic_search_resume],
                groq_llm=LLM(model="gemini/gemini-2.0-flash-exp"),
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
                # config=self.agents_config["resume_writer"],
                verbose=True,
                llm=LLM(model="gemini/gemini-2.0-flash-exp"),
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
                # config=self.agents_config["report_generator"],
                verbose=True,
                llm=LLM(model="gemini/gemini-2.0-flash-exp"),
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
                # config=self.tasks_config["analyze_job_task"],
                agent=self.job_analyzer(),
                output_file="output/job_analysis.json",
                output_pydantic=JobRequirements,
                # description=self.tasks_config["analyze_job_task"]["description"],
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
                # config=self.tasks_config["optimize_resume_task"],
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
                # config=self.tasks_config["research_company_task"],
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
                description="""Utiliser les suggestions d'optimisation et l'analyse du poste pour créer un CV poli au format markdown.
                                Ne pas ajouter de blocs de code markdown comme '```'.
                                Ajouter des emojis et des éléments visuels pour améliorer la lisibilité.
                                Si la longueur de la phrase atteint 94 caractères, veuillez passer la ligne suivante.
                                1. Intégrer les suggestions d'optimisation.
                                2. Formater le CV en PDF,
                                3. LE CV DOIT ÊTRE GÉNÉRÉ EN FRANÇAIS SEULEMENT.""",
                expected_output="""Un document de CV au format markdown bien présenté, incorporant toutes les suggestions d'optimisation.""",
                context= [self.optimize_resume_task(),  self.analyze_job_task(), self.research_company_task()]
)
        @task
        def generate_report_task(self) -> Task:
            return Task(
                # config=self.tasks_config["generate_report_task"],
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

    # Exécuter le crew
    gojob = Gojob()
    gojob.crew().kickoff()

    # Afficher le rapport final which is in the src/gojob/output/final_report.md
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


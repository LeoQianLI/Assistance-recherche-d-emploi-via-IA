import tempfile
import streamlit as st
import os
import time
import codecs
from crewai_tools import FileReadTool, MDXSearchTool, SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from models_st import JobRequirements, ResumeOptimization, CompanyResearch
from litellm.exceptions import RateLimitError

# Créer une classe personnalisée pour lire les fichiers avec un encodage spécifique
class CustomFileReadTool(FileReadTool):
    def _read_file(self, file_path):
        """Lit le fichier avec l'encodage UTF-8."""
        try:
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error: Failed to read file {file_path}. {str(e)}"

def get_llm(role_name):
    """Fonction pour obtenir un LLM avec repli en cas d'erreur"""
    # Liste des modèles à essayer dans l'ordre
    models_to_try = [
        "gemini/gemini-2.0-flash-exp",
        "gpt-3.5-turbo",
        "claude-2",
        "mistral/mistral-7b-instruct"
    ]
    
    # Essayer chaque modèle jusqu'à ce qu'un fonctionne
    for model in models_to_try:
        try:
            # st.info(f"Tentative d'utilisation du modèle {model} pour {role_name}...")
            return LLM(model=model)
        except Exception as e:
            st.warning(f"Échec avec le modèle {model}: {str(e)}")
            continue
    
    # Si aucun modèle ne fonctionne, lever une exception
    raise Exception("Aucun modèle LLM disponible n'a pu être initialisé.")

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
    
    # Vérifier que le fichier temporaire existe et est accessible
    if not os.path.exists(temp_file_path):
        st.error(f"Le fichier temporaire {temp_file_path} n'existe pas.")
    else:
        # st.info(f"Fichier temporaire créé avec succès: {temp_file_path}")
        # Essayer de lire le fichier pour vérifier qu'il est accessible
        try:
            with open(temp_file_path, "r", encoding="utf-8") as f:
                _ = f.read()
            # st.success("Lecture du fichier temporaire réussie.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier temporaire: {str(e)}")

    # Initialiser les outils avec le fichier temporaire et utiliser notre classe personnalisée
    read_resume = CustomFileReadTool(file_path=temp_file_path)
    sematic_search_resume = MDXSearchTool(mdx=temp_file_path)

    @CrewBase
    class Gojob:
        """Gojob crew"""

        @agent
        def resume_analyzer(self) -> Agent:
            return Agent(
                verbose=True,
                groq_llm=get_llm("resume_analyzer"),
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

    # Fonction pour tenter d'exécuter avec plusieurs tentatives
    def retry_with_backoff(func, max_retries=3, initial_delay=5):
        """Exécuter une fonction avec backoff exponentiel en cas d'erreur"""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func()
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e
                st.warning(f"Limite de requêtes atteinte. Nouvelle tentative dans {delay} secondes... (Tentative {attempt+1}/{max_retries})")
                time.sleep(delay)
            except UnicodeDecodeError as e:
                st.error(f"Erreur d'encodage: {str(e)}")
                st.info("Tentative de récupération avec un encodage différent...")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
            except Exception as e:
                st.error(f"Erreur inattendue: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 2  # Backoff exponentiel
                
    try:
        if not os.path.exists("output"):
            os.makedirs("output")
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        gojob = Gojob()
        crew_instance = gojob.crew()
        
        def execute_crew():
            return crew_instance.kickoff()
        
        result = retry_with_backoff(execute_crew, max_retries=3, initial_delay=5)

        output_base_dir = os.getcwd()
        
        possible_paths = [
            os.path.join(output_dir, "final_report.md"),
            os.path.join(output_base_dir, "output", "final_report.md"),
            "output/final_report.md",
        ]
        
        possible_resume_paths = [
            os.path.join(output_dir, "optimized_resume.md"),
            os.path.join(output_base_dir, "output", "optimized_resume.md"),
            "output/optimized_resume.md",
        ]
        
        report_file = next((path for path in possible_paths if os.path.exists(path)), None)
        optimized_resume_file = next((path for path in possible_resume_paths if os.path.exists(path)), None)
        
        # Ajouter du CSS pour améliorer la lisibilité et éviter le défilement horizontal
        st.markdown("""
        <style>
        .report-container {
            max-width: 100%;
            overflow-x: hidden;
            word-wrap: break-word;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .report-container img {
            max-width: 100%;
            height: auto;
        }
        .report-container h1, .report-container h2 {
            color: #0066cc;
        }
        .report-container p, .report-container li {
            font-size: 16px;
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if report_file and os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as file:
                    final_report = file.read()
                st.subheader("Rapport d'Analyse")
                st.markdown(f'<div class="report-container">{final_report}</div>', unsafe_allow_html=True)
                st.success("Rapport généré avec succès!")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du rapport: {str(e)}")
                
        if optimized_resume_file and os.path.exists(optimized_resume_file):
            try:
                with open(optimized_resume_file, "r", encoding="utf-8") as file:
                    optimized_resume = file.read()
                st.subheader("CV Optimisé")
                st.markdown(f'<div class="report-container">{optimized_resume}</div>', unsafe_allow_html=True)
                st.success("CV optimisé généré avec succès!")
            except Exception as e:
                st.error(f"Erreur lors de la lecture du CV optimisé: {str(e)}")
            
        if not report_file and not optimized_resume_file:
            st.error("Les fichiers de rapport n'ont pas été trouvés.")
                
    except RateLimitError as e:
        st.error("Nous avons atteint la limite de requêtes pour le moment. Le système va essayer d'utiliser un autre modèle d'IA.")
        st.error(f"Erreur détaillée: {str(e)}")
        
        try:
            st.info("Tentative d'utilisation d'un modèle alternatif...")
            os.environ["LLM_MODEL"] = "gpt-3.5-turbo"
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


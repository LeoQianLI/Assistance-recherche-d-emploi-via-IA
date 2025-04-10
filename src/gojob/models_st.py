from typing import List, Dict, Optional
from pydantic import BaseModel, Field, confloat

class SkillScore(BaseModel):
    skill_name: str = Field(description="Name of the skill being scored")
    required: bool = Field(description="Whether this skill is required or nice-to-have")
    match_level: confloat(ge=0, le=1) = Field(description="How well the candidate's experience matches (0-1)")
    years_experience: Optional[float] = Field(description="Years of experience with this skill")
    context_score: confloat(ge=0, le=1) = Field(description="How relevant the skill usage context is to the job requirements")

class JobMatchScore(BaseModel):
    overall_match: confloat(ge=0, le=100) = Field(description="Overall match percentage (0-100)")
    technical_skills_match: confloat(ge=0, le=100) = Field(description="Technical skills match percentage")
    soft_skills_match: confloat(ge=0, le=100) = Field(description="Soft skills match percentage")
    experience_match: confloat(ge=0, le=100) = Field(description="Experience level match percentage")
    education_match: confloat(ge=0, le=100) = Field(description="Education requirements match percentage")
    industry_match: confloat(ge=0, le=100) = Field(description="Industry experience match percentage")
    skill_details: List[SkillScore] = Field(description="Detailed scoring for each skill")
    strengths: List[str] = Field(description="List of areas where candidate exceeds requirements")
    gaps: List[str] = Field(description="List of areas needing improvement")
    scoring_factors: List[str] = Field(description="Weights used for different scoring components")

class JobRequirements(BaseModel):
    technical_skills: List[str] = Field(description="List of required technical skills")
    soft_skills: List[str] = Field(description="List of required soft skills")
    experience_requirements: List[str] = Field(description="List of experience requirements")
    key_responsibilities: List[str] = Field(description="List of key job responsibilities")
    education_requirements: List[str] = Field(description="List of education requirements")
    nice_to_have: List[str] = Field(description="List of preferred but not required skills")
    job_title: str = Field(description="Official job title")
    department: Optional[str] = Field(description="Department or team within the company")
    reporting_structure: Optional[str] = Field(description="Who this role reports to and any direct reports")
    job_level: Optional[str] = Field(description="Level of the position (e.g., Entry, Senior, Lead)")
    location_requirements: List[str] = Field(description="Location details including remote/hybrid options")
    work_schedule: Optional[str] = Field(description="Expected work hours and schedule flexibility")
    travel_requirements: Optional[str] = Field(description="Expected travel frequency and scope")
    compensation: List[str] = Field(description="Salary range and compensation details if provided")
    benefits: List[str] = Field(description="List of benefits and perks")
    tools_and_technologies: List[str] = Field(description="Specific tools, software, or technologies used")
    industry_knowledge: List[str] = Field(description="Required industry-specific knowledge")
    certifications_required: List[str] = Field(description="Required certifications or licenses")
    security_clearance: Optional[str] = Field(description="Required security clearance level if any")
    team_size: Optional[str] = Field(description="Size of the immediate team")
    key_projects: List[str] = Field(description="Major projects or initiatives mentioned")
    cross_functional_interactions: List[str] = Field(description="Teams or departments this role interacts with")
    career_growth: List[str] = Field(description="Career development and growth opportunities")
    training_provided: List[str] = Field(description="Training or development programs offered")
    diversity_inclusion: Optional[str] = Field(description="D&I statements or requirements")
    company_values: List[str] = Field(description="Company values mentioned in the job posting")
    job_url: str = Field(description="URL of the job posting")
    posting_date: Optional[str] = Field(description="When the job was posted")
    application_deadline: Optional[str] = Field(description="Application deadline if specified")
    special_instructions: List[str] = Field(description="Any special application instructions or requirements")
    match_score: JobMatchScore = Field(description="Detailed scoring of how well the candidate matches the job requirements")
    score_explanation: List[str] = Field(description="Detailed explanation of how scores were calculated")

class ResumeOptimization(BaseModel):
    content_suggestions: List[str] = Field(description="List of content optimization suggestions with 'before' and 'after' examples")
    skills_to_highlight: List[str] = Field(description="List of skills that should be emphasized based on job requirements")
    achievements_to_add: List[str] = Field(description="List of achievements that should be added or modified")
    keywords_for_ats: List[str] = Field(description="List of important keywords for ATS optimization")
    formatting_suggestions: List[str] = Field(description="List of formatting improvements")

class CompanyResearch(BaseModel):
    recent_developments: List[str] = Field(description="List of recent company news and developments")
    culture_and_values: List[str] = Field(description="Key points about company culture and values")
    market_position: List[str] = Field(description="Information about market position, including competitors and industry standing")
    growth_trajectory: List[str] = Field(description="Information about company's growth and future plans")
    interview_questions: List[str] = Field(description="Strategic questions to ask during the interview")
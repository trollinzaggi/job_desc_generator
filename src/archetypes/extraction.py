"""
Phase 1: LLM-based Requirement Extraction

Extracts structured requirements from JD text using Azure OpenAI.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime


# =============================================================================
# EXTRACTION SCHEMA
# =============================================================================

@dataclass
class RequirementCategory:
    """A category of requirements with required/preferred distinction."""
    required: List[str] = field(default_factory=list)
    preferred: List[str] = field(default_factory=list)


@dataclass 
class SkillsExtraction:
    """Extracted skills by type."""
    technical: RequirementCategory = field(default_factory=RequirementCategory)
    domain: RequirementCategory = field(default_factory=RequirementCategory)
    soft: RequirementCategory = field(default_factory=RequirementCategory)


@dataclass
class EducationExtraction:
    """Extracted education requirements."""
    level: RequirementCategory = field(default_factory=RequirementCategory)  # Bachelor's, Master's, etc.
    fields: List[str] = field(default_factory=list)  # Finance, Mathematics, etc.


@dataclass
class ExperienceExtraction:
    """Extracted experience requirements."""
    years_min: Optional[int] = None
    years_preferred: Optional[int] = None
    specific: RequirementCategory = field(default_factory=RequirementCategory)


@dataclass
class ExtractionResult:
    """Complete extraction result for a single JD."""
    
    jd_id: str
    
    # Requirements
    licenses: RequirementCategory = field(default_factory=RequirementCategory)
    certifications: RequirementCategory = field(default_factory=RequirementCategory)
    skills: SkillsExtraction = field(default_factory=SkillsExtraction)
    education: EducationExtraction = field(default_factory=EducationExtraction)
    experience: ExperienceExtraction = field(default_factory=ExperienceExtraction)
    tools: RequirementCategory = field(default_factory=RequirementCategory)
    languages: RequirementCategory = field(default_factory=RequirementCategory)
    
    # Metadata (passed through from source)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Extraction metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_model: str = ""
    extraction_success: bool = True
    extraction_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "skills" in data and isinstance(data["skills"], dict):
            skills_data = data["skills"]
            data["skills"] = SkillsExtraction(
                technical=RequirementCategory(**skills_data.get("technical", {})),
                domain=RequirementCategory(**skills_data.get("domain", {})),
                soft=RequirementCategory(**skills_data.get("soft", {})),
            )
        
        if "education" in data and isinstance(data["education"], dict):
            edu_data = data["education"]
            level_data = edu_data.get("level", {})
            if isinstance(level_data, dict):
                level = RequirementCategory(**level_data)
            else:
                level = RequirementCategory()
            data["education"] = EducationExtraction(
                level=level,
                fields=edu_data.get("fields", []),
            )
        
        if "experience" in data and isinstance(data["experience"], dict):
            exp_data = data["experience"]
            specific_data = exp_data.get("specific", {})
            if isinstance(specific_data, dict):
                specific = RequirementCategory(**specific_data)
            else:
                specific = RequirementCategory()
            data["experience"] = ExperienceExtraction(
                years_min=exp_data.get("years_min"),
                years_preferred=exp_data.get("years_preferred"),
                specific=specific,
            )
        
        for field_name in ["licenses", "certifications", "tools", "languages"]:
            if field_name in data and isinstance(data[field_name], dict):
                data[field_name] = RequirementCategory(**data[field_name])
        
        return cls(**data)
    
    def get_all_skills_flat(self) -> List[str]:
        """Get all skills as a flat list (for embedding)."""
        skills = []
        for category in [self.skills.technical, self.skills.domain, self.skills.soft]:
            skills.extend(category.required)
            skills.extend(category.preferred)
        return skills
    
    def get_all_requirements_text(self) -> str:
        """Get all requirements as text (for embedding)."""
        parts = []
        
        # Licenses and certs
        all_licenses = self.licenses.required + self.licenses.preferred
        if all_licenses:
            parts.append(f"Licenses: {', '.join(all_licenses)}")
        
        all_certs = self.certifications.required + self.certifications.preferred
        if all_certs:
            parts.append(f"Certifications: {', '.join(all_certs)}")
        
        # Skills
        skills = self.get_all_skills_flat()
        if skills:
            parts.append(f"Skills: {', '.join(skills)}")
        
        # Tools
        all_tools = self.tools.required + self.tools.preferred
        if all_tools:
            parts.append(f"Tools: {', '.join(all_tools)}")
        
        # Education
        all_education = self.education.level.required + self.education.level.preferred
        if all_education or self.education.fields:
            edu_parts = all_education + self.education.fields
            parts.append(f"Education: {', '.join(edu_parts)}")
        
        # Experience
        exp_parts = []
        if self.experience.years_min:
            exp_parts.append(f"{self.experience.years_min}+ years")
        all_specific = self.experience.specific.required + self.experience.specific.preferred
        exp_parts.extend(all_specific)
        if exp_parts:
            parts.append(f"Experience: {', '.join(exp_parts)}")
        
        return "; ".join(parts)


# =============================================================================
# EXTRACTION PROMPT
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert HR analyst specializing in job description analysis.
Your task is to extract structured requirements from job descriptions.

IMPORTANT GUIDELINES:
1. Distinguish between REQUIRED and PREFERRED requirements
2. Be specific - extract actual skill/tool names, not generic categories
3. For licenses/certifications, use standard abbreviations (e.g., "CFA", "Series 7")
4. For education, normalize levels to: "High School", "Associate's", "Bachelor's", "Master's", "PhD"
5. Extract years of experience as integers
6. If something is ambiguous or not mentioned, leave it empty rather than guessing

SKILL CATEGORIES:
- technical: Programming languages, software, platforms, technical methodologies
- domain: Industry-specific knowledge, business processes, regulatory knowledge
- soft: Communication, leadership, interpersonal skills

OUTPUT FORMAT: You must respond with valid JSON only, no other text."""

EXTRACTION_USER_PROMPT = """Extract structured requirements from this job description.

JOB DESCRIPTION:
{job_description}

EXPERTISE/SKILLS SECTION (if available):
{expertise}

TEAM DESCRIPTION (for context):
{team_description}

Extract and return as JSON with this exact structure:
{{
  "licenses": {{
    "required": ["license1", "license2"],
    "preferred": ["license3"]
  }},
  "certifications": {{
    "required": ["cert1"],
    "preferred": ["cert2"]
  }},
  "skills": {{
    "technical": {{
      "required": ["Python", "SQL"],
      "preferred": ["Bloomberg Terminal"]
    }},
    "domain": {{
      "required": ["derivatives pricing"],
      "preferred": ["risk management"]
    }},
    "soft": {{
      "required": ["communication"],
      "preferred": ["leadership"]
    }}
  }},
  "education": {{
    "level": {{
      "required": ["Bachelor's"],
      "preferred": ["Master's"]
    }},
    "fields": ["Finance", "Mathematics", "Economics"]
  }},
  "experience": {{
    "years_min": 3,
    "years_preferred": 5,
    "specific": {{
      "required": ["trading experience"],
      "preferred": ["client-facing"]
    }}
  }},
  "tools": {{
    "required": ["Bloomberg"],
    "preferred": ["Internal trading systems"]
  }},
  "languages": {{
    "required": ["English"],
    "preferred": ["Mandarin"]
  }}
}}

Return ONLY valid JSON, no markdown formatting or explanation."""


# =============================================================================
# REQUIREMENT EXTRACTOR
# =============================================================================

class RequirementExtractor:
    """
    Extract structured requirements from JD text using Azure OpenAI.
    """
    
    def __init__(
        self,
        deployment_name: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        model_name: str = "gpt-4o",
    ):
        """
        Initialize extractor with Azure OpenAI credentials.
        
        Args:
            deployment_name: Azure OpenAI deployment name (or set AZURE_OPENAI_CHAT_DEPLOYMENT)
            azure_endpoint: Azure endpoint (or set AZURE_OPENAI_ENDPOINT)
            api_key: API key (or set AZURE_OPENAI_API_KEY)
            api_version: API version
            model_name: Model name for tracking
        """
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.model_name = model_name
        self._client = None
        
        if not self.deployment_name:
            raise ValueError(
                "Deployment name required. Pass deployment_name or set AZURE_OPENAI_CHAT_DEPLOYMENT env var."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Endpoint required. Pass azure_endpoint or set AZURE_OPENAI_ENDPOINT env var."
            )
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key or set AZURE_OPENAI_API_KEY env var."
            )
    
    def _get_client(self):
        """Lazy load the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("openai required. Install with: pip install openai")
            
            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
            )
        return self._client
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to JSON, handling common issues."""
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {text[:500]}")
    
    def extract_single(
        self,
        jd_id: str,
        job_description: str,
        expertise: str = "",
        team_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """
        Extract requirements from a single JD.
        
        Args:
            jd_id: Unique identifier for the JD
            job_description: Main JD text
            expertise: Expertise/skills section if available
            team_description: Team description for context
            metadata: Additional metadata to pass through
            
        Returns:
            ExtractionResult with extracted requirements
        """
        client = self._get_client()
        
        user_prompt = EXTRACTION_USER_PROMPT.format(
            job_description=job_description or "(not provided)",
            expertise=expertise or "(not provided)",
            team_description=team_description or "(not provided)",
        )
        
        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000,
            )
            
            response_text = response.choices[0].message.content
            extracted = self._parse_llm_response(response_text)
            
            # Build result
            result = ExtractionResult(
                jd_id=jd_id,
                metadata=metadata or {},
                extraction_model=self.model_name,
            )
            
            # Parse licenses
            if "licenses" in extracted:
                result.licenses = RequirementCategory(
                    required=extracted["licenses"].get("required", []),
                    preferred=extracted["licenses"].get("preferred", []),
                )
            
            # Parse certifications
            if "certifications" in extracted:
                result.certifications = RequirementCategory(
                    required=extracted["certifications"].get("required", []),
                    preferred=extracted["certifications"].get("preferred", []),
                )
            
            # Parse skills
            if "skills" in extracted:
                skills = extracted["skills"]
                result.skills = SkillsExtraction(
                    technical=RequirementCategory(
                        required=skills.get("technical", {}).get("required", []),
                        preferred=skills.get("technical", {}).get("preferred", []),
                    ),
                    domain=RequirementCategory(
                        required=skills.get("domain", {}).get("required", []),
                        preferred=skills.get("domain", {}).get("preferred", []),
                    ),
                    soft=RequirementCategory(
                        required=skills.get("soft", {}).get("required", []),
                        preferred=skills.get("soft", {}).get("preferred", []),
                    ),
                )
            
            # Parse education
            if "education" in extracted:
                edu = extracted["education"]
                result.education = EducationExtraction(
                    level=RequirementCategory(
                        required=edu.get("level", {}).get("required", []),
                        preferred=edu.get("level", {}).get("preferred", []),
                    ),
                    fields=edu.get("fields", []),
                )
            
            # Parse experience
            if "experience" in extracted:
                exp = extracted["experience"]
                result.experience = ExperienceExtraction(
                    years_min=exp.get("years_min"),
                    years_preferred=exp.get("years_preferred"),
                    specific=RequirementCategory(
                        required=exp.get("specific", {}).get("required", []),
                        preferred=exp.get("specific", {}).get("preferred", []),
                    ),
                )
            
            # Parse tools
            if "tools" in extracted:
                result.tools = RequirementCategory(
                    required=extracted["tools"].get("required", []),
                    preferred=extracted["tools"].get("preferred", []),
                )
            
            # Parse languages
            if "languages" in extracted:
                result.languages = RequirementCategory(
                    required=extracted["languages"].get("required", []),
                    preferred=extracted["languages"].get("preferred", []),
                )
            
            return result
        
        except Exception as e:
            # Return failed extraction result
            return ExtractionResult(
                jd_id=jd_id,
                metadata=metadata or {},
                extraction_model=self.model_name,
                extraction_success=False,
                extraction_error=str(e),
            )
    
    def extract_batch(
        self,
        df: pd.DataFrame,
        jd_id_field: str = "jd_id",
        job_description_field: str = "jd_text",
        expertise_field: Optional[str] = None,
        team_description_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        show_progress: bool = True,
        save_interval: int = 50,
        output_path: Optional[str] = None,
        max_workers: int = 1,
        resume: bool = True,
    ) -> List[ExtractionResult]:
        """
        Extract requirements from a batch of JDs.
        
        Args:
            df: DataFrame with JD data
            jd_id_field: Column name for JD ID
            job_description_field: Column name for job description text
            expertise_field: Column name for expertise section (optional)
            team_description_field: Column name for team description (optional)
            metadata_fields: Additional columns to include in metadata
            show_progress: Print progress updates
            save_interval: Save intermediate results every N records
            output_path: Path to save intermediate/final results
            max_workers: Number of parallel workers (1=sequential, >1=parallel)
            resume: If True, skip JDs that were already extracted (requires output_path)
            
        Returns:
            List of ExtractionResult objects
        """
        # Load existing results for resume
        existing_results = {}
        if resume and output_path and Path(output_path).exists():
            try:
                existing = self.load_results(output_path)
                existing_results = {r.jd_id: r for r in existing}
                if show_progress:
                    print(f"  Resuming: found {len(existing_results)} existing extractions")
            except Exception as e:
                if show_progress:
                    print(f"  Could not load existing results: {e}")
        
        # Prepare extraction tasks
        tasks = []
        for idx, row in df.iterrows():
            jd_id = str(row[jd_id_field])
            
            # Skip if already extracted
            if jd_id in existing_results:
                continue
            
            # Get text fields
            job_description = str(row.get(job_description_field, ""))
            expertise = str(row.get(expertise_field, "")) if expertise_field else ""
            team_description = str(row.get(team_description_field, "")) if team_description_field else ""
            
            # Build metadata
            metadata = {}
            if metadata_fields:
                for fld in metadata_fields:
                    if fld in row:
                        metadata[fld] = row[fld]
            
            tasks.append({
                "jd_id": jd_id,
                "job_description": job_description,
                "expertise": expertise,
                "team_description": team_description,
                "metadata": metadata,
            })
        
        if show_progress:
            print(f"  Extracting {len(tasks)} JDs (skipped {len(existing_results)} existing)")
        
        if not tasks:
            return list(existing_results.values())
        
        # Process tasks
        new_results = []
        
        if max_workers <= 1:
            # Sequential processing
            new_results = self._extract_sequential(tasks, show_progress, save_interval, output_path, existing_results)
        else:
            # Parallel processing
            new_results = self._extract_parallel(tasks, max_workers, show_progress, save_interval, output_path, existing_results)
        
        # Combine with existing results
        all_results = list(existing_results.values()) + new_results
        
        # Save final results
        if output_path:
            self._save_results(all_results, output_path)
        
        if show_progress:
            successful = sum(1 for r in all_results if r.extraction_success)
            print(f"\nExtraction complete: {successful}/{len(all_results)} successful")
        
        return all_results
    
    def _extract_sequential(
        self,
        tasks: List[Dict],
        show_progress: bool,
        save_interval: int,
        output_path: Optional[str],
        existing_results: Dict[str, ExtractionResult],
    ) -> List[ExtractionResult]:
        """Sequential extraction."""
        results = []
        total = len(tasks)
        
        for idx, task in enumerate(tasks):
            if show_progress and (idx + 1) % 10 == 0:
                print(f"  Extracting {idx + 1}/{total}...")
            
            result = self.extract_single(**task)
            results.append(result)
            
            # Save intermediate results
            if output_path and save_interval and (idx + 1) % save_interval == 0:
                all_so_far = list(existing_results.values()) + results
                self._save_results(all_so_far, output_path)
        
        return results
    
    def _extract_parallel(
        self,
        tasks: List[Dict],
        max_workers: int,
        show_progress: bool,
        save_interval: int,
        output_path: Optional[str],
        existing_results: Dict[str, ExtractionResult],
    ) -> List[ExtractionResult]:
        """Parallel extraction using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        results = []
        results_lock = threading.Lock()
        total = len(tasks)
        completed = 0
        
        if show_progress:
            print(f"  Using {max_workers} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.extract_single, **task): task
                for task in tasks
            }
            
            # Process as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                except Exception as e:
                    task = future_to_task[future]
                    result = ExtractionResult(
                        jd_id=task["jd_id"],
                        metadata=task.get("metadata", {}),
                        extraction_model=self.model_name,
                        extraction_success=False,
                        extraction_error=str(e),
                    )
                
                with results_lock:
                    results.append(result)
                    completed += 1
                    
                    if show_progress and completed % 10 == 0:
                        print(f"  Completed {completed}/{total}...")
                    
                    # Save intermediate results
                    if output_path and save_interval and completed % save_interval == 0:
                        all_so_far = list(existing_results.values()) + results
                        self._save_results(all_so_far, output_path)
        
        return results
    
    def _save_results(self, results: List[ExtractionResult], output_path: str) -> None:
        """Save extraction results to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_results(path: str) -> List[ExtractionResult]:
        """Load extraction results from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return [ExtractionResult.from_dict(d) for d in data]

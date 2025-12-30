"""
Phase 5: Archetype Naming

Generate meaningful names for job archetypes using LLM.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from .aggregation import JobArchetype


NAMING_SYSTEM_PROMPT = """You are an expert HR analyst specializing in job taxonomy and classification.
Your task is to generate clear, professional names for job archetypes based on their characteristics.

GUIDELINES:
1. Names should be concise (2-5 words) and descriptive
2. Use standard industry terminology
3. Capture the core function/specialization of the role
4. Avoid overly generic names like "Business Professional" or "Technical Role"
5. Consider the division/function when relevant (e.g., "Investment Banking M&A" vs "Corporate M&A")

OUTPUT FORMAT: Respond with valid JSON only."""

NAMING_USER_PROMPT = """Generate a name for this job archetype based on the following characteristics:

CLUSTER SIZE: {member_count} JDs

COMMON TITLES:
{representative_titles}

TOP DIVISION: {top_division}
TOP FUNCTION: {top_function}

TOP REQUIRED SKILLS:
{top_skills}

TOP LICENSES/CERTIFICATIONS:
{top_licenses}

TOP TOOLS:
{top_tools}

Generate a professional archetype name. Return JSON:
{{
  "archetype_name": "Name of Archetype",
  "archetype_id": "lowercase-hyphenated-id",
  "alternative_names": ["Alternative 1", "Alternative 2"],
  "reasoning": "Brief explanation of why this name fits"
}}

Return ONLY valid JSON."""


STANDARDIZATION_SYSTEM_PROMPT = """You are an expert at standardizing job-related terminology.
Your task is to normalize skill names, tool names, and certifications to their canonical forms.

GUIDELINES:
1. Use official/standard names (e.g., "Python" not "python programming")
2. Expand common abbreviations where helpful (e.g., "ML" -> "Machine Learning")
3. Combine duplicates (e.g., "MS Excel", "Microsoft Excel", "Excel" -> "Microsoft Excel")
4. Keep industry-standard abbreviations (e.g., "SQL", "CFA", "FRM")

OUTPUT FORMAT: Respond with valid JSON only."""

STANDARDIZATION_USER_PROMPT = """Standardize these skill/tool names. Group duplicates and return canonical forms.

ITEMS TO STANDARDIZE:
{items}

Return JSON mapping original names to standardized names:
{{
  "original_name_1": "Standardized Name",
  "original_name_2": "Standardized Name",
  ...
}}

Return ONLY valid JSON."""


class ArchetypeNamer:
    """
    Generate names and standardize terminology for job archetypes.
    """
    
    def __init__(
        self,
        deployment_name: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
    ):
        """
        Initialize namer with Azure OpenAI credentials.
        
        Args:
            deployment_name: Azure OpenAI deployment name
            azure_endpoint: Azure endpoint
            api_key: API key
            api_version: API version
        """
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self._client = None
    
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
        """Parse LLM response to JSON."""
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
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {text[:500]}")
    
    def _get_top_items(
        self,
        freq_map: Dict[str, float],
        n: int = 5,
    ) -> str:
        """Get top N items from frequency map as formatted string."""
        if not freq_map:
            return "(none)"
        
        sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)[:n]
        return ", ".join([f"{item} ({freq:.0%})" for item, freq in sorted_items])
    
    def generate_name(
        self,
        archetype: JobArchetype,
    ) -> Dict[str, Any]:
        """
        Generate a name for an archetype.
        
        Args:
            archetype: JobArchetype to name
            
        Returns:
            Dict with archetype_name, archetype_id, alternative_names, reasoning
        """
        client = self._get_client()
        
        # Build skills string
        all_skills = {}
        for skill_type in [archetype.skills.technical, archetype.skills.domain]:
            all_skills.update(skill_type.required)
            all_skills.update(skill_type.preferred)
        
        # Build licenses/certs string
        all_licenses = {}
        all_licenses.update(archetype.licenses.required)
        all_licenses.update(archetype.licenses.preferred)
        all_licenses.update(archetype.certifications.required)
        all_licenses.update(archetype.certifications.preferred)
        
        # Build tools string
        all_tools = {}
        all_tools.update(archetype.tools.required)
        all_tools.update(archetype.tools.preferred)
        
        # Get top division/function
        top_division = list(archetype.division_distribution.keys())[0] if archetype.division_distribution else "(none)"
        top_function = list(archetype.function_distribution.keys())[0] if archetype.function_distribution else "(none)"
        
        user_prompt = NAMING_USER_PROMPT.format(
            member_count=archetype.member_count,
            representative_titles="\n".join([f"- {t}" for t in archetype.representative_titles[:5]]) or "(none)",
            top_division=top_division,
            top_function=top_function,
            top_skills=self._get_top_items(all_skills),
            top_licenses=self._get_top_items(all_licenses),
            top_tools=self._get_top_items(all_tools),
        )
        
        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": NAMING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            response_text = response.choices[0].message.content
            return self._parse_llm_response(response_text)
        
        except Exception as e:
            # Fallback: generate basic name from titles
            if archetype.representative_titles:
                # Extract common words from titles
                words = []
                for title in archetype.representative_titles[:3]:
                    words.extend(title.split())
                
                # Simple heuristic name
                common_words = [w for w in words if len(w) > 3 and w[0].isupper()]
                if common_words:
                    name = " ".join(common_words[:3])
                else:
                    name = f"Cluster {archetype.cluster_id}"
            else:
                name = f"Cluster {archetype.cluster_id}"
            
            return {
                "archetype_name": name,
                "archetype_id": re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-'),
                "alternative_names": [],
                "reasoning": f"Fallback due to error: {str(e)}",
            }
    
    def name_all_archetypes(
        self,
        archetypes: List[JobArchetype],
        show_progress: bool = True,
    ) -> List[JobArchetype]:
        """
        Generate names for all archetypes.
        
        Args:
            archetypes: List of JobArchetype objects
            show_progress: Print progress
            
        Returns:
            List of JobArchetype with names populated
        """
        for i, archetype in enumerate(archetypes):
            if show_progress:
                print(f"Naming archetype {i+1}/{len(archetypes)} (cluster {archetype.cluster_id})...")
            
            naming_result = self.generate_name(archetype)
            
            archetype.label = naming_result.get("archetype_name")
            archetype.archetype_id = naming_result.get("archetype_id")
            
            if show_progress:
                print(f"  -> {archetype.label}")
        
        return archetypes
    
    def standardize_skills(
        self,
        skills: List[str],
        batch_size: int = 50,
    ) -> Dict[str, str]:
        """
        Standardize a list of skill names.
        
        Args:
            skills: List of skill names to standardize
            batch_size: Number of skills per LLM call
            
        Returns:
            Dict mapping original names to standardized names
        """
        client = self._get_client()
        
        all_mappings = {}
        
        for i in range(0, len(skills), batch_size):
            batch = skills[i:i + batch_size]
            
            user_prompt = STANDARDIZATION_USER_PROMPT.format(
                items="\n".join([f"- {s}" for s in batch])
            )
            
            try:
                response = client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": STANDARDIZATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                
                response_text = response.choices[0].message.content
                batch_mappings = self._parse_llm_response(response_text)
                all_mappings.update(batch_mappings)
            
            except Exception as e:
                # Fallback: return items as-is
                for skill in batch:
                    all_mappings[skill] = skill
        
        return all_mappings
    
    def standardize_archetype_skills(
        self,
        archetypes: List[JobArchetype],
        show_progress: bool = True,
    ) -> tuple:
        """
        Standardize skills across all archetypes.
        
        Args:
            archetypes: List of JobArchetype objects
            show_progress: Print progress
            
        Returns:
            Tuple of (updated archetypes, skill mappings)
        """
        # Collect all unique skills
        all_skills = set()
        
        for archetype in archetypes:
            for skill_type in [archetype.skills.technical, archetype.skills.domain, archetype.skills.soft]:
                all_skills.update(skill_type.required.keys())
                all_skills.update(skill_type.preferred.keys())
        
        if show_progress:
            print(f"Standardizing {len(all_skills)} unique skills...")
        
        # Get standardized mappings
        mappings = self.standardize_skills(list(all_skills))
        
        if show_progress:
            # Count how many were changed
            changed = sum(1 for k, v in mappings.items() if k != v)
            print(f"  Standardized {changed} skills")
        
        # Apply mappings to archetypes
        def apply_mapping(freq_map: Dict[str, float], mappings: Dict[str, str]) -> Dict[str, float]:
            new_map = {}
            for skill, freq in freq_map.items():
                standardized = mappings.get(skill, skill)
                if standardized in new_map:
                    new_map[standardized] = max(new_map[standardized], freq)
                else:
                    new_map[standardized] = freq
            return new_map
        
        for archetype in archetypes:
            for skill_type in [archetype.skills.technical, archetype.skills.domain, archetype.skills.soft]:
                skill_type.required = apply_mapping(skill_type.required, mappings)
                skill_type.preferred = apply_mapping(skill_type.preferred, mappings)
        
        return archetypes, mappings
    
    def save_skill_mappings(
        self,
        mappings: Dict[str, str],
        output_path: str,
    ) -> None:
        """Save skill standardization mappings."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(mappings, f, indent=2, sort_keys=True)

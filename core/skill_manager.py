import os
import yaml
import structlog
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

logger = structlog.get_logger(__name__)

@dataclass
class SkillMetadata:
    name: str
    description: str
    folder_path: Path

class SkillManager:
    def __init__(self, skills_dir: str = "resync/skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_metadata: List[SkillMetadata] = []
        self._load_all_metadata()

    def _load_all_metadata(self):
        """Phase 1: Carrega apenas os metadados (YAML Frontmatter) no boot."""
        if not self.skills_dir.exists():
            logger.warning("skills_directory_not_found", path=str(self.skills_dir))
            return

        for skill_folder in self.skills_dir.iterdir():
            if skill_folder.is_dir():
                skill_file = skill_folder / "SKILL.md"
                if skill_file.exists():
                    metadata = self._parse_frontmatter(skill_file)
                    if metadata:
                        self.skills_metadata.append(
                            SkillMetadata(
                                name=metadata.get("name", skill_folder.name),
                                description=metadata.get("description", ""),
                                folder_path=skill_folder
                            )
                        )
        logger.info("skills_loaded", count=len(self.skills_metadata))

    def _parse_frontmatter(self, file_path: Path) -> dict:
        """Extrai o bloco YAML entre '---' no topo do arquivo."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        return yaml.safe_load(parts[1])
        except Exception as e:
            logger.error("error_parsing_skill_frontmatter", file=str(file_path), error=str(e))
        return {}

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """Phase 2: Deep Loading. Carrega o texto completo sob demanda."""
        for skill in self.skills_metadata:
            if skill.name == skill_name:
                skill_file = skill.folder_path / "SKILL.md"
                with open(skill_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove o frontmatter para não confundir o LLM
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        return parts[2].strip() if len(parts) >= 3 else content
                return None
        return None

# Singleton Pattern similar ao AgentManager
_skill_manager = None
def get_skill_manager() -> SkillManager:
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager

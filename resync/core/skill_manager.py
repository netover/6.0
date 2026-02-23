# pylint: disable=all
# mypy: no-rerun
"""
Skill Manager - Carrega e gerencia skills do sistema Resync.

v2.0 - Refatorado para:
- Indexação O(1) por nome
- Cache de conteúdo com invalidação por mtime
- Validação de frontmatter
- Injeção via DI (EnterpriseState)
- Singleton deprecated (mantido para compatibilidade)

v2.1 - Adicionado:
- IO async com aiofiles para alta concorrência
- build_skill_context_async para caminho hot
"""

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
import structlog

# Async file IO (optional, graceful fallback)
try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Schema de campos permitidos no frontmatter
ALLOWED_FRONTMATTER_FIELDS = frozenset(
    {
        "name",
        "description",
        "tags",
        "intents",
        "tools_expected",
        "version",
        "author",
        "created_at",
    }
)

# Limites de segurança
MAX_SKILL_SIZE = 2500  # chars por skill
MAX_TOTAL_CONTEXT = 6000  # chars total (mantido do original)


@dataclass
class SkillMetadata:
    """Metadados de uma skill."""

    name: str
    description: str
    folder_path: Path
    tags: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    tools_expected: List[str] = field(default_factory=list)
    version: str = "1.0"
    file_mtime: float = 0.0


class SkillManager:
    """
    Gerencia skills do sistema Resync.

    Features:
    - Indexação O(1) por nome via _by_name dict
    - Cache de conteúdo com invalidação por mtime
    - Validação de frontmatter
    - Limites de tamanho por skill

    Usage via DI:
        skill_manager = SkillManager(skills_dir="resync/skills")
        # ou obter de EnterpriseState
    """

    def __init__(self, skills_dir: str | None = None):
        # 1) Override por env
        env_dir = os.getenv("RESYNC_SKILLS_DIR")

        # 2) Escolher diretório
        candidate = skills_dir or env_dir or "skills"
        self.skills_dir = Path(candidate)

        # 3) Fallback automático para estrutura do projeto
        if not self.skills_dir.exists():
            fallback = Path("resync/skills")
            if fallback.exists():
                self.skills_dir = fallback

        # Indexação e cache
        self.skills_metadata: List[SkillMetadata] = []
        self._by_name: Dict[str, SkillMetadata] = {}  # O(1) lookup
        self._content_cache: Dict[str, Tuple[float, str]] = {}  # (mtime, content)

        # Validar diretório de skills (segurança)
        self._validate_skills_dir()

        # Carregar metadados
        self._load_all_metadata()

    def _validate_skills_dir(self) -> None:
        """Valida o diretório de skills por segurança."""
        if not self.skills_dir.exists():
            logger.warning("skills_directory_not_found", path=str(self.skills_dir))
            return

        # Verificar se é um diretório
        if not self.skills_dir.is_dir():
            logger.error("skills_path_not_directory", path=str(self.skills_dir))
            return

        # Em produção, verificar se não é world-writable
        try:
            mode = self.skills_dir.stat().st_mode
            if mode & 0o002:  # world-writable
                logger.warning(
                    "skills_dir_world_writable",
                    path=str(self.skills_dir),
                    hint="Consider restricting permissions in production",
                )
        except OSError:
            pass

    def _load_all_metadata(self) -> None:
        """Phase 1: Carrega metadados (YAML Frontmatter) no boot."""
        if not self.skills_dir.exists():
            return

        for skill_folder in self.skills_dir.iterdir():
            if not skill_folder.is_dir():
                continue

            skill_file = skill_folder / "SKILL.md"
            if not skill_file.exists():
                continue

            metadata = self._parse_frontmatter(skill_file)
            if not metadata:
                continue

            # Obter mtime para cache invalidation
            try:
                file_mtime = skill_file.stat().st_mtime
            except OSError:
                file_mtime = 0.0

            skill_meta = SkillMetadata(
                name=metadata.get("name", skill_folder.name),
                description=metadata.get("description", ""),
                folder_path=skill_folder,
                tags=metadata.get("tags", []) or [],
                intents=metadata.get("intents", []) or [],
                tools_expected=metadata.get("tools_expected", []) or [],
                version=metadata.get("version", "1.0") or "1.0",
                file_mtime=file_mtime,
            )

            self.skills_metadata.append(skill_meta)
            self._by_name[skill_meta.name] = skill_meta

        logger.info(
            "skills_loaded",
            path=str(self.skills_dir),
            count=len(self.skills_metadata),
            skills=list(self._by_name.keys()),
        )

    def _parse_frontmatter(self, file_path: Path) -> dict:
        """
        Extrai e valida o bloco YAML entre '---' no topo do arquivo.

        Retorna apenas campos permitidos, rejeitando dados inesperados.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.startswith("---"):
                    return {}

                parts = content.split("---", 2)
                if len(parts) < 3:
                    return {}

                raw_data = yaml.safe_load(parts[1])
                if not isinstance(raw_data, dict):
                    return {}

                # Validar e filtrar campos permitidos
                validated: dict[str, Any] = {}
                for key in ALLOWED_FRONTMATTER_FIELDS:
                    if key in raw_data:
                        value = raw_data[key]
                        # Converter para tipos seguros
                        if key in ("tags", "intents", "tools_expected"):
                            if isinstance(value, list):
                                validated[key] = [str(item) for item in value]
                            elif isinstance(value, str):
                                validated[key] = [value]
                            else:
                                validated[key] = []
                        elif key in ("name", "description", "version"):
                            validated[key] = str(value) if value else ""
                        else:
                            validated[key] = value

                # Normalizar nome (lowercase, slug)
                if "name" in validated:
                    validated["name"] = validated["name"].lower().replace(" ", "-")

                return validated

        except yaml.YAMLError as e:
            logger.error(
                "skill_frontmatter_yaml_error", file=str(file_path), error=str(e)
            )
        except Exception as e:
            logger.error(
                "error_parsing_skill_frontmatter", file=str(file_path), error=str(e)
            )

        return {}

    def get_skill_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        """
        Obtém metadados de uma skill por nome.

        O(1) via índice interno.
        """
        return self._by_name.get(skill_name)

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """
        Phase 2: Deep Loading. Carrega o texto completo sob demanda.

        Features:
        - Lookup O(1) por nome
        - Cache com invalidação por mtime
        - Limite de tamanho por skill
        - Remove frontmatter do conteúdo
        """
        skill = self._by_name.get(skill_name)
        if skill is None:
            logger.warning("skill_not_found", skill=skill_name)
            return None

        skill_file = skill.folder_path / "SKILL.md"
        if not skill_file.exists():
            return None

        try:
            current_mtime = skill_file.stat().st_mtime
        except OSError:
            current_mtime = 0.0

        # Verificar cache
        cache_key = skill_name
        if cache_key in self._content_cache:
            cached_mtime, cached_content = self._content_cache[cache_key]
            if cached_mtime == current_mtime:
                return cached_content

        # Carregar conteúdo
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error("skill_read_error", skill=skill_name, error=str(e))
            return None

        # Remover frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            content = parts[2].strip() if len(parts) >= 3 else content

        # Aplicar limite de tamanho
        if len(content) > MAX_SKILL_SIZE:
            content = content[:MAX_SKILL_SIZE] + "\n... [truncated]"
            logger.debug(
                "skill_content_truncated",
                skill=skill_name,
                original_size=len(content),
                max_size=MAX_SKILL_SIZE,
            )

        # Atualizar cache
        self._content_cache[cache_key] = (current_mtime, content)

        return content

    async def get_skill_content_async(self, skill_name: str) -> Optional[str]:
        """
        Async version of get_skill_content for high-concurrency paths.

        Uses aiofiles for non-blocking IO. Falls back to sync version
        if aiofiles is not available.

        Features:
        - Non-blocking file read
        - Same caching as sync version
        - Same size limits and frontmatter removal
        """
        skill = self._by_name.get(skill_name)
        if skill is None:
            logger.warning("skill_not_found_async", skill=skill_name)
            return None

        skill_file = skill.folder_path / "SKILL.md"
        if not skill_file.exists():
            return None

        try:
            current_mtime = skill_file.stat().st_mtime
        except OSError:
            current_mtime = 0.0

        # Check cache first
        cache_key = skill_name
        if cache_key in self._content_cache:
            cached_mtime, cached_content = self._content_cache[cache_key]
            if cached_mtime == current_mtime:
                return cached_content

        # Load content (async if aiofiles available)
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(skill_file, "r", encoding="utf-8") as f:
                    content = await f.read()
            else:
                # Fallback to sync (still works, just blocks)
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as e:
            logger.error("skill_read_error_async", skill=skill_name, error=str(e))
            return None

        # Remove frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            content = parts[2].strip() if len(parts) >= 3 else content

        # Apply size limit
        if len(content) > MAX_SKILL_SIZE:
            content = content[:MAX_SKILL_SIZE] + "\n... [truncated]"
            logger.debug(
                "skill_content_truncated_async",
                skill=skill_name,
                original_size=len(content),
                max_size=MAX_SKILL_SIZE,
            )

        # Update cache
        self._content_cache[cache_key] = (current_mtime, content)

        return content

    async def build_skill_context_async(
        self,
        skill_names: List[str],
        max_skills: int = 3,
        max_total_size: int = MAX_TOTAL_CONTEXT,
    ) -> str:
        """
        Async version of build_skill_context for hot paths.

        Builds skill context using async content loading.
        Recommended for high-concurrency endpoints.
        """
        if not skill_names:
            return ""

        available_names = set(self._by_name.keys())
        chunks = []
        total_size = 0

        for name in skill_names[:max_skills]:
            if name not in available_names:
                logger.warning("mapped_skill_missing_async", skill=name)
                continue

            content = await self.get_skill_content_async(name)
            if not content:
                continue

            chunk = f"# Skill: {name}\n{content}"
            chunk_size = len(chunk)

            if total_size + chunk_size > max_total_size:
                remaining = max_total_size - total_size
                if remaining > 100:
                    chunk = chunk[:remaining] + "\n... [truncated]"
                    chunks.append(chunk)
                break

            chunks.append(chunk)
            total_size += chunk_size

        if not chunks:
            return ""

        result = "\n\n".join(chunks)

        # Observability
        logger.info(
            "skill_context_injected",
            skills_injected=len(chunks),
            total_size=len(result),
            skills=[
                name for name in skill_names[:max_skills] if name in available_names
            ],
            async_mode=True,
        )

        return result

    def build_skill_context(
        self,
        skill_names: List[str],
        max_skills: int = 3,
        max_total_size: int = MAX_TOTAL_CONTEXT,
    ) -> str:
        """
        Constrói contexto de skills para injeção no prompt do agente.

        Args:
            skill_names: Lista de nomes de skills
            max_skills: Máximo de skills a injetar (default: 3)
            max_total_size: Tamanho máximo total (default: 6000)

        Returns:
            String formatada com skills, limitada em tamanho
        """
        if not skill_names:
            return ""

        available_names = set(self._by_name.keys())
        chunks = []
        total_size = 0

        for name in skill_names[:max_skills]:
            if name not in available_names:
                logger.warning("mapped_skill_missing", skill=name)
                continue

            content = self.get_skill_content(name)
            if not content:
                continue

            chunk = f"# Skill: {name}\n{content}"
            chunk_size = len(chunk)
            truncated = False

            if total_size + chunk_size > max_total_size:
                # Truncar último chunk se necessário
                remaining = max_total_size - total_size
                if remaining > 100:  # Só incluir se houver espaço mínimo
                    chunk = chunk[:remaining] + "\n... [truncated]"
                    chunks.append(chunk)
                    truncated = True
                break

            chunks.append(chunk)
            total_size += chunk_size

        if not chunks:
            return ""

        result = "\n\n".join(chunks)

        # Observability - log skill injection details
        injected_skills = [
            name for name in skill_names[:max_skills] if name in available_names
        ]
        missing_skills = [
            name for name in skill_names[:max_skills] if name not in available_names
        ]

        logger.info(
            "skill_context_built",
            skills_injected=len(chunks),
            total_size=len(result),
            skills=injected_skills,
            missing_skills=missing_skills if missing_skills else None,
            truncated=truncated if "truncated" in dir() else False,
            async_mode=False,
        )

        return result

    def list_skills(self) -> List[Dict[str, Any]]:
        """Lista todas as skills disponíveis (para API/admin)."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "intents": skill.intents,
                "version": skill.version,
            }
            for skill in self.skills_metadata
        ]

    def reload(self) -> None:
        """Recarrega todas as skills (para hot-reload via admin)."""
        self.skills_metadata.clear()
        self._by_name.clear()
        self._content_cache.clear()
        self._load_all_metadata()
        logger.info("skills_reloaded", count=len(self.skills_metadata))


# =============================================================================
# DEPRECATED: Singleton global (mantido para compatibilidade)
# =============================================================================

_skill_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    """
    Get SkillManager singleton.

    DEPRECATED: Use DI via EnterpriseState instead.
    This function will be removed in a future version.

    Migration:
        # Old (deprecated):
        from resync.core.skill_manager import get_skill_manager
        sm = get_skill_manager()

        # New (recommended):
        from resync.core.wiring import get_skill_manager
        sm = get_skill_manager(request)  # via FastAPI Depends
    """
    global _skill_manager

    warnings.warn(
        "get_skill_manager() from skill_manager module is deprecated. "
        "Use DI via EnterpriseState (wiring.get_skill_manager).",
        DeprecationWarning,
        stacklevel=2,
    )

    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager


def create_skill_manager(skills_dir: str | None = None) -> SkillManager:
    """
    Factory function to create a SkillManager instance.

    Use this for explicit instantiation (recommended for DI).
    """
    return SkillManager(skills_dir=skills_dir)

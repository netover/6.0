# pylint: disable=all
# mypy: no-rerun
"""
Supervised Agent Evolution System for TWS/HWA

Allows agents to learn from feedback and improve over time,
but with human oversight and control.

Key Principles:
1. ✅ Automatic feedback collection
2. ✅ Automatic pattern detection
3. ✅ Automatic improvement suggestions
4. ⚠️ HUMAN approval required (supervised)
5. ✅ Sandbox testing mandatory
6. ✅ Monitored deployment with auto-rollback

Author: Resync Team
Version: 5.9.9
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import aiofiles
import structlog
from pydantic import BaseModel, Field

from resync.core.llm_config import get_llm_config
from resync.core.task_tracker import create_tracked_task

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class FeedbackType(str, Enum):
    """Type of user feedback."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    COMMENT = "comment"


class AgentFeedback(BaseModel):
    """User feedback on agent performance."""

    id: str
    agent_name: str  # e.g., "job_analyst", "dependency_specialist"
    task: str  # e.g., "analyze_job:PAYROLL_NIGHTLY"
    output: dict  # Agent's analysis/recommendation
    feedback_type: FeedbackType
    user_comment: str | None = None
    correct: bool  # True if feedback is positive
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # TWS/HWA specific context
    job_name: str | None = None
    job_type: str | None = None  # BACKUP, ETL, REPORT, etc.
    error_type: str | None = None


class DetectedPattern(BaseModel):
    """Pattern detected in feedback data."""

    id: str
    pattern_type: str  # e.g., "missing_dependency", "false_alert"
    description: str
    frequency: int  # How many times seen
    confidence: float  # 0-1
    examples: list[str]  # Example job names

    # TWS/HWA specific
    job_pattern: str | None = None  # e.g., "BACKUP_*", "ETL_*"

    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImprovementSuggestion(BaseModel):
    """Suggested improvement to agent behavior."""

    id: str
    agent_name: str
    pattern_id: str

    current_prompt: str
    proposed_prompt: str

    rationale: str
    estimated_impact: str  # e.g., "+15% accuracy"

    status: str = "pending"  # pending, testing, approved, rejected, deployed

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: str | None = None
    approved_at: datetime | None = None


class SandboxTestResult(BaseModel):
    """Results from sandbox testing."""

    suggestion_id: str

    test_cases_count: int

    current_accuracy: float
    improved_accuracy: float
    improvement_pct: float

    regressions_detected: list[str]
    side_effects: list[str]

    safe_to_deploy: bool

    tested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Feedback Collection
# =============================================================================


class AgentFeedbackCollector:
    """
    Collects and stores feedback on agent performance.

    Examples of TWS/HWA feedback:

    1. Job Analyst missed dependency:
       Job: PAYROLL_NIGHTLY
       Output: {"dependencies": ["BACKUP_DB"]}
       Feedback: 👎 "Missed dependency with TIMEKEEPING_CLOSE"

    2. False alert on normal delay:
       Job: ETL_CUSTOMER
       Output: {"alert": "Job delayed 15 min"}
       Feedback: 👎 "ETL jobs always run 2-4h, this is normal"

    3. Correct analysis:
       Job: REPORT_MONTHLY
       Output: {"risk": "high", "reason": "30+ dependencies"}
       Feedback: 👍 "Correct assessment"
    """

    def __init__(self):
        self.storage_path = Path("data/agent_feedback")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def collect_feedback(
        self,
        agent_name: str,
        task: str,
        output: dict,
        feedback_type: FeedbackType,
        user_comment: str | None = None,
        job_name: str | None = None,
    ) -> AgentFeedback:
        """
        Collect user feedback on agent output.

        Example:
            feedback = await collector.collect_feedback(
                agent_name="job_analyst",
                task="analyze_job:PAYROLL_NIGHTLY",
                output={"dependencies": ["BACKUP_DB"], "risk": "medium"},
                feedback_type=FeedbackType.THUMBS_DOWN,
                user_comment="Missed dependency with TIMEKEEPING_CLOSE",
                job_name="PAYROLL_NIGHTLY"
            )
        """
        feedback = AgentFeedback(
            id=self._generate_id(),
            agent_name=agent_name,
            task=task,
            output=output,
            feedback_type=feedback_type,
            user_comment=user_comment,
            correct=feedback_type == FeedbackType.THUMBS_UP,
            job_name=job_name,
            job_type=self._infer_job_type(job_name) if job_name else None,
        )

        await self._store_feedback(feedback)

        logger.info(
            "feedback_collected",
            agent=agent_name,
            task=task,
            feedback_type=feedback_type,
            correct=feedback.correct,
        )

        # Trigger pattern analysis asynchronously
        await create_tracked_task(
            self._analyze_patterns_async(agent_name), name="analyze_patterns_async"
        )

        return feedback

    def _infer_job_type(self, job_name: str) -> str:
        """
        Infer job type from name (TWS/HWA patterns).

        Examples:
        - BACKUP_DB → BACKUP
        - ETL_CUSTOMER → ETL
        - PAYROLL_NIGHTLY → PAYROLL
        - REPORT_MONTHLY → REPORT
        """
        name_upper = job_name.upper()

        if name_upper.startswith("BACKUP_"):
            return "BACKUP"
        if name_upper.startswith("ETL_"):
            return "ETL"
        if name_upper.startswith("PAYROLL_"):
            return "PAYROLL"
        if name_upper.startswith("REPORT_"):
            return "REPORT"
        if name_upper.startswith("RECOVERY_"):
            return "RECOVERY"
        if name_upper.startswith("RESTART_"):
            return "RESTART"
        if name_upper.startswith("FILE_WATCHER_"):
            return "FILE_WATCHER"
        return "OTHER"

    async def _store_feedback(self, feedback: AgentFeedback):
        """Store feedback to disk."""
        file_path = self.storage_path / f"{feedback.id}.json"

        async with aiofiles.open(file_path, "w") as f:
            await f.write(feedback.model_dump_json(indent=2))

    async def _analyze_patterns_async(self, agent_name: str):
        """Trigger pattern analysis (runs in background)."""
        try:
            analyzer = PatternDetector()
            await analyzer.analyze_feedback(agent_name)
        except Exception as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Pattern analysis failed: %s", e, exc_info=True)

    def _generate_id(self) -> str:
        """Generate unique feedback ID."""
        return f"feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


# =============================================================================
# Pattern Detection
# =============================================================================


class PatternDetector:
    """
    Detects patterns in feedback data.

    TWS/HWA Patterns Examples:

    1. Missing Dependency Pattern:
       - PAYROLL_* jobs often missing TIMEKEEPING dependency
       - REPORT_* jobs missing CLOSE_OF_BUSINESS dependency

    2. False Alert Pattern:
       - ETL_* jobs flagged as delayed but 2-4h is normal
       - BACKUP_* late on weekends is expected (low priority)

    3. Job Type Patterns:
       - BACKUP_* jobs always critical
       - RECOVERY_* jobs expected to fail sometimes (retry logic)
    """

    def __init__(self):
        self.storage_path = Path("data/agent_patterns")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def analyze_feedback(self, agent_name: str):
        """
        Analyze feedback for patterns.

        Uses LLM to detect patterns in negative feedback.
        """
        # Load recent feedback (last 30 days)
        feedbacks = await self._load_recent_feedback(agent_name, days=30)

        # Filter negative feedback
        negative = [f for f in feedbacks if not f.correct]

        if len(negative) < 3:
            logger.debug("Not enough negative feedback for %s", agent_name)
            return

        # Group by job type
        by_job_type = self._group_by_job_type(negative)

        # Detect patterns using LLM
        for job_type, feedback_list in by_job_type.items():
            if len(feedback_list) >= 2:
                pattern = await self._detect_pattern_llm(
                    agent_name, job_type, feedback_list
                )

                if pattern:
                    await self._store_pattern(pattern)

                    # Trigger improvement suggestion
                    await create_tracked_task(
                        self._create_improvement_suggestion(pattern)
                    )

    def _group_by_job_type(
        self, feedbacks: list[AgentFeedback]
    ) -> dict[str, list[AgentFeedback]]:
        """Group feedback by job type."""
        grouped = {}

        for feedback in feedbacks:
            job_type = feedback.job_type or "OTHER"
            if job_type not in grouped:
                grouped[job_type] = []
            grouped[job_type].append(feedback)

        return grouped

    async def _detect_pattern_llm(
        self, agent_name: str, job_type: str, feedbacks: list[AgentFeedback]
    ) -> DetectedPattern | None:
        """
        Use LLM to detect pattern in feedback.

        Prompt LLM with examples and ask it to identify pattern.
        """
        llm_config = get_llm_config()
        model = llm_config.get_model()

        # Build prompt with feedback examples
        examples = []
        for f in feedbacks:
            examples.append(
                {
                    "job_name": f.job_name,
                    "agent_output": f.output,
                    "user_feedback": f.user_comment,
                }
            )

        prompt = f"""
You are analyzing feedback on a TWS/HWA workload automation agent.

Agent: {agent_name}
Job Type: {job_type}

Negative Feedback Examples:
{self._format_examples(examples)}

Task:
Identify if there is a recurring pattern in these failures.

TWS/HWA Context:
- BACKUP_* jobs: Critical, run before other jobs
- ETL_* jobs: Long-running (2-4h normal), data extraction
- PAYROLL_* jobs: Critical, depend on TIMEKEEPING data
- REPORT_* jobs: Run after CLOSE_OF_BUSINESS
- RECOVERY_* jobs: Expected to retry, not always errors

Output JSON format:
{{
    "pattern_found": true/false,
    "pattern_type": "missing_dependency" | "false_alert" | "incorrect_risk" | "other",
    "description": "Brief description",
    "job_pattern": "BACKUP_*" | "ETL_*" | specific pattern,
    "confidence": 0.0-1.0
}}
"""

        try:
            # Call LLM (using litellm or openai)
            # Simplified for example - actual implementation would use
            # proper LLM client
            response = self._call_llm(model, prompt)

            # Parse response
            import json

            result = json.loads(response)

            if result.get("pattern_found"):
                pattern = DetectedPattern(
                    id=self._generate_id(),
                    pattern_type=result["pattern_type"],
                    description=result["description"],
                    frequency=len(feedbacks),
                    confidence=result["confidence"],
                    examples=[f.job_name for f in feedbacks],
                    job_pattern=result.get("job_pattern"),
                )

                logger.info(
                    "pattern_detected",
                    agent=agent_name,
                    job_type=job_type,
                    pattern_type=pattern.pattern_type,
                    confidence=pattern.confidence,
                )

                return pattern

        except Exception as e:
            logger.error("Pattern detection failed: %s", e, exc_info=True)

        return None

    def _format_examples(self, examples: list[dict]) -> str:
        """Format examples for LLM prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(
                f"{i}. Job: {ex['job_name']}\n"
                f"   Agent output: {ex['agent_output']}\n"
                f"   User feedback: {ex['user_feedback']}\n"
            )
        return "\n".join(formatted)

    def _call_llm(self, model: str, prompt: str) -> str:
        """Call LLM (placeholder - implement with litellm)."""
        # STUB: Returns hardcoded response;
        # implement with litellm for real pattern detection
        return (
            '{"pattern_found": true, "pattern_type": "missing_dependency", '
            '"description": "PAYROLL jobs missing TIMEKEEPING dependency", '
            '"job_pattern": "PAYROLL_*", "confidence": 0.85}'
        )

    async def _store_pattern(self, pattern: DetectedPattern):
        """Store detected pattern."""
        file_path = self.storage_path / f"{pattern.id}.json"

        async with aiofiles.open(file_path, "w") as f:
            await f.write(pattern.model_dump_json(indent=2))

    async def _create_improvement_suggestion(self, pattern: DetectedPattern):
        """Create improvement suggestion based on pattern."""
        suggester = ImprovementSuggester()
        await suggester.create_suggestion(pattern)

    async def _load_recent_feedback(
        self, agent_name: str, days: int = 30
    ) -> list[AgentFeedback]:
        """Load recent feedback from disk."""
        import json

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        feedbacks = []

        feedback_dir = Path("data/agent_feedback")
        if not feedback_dir.exists():
            return []

        for file_path in feedback_dir.glob("*.json"):
            try:
                async with aiofiles.open(file_path) as f:
                    content = await f.read()
                    data = json.loads(content)
                    feedback = AgentFeedback(**data)

                    if (
                        feedback.agent_name == agent_name
                        and feedback.timestamp >= cutoff
                    ):
                        feedbacks.append(feedback)
            except Exception as exc:
                logger.debug(
                    "suppressed_exception", error=str(exc), exc_info=True
                )  # was: pass

        return feedbacks

    def _generate_id(self) -> str:
        """Generate unique pattern ID."""
        return f"pattern_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


# =============================================================================
# Improvement Suggester
# =============================================================================


class ImprovementSuggester:
    """
    Generates improvement suggestions based on detected patterns.

    Uses LLM to propose prompt improvements.
    """

    def __init__(self):
        self.storage_path = Path("data/agent_improvements")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def create_suggestion(
        self, pattern: DetectedPattern
    ) -> ImprovementSuggestion:
        """
        Create improvement suggestion based on pattern.

        Uses LLM to suggest prompt modification.
        """
        # Get current agent prompt
        current_prompt = self._get_current_prompt()

        # Ask LLM for improved prompt
        proposed_prompt = await self._generate_improved_prompt(current_prompt, pattern)

        suggestion = ImprovementSuggestion(
            id=self._generate_id(),
            agent_name=self._infer_agent_name(pattern.pattern_type),
            pattern_id=pattern.id,
            current_prompt=current_prompt,
            proposed_prompt=proposed_prompt,
            rationale=self._generate_rationale(pattern),
            estimated_impact=f"+{int(pattern.confidence * 20)}% accuracy (estimated)",
        )

        await self._store_suggestion(suggestion)

        logger.info(
            "improvement_suggested",
            pattern_type=pattern.pattern_type,
            confidence=pattern.confidence,
        )

        return suggestion

    def _get_current_prompt(self) -> str:
        """Get current agent prompt."""
        # STUB: Returns default prompt — implement agent config lookup
        return "You are a TWS/HWA job analyst. Analyze job dependencies and risks."

    async def _generate_improved_prompt(
        self, current_prompt: str, pattern: DetectedPattern
    ) -> str:
        """
        Generate improved prompt using LLM.

        Similar to MetaPrompt approach from article.
        """
        llm_config = get_llm_config()
        model = llm_config.get_model()

        meta_prompt = f"""
You are a prompt optimization expert for TWS/HWA workload automation.

Current Agent Prompt:
{current_prompt}

Detected Pattern (Failure):
- Type: {pattern.pattern_type}
- Description: {pattern.description}
- Job Pattern: {pattern.job_pattern}
- Examples: {", ".join(pattern.examples)}
- Frequency: {pattern.frequency} occurrences

Task:
Write an improved prompt that addresses this pattern.

TWS/HWA Rules to Consider:
1. BACKUP_* jobs are always critical and run before others
2. PAYROLL_* jobs depend on TIMEKEEPING_* jobs
3. ETL_* jobs are long-running (2-4h normal)
4. REPORT_* jobs wait for CLOSE_OF_BUSINESS
5. RECOVERY_* jobs are expected to retry

Output only the improved prompt (no explanation).
"""

        try:
            improved = self._call_llm(model, meta_prompt)
            return improved.strip()
        except Exception as e:
            logger.error("Prompt improvement failed: %s", e, exc_info=True)
            return current_prompt

    def _call_llm(self, model: str, prompt: str) -> str:
        """Call LLM (placeholder)."""
        # STUB: Returns hardcoded prompt;
        # implement with litellm for real prompt improvement
        return """
You are a TWS/HWA job analyst. Analyze job dependencies and risks.

IMPORTANT RULES FOR PAYROLL JOBS:
- PAYROLL_* jobs ALWAYS depend on TIMEKEEPING_CLOSE
- Check for this dependency explicitly
- Flag as high-risk if TIMEKEEPING dependency is missing

When analyzing jobs:
1. Check job name patterns (BACKUP_*, ETL_*, PAYROLL_*, REPORT_*)
2. Apply type-specific rules
3. Verify critical dependencies
4. Assess risk based on dependency chain
"""

    def _generate_rationale(self, pattern: DetectedPattern) -> str:
        """Generate human-readable rationale."""
        return (
            "Pattern detected: "
            f"{pattern.description}. Seen {pattern.frequency} times "
            f"with {int(pattern.confidence * 100)}% confidence."
        )

    def _infer_agent_name(self, pattern_type: str) -> str:
        """Infer which agent needs improvement."""
        if "dependency" in pattern_type.lower():
            return "dependency_specialist"
        if "risk" in pattern_type.lower():
            return "job_analyst"
        return "general_agent"

    async def _store_suggestion(self, suggestion: ImprovementSuggestion):
        """Store suggestion to disk."""
        file_path = self.storage_path / f"{suggestion.id}.json"

        async with aiofiles.open(file_path, "w") as f:
            await f.write(suggestion.model_dump_json(indent=2))

    def _generate_id(self) -> str:
        """Generate unique suggestion ID."""
        return f"suggestion_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


# =============================================================================
# Sandbox Testing
# =============================================================================


class SandboxTester:
    """
    Tests improvements in isolated environment.

    Runs both current and improved prompts on historical test cases
    to compare performance.
    """

    async def test_improvement(
        self, suggestion: ImprovementSuggestion
    ) -> SandboxTestResult:
        """
        Test improvement suggestion in sandbox.

        Loads historical test cases and runs both current and improved
        versions to compare.
        """
        # Load test cases
        test_cases = self._load_test_cases(suggestion.agent_name, count=20)

        # Run current version
        current_results = self._run_test_cases(
            suggestion.agent_name, suggestion.current_prompt, test_cases
        )

        # Run improved version
        improved_results = self._run_test_cases(
            suggestion.agent_name, suggestion.proposed_prompt, test_cases
        )

        # Compare
        current_accuracy = self._calculate_accuracy(current_results)
        improved_accuracy = self._calculate_accuracy(improved_results)

        # Detect regressions
        regressions = self._detect_regressions(current_results, improved_results)

        result = SandboxTestResult(
            suggestion_id=suggestion.id,
            test_cases_count=len(test_cases),
            current_accuracy=current_accuracy,
            improved_accuracy=improved_accuracy,
            improvement_pct=improved_accuracy - current_accuracy,
            regressions_detected=regressions,
            side_effects=[],
            safe_to_deploy=len(regressions) == 0
            and improved_accuracy > current_accuracy,
        )

        logger.info(
            "sandbox_test_completed",
            suggestion_id=suggestion.id,
            improvement=f"+{result.improvement_pct * 100:.1f}%",
            safe=result.safe_to_deploy,
        )

        return result

    def _load_test_cases(self, agent_name: str, count: int = 20) -> list[dict]:
        """
        Load historical test cases.

        Uses past successful analyses as ground truth.
        """
        # STUB: Returns hardcoded test case — implement DB integration
        return [
            {
                "job_name": "PAYROLL_NIGHTLY",
                "expected_dependencies": ["BACKUP_DB", "TIMEKEEPING_CLOSE"],
                "expected_risk": "high",
            },
            # ... more test cases
        ]

    def _run_test_cases(
        self, agent_name: str, prompt: str, test_cases: list[dict]
    ) -> list[dict]:
        """Run test cases with given prompt."""
        # STUB: Returns empty — implement agent test execution
        return []

    def _calculate_accuracy(self, results: list[dict]) -> float:
        """Calculate accuracy from test results."""
        if not results:
            return 0.0

        correct = sum(1 for r in results if r.get("correct", False))
        return correct / len(results)

    def _detect_regressions(
        self, current: list[dict], improved: list[dict]
    ) -> list[str]:
        """Detect cases where improved version performs worse."""
        regressions = []

        for i, (curr, impr) in enumerate(zip(current, improved, strict=False)):
            if curr.get("correct") and not impr.get("correct"):
                regressions.append(f"Test case {i}: Regression detected")

        return regressions

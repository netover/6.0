"""
Example: Using Central LLM Configuration

Shows how to use centralized LLM config instead of hardcoded models.

BEFORE (WRONG):
    model = "gpt-4o"  # ❌ Hardcoded!

AFTER (CORRECT):
    from resync.core.llm_config import get_llm_config
    model = get_llm_config().get_model()  # ✅ Central config!

Author: Resync Team
Version: 5.9.9

NOTE: This file contains documentation examples only.
      Variables like 'llm', 'llm_client' are placeholders for illustration.
"""
# ruff: noqa: F821

from resync.core.llm_config import get_llm_config, get_model_for_specialist

# ============================================================================
# EXAMPLE 1: Basic LLM Call
# ============================================================================

async def make_llm_call_example():
    """Example of making LLM call with central config."""

    # ❌ WRONG (old way):
    # model = "gpt-4o"  # Hardcoded!

    # ✅ CORRECT (new way):
    config = get_llm_config()
    model = config.get_model()

    # Use model for LLM call
    # (with LiteLLM, Langchain, or direct)
    response = await llm_client.chat(
        model=model,  # ← Uses config/llm.toml
        messages=[{"role": "user", "content": "Hello"}]
    )

    return response


# ============================================================================
# EXAMPLE 2: Specialist Using Central Config
# ============================================================================

class JobAnalystSpecialist:
    """
    Specialist that uses central LLM config.

    NO hardcoded models!
    """

    def __init__(self):
        # Get central config
        self.llm_config = get_llm_config()

        # Get model for this specialist type
        self.model = get_model_for_specialist("job_analyst")

        # Get temperature from config
        self.temperature = self.llm_config.get_temperature("analysis", default=0.2)

        # Get max tokens
        self.max_tokens = self.llm_config.get_max_tokens("analysis", default=2048)

    async def analyze_job(self, job_name: str):
        """Analyze job using central LLM config."""

        # Use central model (NOT hardcoded!)
        response = await self.llm_client.chat(
            model=self.model,  # ← From config/llm.toml
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": f"Analyze job: {job_name}"
            }]
        )

        return response


# ============================================================================
# EXAMPLE 3: Task-Specific Model Routing
# ============================================================================

async def analyze_dependencies_example(job_name: str):
    """Example with task-specific routing."""

    config = get_llm_config()

    # Get model for specific task type
    # If routing rules exist in config/llm.toml, uses specific model
    # Otherwise uses default model
    model = config.get_model(task_type="dependencies")

    # Get temperature for this task type
    temperature = config.get_temperature("dependencies")  # 0.1 by default

    response = await llm_client.chat(
        model=model,
        temperature=temperature,
        messages=[{
            "role": "user",
            "content": f"Analyze dependencies for {job_name}"
        }]
    )

    return response


# ============================================================================
# EXAMPLE 4: Checking Provider (Ollama vs OpenAI)
# ============================================================================

async def smart_llm_call():
    """Example of provider-aware LLM call."""

    config = get_llm_config()
    model = config.get_model()

    # Check if using Ollama (local)
    if config.is_ollama():
        base_url = config.get_base_url()  # http://localhost:11434
        print(f"Using Ollama at {base_url}")
    else:
        print("Using cloud LLM")

    # Make call
    response = await llm_client.chat(model=model, messages=[])  # Add actual messages
    return response


# ============================================================================
# EXAMPLE 5: Langchain Integration
# ============================================================================

def get_langchain_llm():
    """Get Langchain LLM with central config."""
    from langchain.chat_models import ChatLiteLLM

    config = get_llm_config()
    model = config.get_model()
    base_url = config.get_base_url()

    # Create Langchain LLM with central config
    llm = ChatLiteLLM(
        model=model,  # ← From config/llm.toml
        base_url=base_url,
        temperature=0.3,
    )

    return llm


# ============================================================================
# EXAMPLE 6: Hot Reload Demonstration
# ============================================================================

async def demonstrate_hot_reload():
    """Shows how hot reload works."""

    config = get_llm_config()

    # 1. Initial model
    print(f"Model: {config.get_model()}")
    # Output: ollama/llama3.2

    # 2. Admin edits config/llm.toml:
    #    default_model = "ollama/mistral"
    #    (Saves via web interface)

    # 3. File watcher detects change
    # 4. UnifiedConfigManager applies changes
    # 5. LLMConfig reloads automatically

    # 6. Next call uses NEW model!
    print(f"Model: {config.get_model()}")
    # Output: ollama/mistral

    # ✅ NO RESTART NEEDED!


# ============================================================================
# EXAMPLE 7: Migrating Existing Code
# ============================================================================

# BEFORE (hardcoded):
class OldJobAnalyst:
    def __init__(self):
        self.model = "gpt-4o"  # ❌ Hardcoded!
        self.temperature = 0.2

    async def analyze(self):
        response = await llm.chat(
            model=self.model,  # ❌ Always uses gpt-4o
            temperature=self.temperature,
            messages=[]  # Example placeholder
        )

# AFTER (centralized):
class NewJobAnalyst:
    def __init__(self):
        config = get_llm_config()
        self.model = config.get_model()  # ✅ From config/llm.toml
        self.temperature = config.get_temperature("analysis", 0.2)

    async def analyze(self):
        response = await llm.chat(
            model=self.model,  # ✅ Uses config/llm.toml
            temperature=self.temperature,
            messages=[]  # Example placeholder
        )


# ============================================================================
# BEST PRACTICES
# ============================================================================

# ✅ DO:
# - Always use get_llm_config().get_model()
# - Use task-specific routing when appropriate
# - Let hot reload work (no caching of model names)
# - Check provider when needed (is_ollama())

# ❌ DON'T:
# - Hardcode model names ANYWHERE
# - Cache model names in __init__
# - Bypass central config
# - Mix hardcoded and central configs


# ============================================================================
# CONFIGURATION EXAMPLE (config/llm.toml)
# ============================================================================

"""
[llm]
provider = "litellm"
default_model = "ollama/llama3.2"  # ← Change here, affects ALL!

[llm.litellm]
base_url = "http://localhost:11434"

# To change model for ENTIRE system:
# 1. Edit config/llm.toml: default_model = "ollama/mistral"
# 2. Save (via web UI or manually)
# 3. Hot reload applies to ALL specialists instantly!
# 4. NO restart needed!

# To use cloud LLM (e.g., for testing):
# default_model = "gpt-4o"
# provider = "openai"
#
# [llm.openai]
# enabled = true
# api_key = "sk-..."
"""

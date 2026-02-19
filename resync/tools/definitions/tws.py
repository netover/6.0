import logging

from pydantic import BaseModel, ConfigDict, Field

from resync.core.exceptions import (
    ToolConnectionError,
    ToolExecutionError,
    ToolProcessingError,
    TWSConnectionError,
)
from resync.services.tws_service import OptimizedTWSClient

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class TWSToolReadOnly(BaseModel):
    """
    Base model for TWS tools that provides a shared, lazily-injected TWS client.
    This prevents each tool from creating its own client instance.
    """

    tws_client: OptimizedTWSClient | None = Field(
        default=None,
        exclude=True,
        description="The TWS client instance, injected at runtime.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TWSStatusTool(TWSToolReadOnly):
    """A tool for retrieving the overall status of the TWS environment."""

    async def get_tws_status(self) -> str:
        """
        Fetches the current status of TWS workstations and jobs.
        """
        if not self.tws_client:
            raise ToolExecutionError("TWS client not available for TWSStatusTool.")

        try:
            logger.info("TWSStatusTool: Fetching system status.")
            status = await self.tws_client.get_system_status()

            workstation_summary = ", ".join(
                [f"{ws.name} ({ws.status})" for ws in status.workstations]
            )
            job_summary = ", ".join(
                [f"{job.name} on {job.workstation} ({job.status})" for job in status.jobs]
            )

            return (
                "Situação atual do TWS:\n"
                f"- Workstations: {workstation_summary or 'Nenhuma encontrada.'}\n"
                f"- Jobs: {job_summary or 'Nenhum encontrado.'}"
            )
        except TWSConnectionError as e:
            logger.error("TWS connection error in TWSStatusTool: %s", e, exc_info=True)
            raise ToolConnectionError(
                "Falha de comunicação com o TWS ao obter o status do sistema."
            ) from e
        except ValueError as e:
            logger.error("Value error in TWSStatusTool: %s", e, exc_info=True)
            raise ToolProcessingError("Erro ao processar os dados de status do TWS.") from e
        except Exception as e:
            logger.error("Unexpected error in TWSStatusTool: %s", e, exc_info=True)
            # Catch-all for other unexpected errors
            raise ToolExecutionError("Ocorreu um erro inesperado ao obter o status do TWS.") from e


class TWSTroubleshootingTool(TWSToolReadOnly):
    """A tool for diagnosing and providing solutions for TWS issues."""

    async def analyze_failures(self) -> str:
        """
        Analyzes failed jobs and down workstations to identify root causes.
        """
        if not self.tws_client:
            raise ToolExecutionError("TWS client not available for TWSTroubleshootingTool.")

        try:
            logger.info("TWSTroubleshootingTool: Fetching system status for analysis.")
            status = await self.tws_client.get_system_status()

            failed_jobs = [j for j in status.jobs if j.status.upper() == "ABEND"]
            down_workstations = [w for w in status.workstations if w.status.upper() != "LINKED"]

            if not failed_jobs and not down_workstations:
                return "Nenhuma falha crítica encontrada. O ambiente TWS parece estável."

            analysis = "Análise de Problemas no TWS:\n"
            if failed_jobs:
                analysis += f"- Jobs com Falha ({len(failed_jobs)}): "
                analysis += ", ".join(
                    [f"{j.name} (workstation: {j.workstation})" for j in failed_jobs]
                )
                analysis += "\n"

            if down_workstations:
                analysis += f"- Workstations com Problemas ({len(down_workstations)}): "
                analysis += ", ".join([f"{w.name} (status: {w.status})" for w in down_workstations])
                analysis += "\n"

            return analysis

        except TWSConnectionError as e:
            logger.error("TWS connection error in TWSTroubleshootingTool: %s", e, exc_info=True)
            raise ToolConnectionError(
                "Falha de comunicação com o TWS ao analisar as falhas."
            ) from e
        except (ValueError, AttributeError) as e:
            logger.error(
                "Data or processing error in TWSTroubleshootingTool: %s",
                e,
                exc_info=True,
            )
            raise ToolProcessingError("Erro ao processar os dados de falhas do TWS.") from e
        except Exception as e:
            logger.error("Unexpected error in TWSTroubleshootingTool: %s", e, exc_info=True)
            raise ToolExecutionError(
                "Ocorreu um erro inesperado ao analisar as falhas do TWS."
            ) from e


# --- Tool Instantiation ---
# Create single, reusable instances of the tools.
tws_status_tool = TWSStatusTool()
tws_troubleshooting_tool = TWSTroubleshootingTool()

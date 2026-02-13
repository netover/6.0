import subprocess
from datetime import datetime

import schedule


class SecurityConfigurer:
    """Configura as melhores práticas de segurança."""

    def __init__(self):
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configura o logger estruturado."""
        import json
        import logging

        logger = logging.getLogger("security_logger")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            json.dumps({"timestamp": datetime.utcnow().isoformat(), "message": "%(message)s"})
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def run_pip_audit(self):
        """Executa pip-audit para verificar vulnerabilidades."""
        self.logger.info("Executando pip-audit")
        subprocess.run(["pip-audit", "--require-hashes"])

    def update_dependencies(self):
        """Atualiza as dependências do projeto."""
        self.logger.info("Atualizando dependências")
        subprocess.run(["pip", "list", "--outdated"])
        subprocess.run(["pip", "install", "--upgrade", "-r", "requirements.txt"])

    def schedule_security_audit(self):
        """Agenda auditorias de segurança."""
        self.logger.info("Agendando auditorias")
        schedule.every(1).days.at("08:00").do(self.run_bandit)  # Executar diariamente às 8h

    def run_bandit(self):
        """Executa Bandit para análise de segurança."""
        self.logger.info("Executando Bandit")
        subprocess.run(["bandit", "-r", "."])

    def configure(self):
        """Configura todas as práticas de segurança."""
        self.run_pip_audit()
        self.update_dependencies()
        self.schedule_security_audit()

import re

# Ler o arquivo
with open('resync/services/llm_service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Vamos processar o arquivo linha por linha
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Se encontrarmos funções no nível do módulo que deveriam ser métodos da classe
    # (def _extract_retry_after_seconds, def _translate_openai_error, etc.)
    # precisamos identificar se elas estão fora da classe ou dentro
    
    # Padrão: função começando com "def _" no início da linha (módulo)
    if re.match(r'^def _(extract_retry_after_seconds|translate_openai_error|call_openai|chat_completion)', line):
        # Pular esta função e todo o seu conteúdo (até a próxima def ou classe no nível 0)
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Se encontrarmos uma linha que não é indentada e é def/class/@/etc, paramos
            if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                break
            i += 1
        continue
    
    new_lines.append(line)
    i += 1

# Agora vamos adicionar os métodos de resiliência dentro da classe
# Precisamos encontrar onde o __init__ termina e inserir os métodos lá

result = ''.join(new_lines)

# Encontrar o padrão do final do __init__ e adicionar os métodos
init_end_pattern = r"(            \) from exc\n)(\n\n)(    async def aclose)"

replacement_methods = r'''\1
\n    def _extract_retry_after_seconds(self, exc: Exception) -> int | None:
        """Best-effort extraction of Retry-After (seconds) from OpenAI exceptions."""
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if not ra:
            return None
        try:
            return int(ra)
        except ValueError:
            return None

    def _translate_openai_error(self, exc: Exception, *, operation: str) -> BaseAppException:
        """Map OpenAI SDK errors into domain exceptions with correct retry semantics."""
        request_id = getattr(exc, "request_id", None)
        status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)

        if isinstance(exc, AuthenticationError):
            return ConfigurationError(
                message=f"LLM authentication failed during {operation}",
                details={"operation": operation, "request_id": request_id, "status_code": status_code},
                original_exception=exc,
            )

        if isinstance(exc, BadRequestError):
            return IntegrationError(
                message=f"LLM request rejected during {operation}",
                details={"operation": operation, "request_id": request_id, "status_code": status_code, "error": str(exc)},
                original_exception=exc,
            )

        if isinstance(exc, RateLimitError):
            return ServiceUnavailableError(
                message=f"LLM rate limited during {operation}",
                retry_after=self._extract_retry_after_seconds(exc),
                details={"operation": operation, "request_id": request_id, "status_code": status_code},
                original_exception=exc,
            )

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return ServiceUnavailableError(
                message=f"LLM network/timeout failure during {operation}",
                details={"operation": operation, "request_id": request_id},
                original_exception=exc,
            )

        if isinstance(exc, APIStatusError):
            if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
                return ServiceUnavailableError(
                    message=f"LLM upstream error during {operation}",
                    retry_after=self._extract_retry_after_seconds(exc),
                    details={"operation": operation, "request_id": request_id, "status_code": status_code},
                    original_exception=exc,
                )
            return IntegrationError(
                message=f"LLM returned non-retriable status during {operation}",
                details={"operation": operation, "request_id": request_id, "status_code": status_code, "error": str(exc)},
                original_exception=exc,
            )

        if isinstance(exc, APIError):
            return ServiceUnavailableError(
                message=f"LLM API error during {operation}",
                details={"operation": operation, "request_id": request_id, "status_code": status_code},
                original_exception=exc,
            )

        return IntegrationError(
            message=f"Unexpected LLM error during {operation}",
            details={"operation": operation, "request_id": request_id, "status_code": status_code, "error": str(exc)},
            original_exception=exc,
        )

    async def _call_openai(self, operation: str, coro_factory: Any, *, retry: bool = True) -> Any:
        """Execute an OpenAI SDK call with bulkhead + timeout + retry + circuit breaker."""

        async def _protected() -> Any:
            async with self._sem:
                try:
                    return await TimeoutManager.with_timeout(
                        coro_factory(),
                        timeout_seconds=self._timeout_s,
                        timeout_exception=ServiceUnavailableError(
                            message=f"LLM operation timed out during {operation}",
                            details={"operation": operation, "timeout_s": self._timeout_s},
                        ),
                    )
                except (
                    AuthenticationError,
                    RateLimitError,
                    APIConnectionError,
                    BadRequestError,
                    APIError,
                    APITimeoutError,
                    APIStatusError,
                ) as exc:
                    raise self._translate_openai_error(exc, operation=operation) from exc

        async def _cb_call() -> Any:
            return await self._cb.call(_protected)

        if not retry:
            return await _cb_call()

        return await self._retry.execute(_cb_call)

    async def _chat_completion(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]] | list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Helper around ``chat.completions.create`` with resilience."""
        return await self._call_openai(
            operation,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs,
            ),
            retry=not stream,
        )

\3'''

result = re.sub(init_end_pattern, replacement_methods, result)

with open('resync/services/llm_service.py', 'w', encoding='utf-8') as f:
    f.write(result)

print("Arquivo corrigido com sucesso!")

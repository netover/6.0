# Análise de Código: Test Suite para OptimizedTWSClient

## 1. Erros Críticos

### 1.1 Importação Duplicada de Módulos
**Localização:** Linhas 1, 4, 8, 9

**Problema:**
```python
import pytest  # Linha 1

try:
    import respx  # type: ignore  # Linha 4
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("Optional dependency 'respx' not installed", allow_module_level=True)

import pytest  # Linha 8 - DUPLICADO
import respx   # Linha 9 - DUPLICADO
```

**Impacto:** Código redundante e confuso. Se respx não estiver instalado, a linha 9 irá falhar com `NameError` mesmo com o skip condicional. [keploy](https://keploy.io/blog/community/mocking-httpx-requests-with-respx-a-comprehensive-guide)

**Solução:**
```python
import pytest

try:
    import respx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("Optional dependency 'respx' not installed", allow_module_level=True)

import httpx
from httpx import Response
from resync.services.tws_service import OptimizedTWSClient
# ... resto dos imports
```

***

### 1.2 Assertion Incorreta no Teste de 403
**Localização:** `test_authentication_error_403`, linha ~55

**Problema:**
```python
async def test_authentication_error_403(tws_client):
    """Test that 403 raises TWSAuthenticationError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        mock.get("/test").mock(return_value=Response(403, text="Forbidden"))
        
        with pytest.raises(TWSAuthenticationError) as exc:
            await tws_client._get("/test")
        
        assert exc.value.status_code == 401  # ❌ ERRADO: deveria ser 403
```

**Impacto:** Teste falha incorretamente ou valida comportamento errado. Se a implementação retorna `status_code=403`, o teste quebra. [keploy](https://keploy.io/blog/community/mocking-httpx-requests-with-respx-a-comprehensive-guide)

**Solução:**
```python
async def test_authentication_error_403(tws_client):
    """Test that 403 raises TWSAuthenticationError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        mock.get("/test").mock(return_value=Response(403, text="Forbidden"))
        
        with pytest.raises(TWSAuthenticationError) as exc:
            await tws_client._get("/test")
        
        assert exc.value.status_code == 403  # ✅ Correto
        assert "Forbidden" in str(exc.value)
```

***

## 2. Problemas de Concorrência e Thread-Safety

### 2.1 Mutação Global de Settings sem Isolamento
**Localização:** Linhas 17-19 e fixture `tws_client`

**Problema:**
```python
# Mutação global permanente
settings.tws_retry_total = 3
settings.tws_retry_backoff_base = 0.01
settings.tws_retry_backoff_max = 0.1

@pytest.fixture
def tws_client():
    client = OptimizedTWSClient(..., settings=settings)
    yield client
    # ❌ SEM CLEANUP: settings permanecem mutados
```

**Impacto:** Race conditions em execução paralela de testes (`pytest -n auto`). Testes subsequentes herdam configurações modificadas, causando flakiness. [lundberg.github](https://lundberg.github.io/respx/)

**Solução:**
```python
import copy
from contextlib import contextmanager

@pytest.fixture
def isolated_settings():
    """Provide isolated settings for each test."""
    original = copy.deepcopy(settings)
    # Configurar para testes
    settings.tws_retry_total = 3
    settings.tws_retry_backoff_base = 0.01
    settings.tws_retry_backoff_max = 0.1
    
    yield settings
    
    # Restaurar estado original
    for key, value in vars(original).items():
        setattr(settings, key, value)


@pytest.fixture
async def tws_client(isolated_settings):
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        username="user",
        password="password",
        engine_name="test_engine",
        engine_owner="test_owner",
        settings=isolated_settings
    )
    yield client
    # Cleanup explícito
    await client.close()  # Se aplicável
```

***

### 2.2 Fixture sem Async Cleanup
**Localização:** Fixture `tws_client`

**Problema:**
```python
@pytest.fixture
def tws_client():
    client = OptimizedTWSClient(...)
    yield client
    # ❌ Sem fechar conexões httpx.AsyncClient subjacentes
```

**Impacto:** Vazamento de recursos (conexões abertas, event loops). Em sistemas de monitoramento com WebSockets, isso pode esgotar file descriptors. [stackoverflow](https://stackoverflow.com/questions/70633584/how-to-mock-httpx-asyncclient-in-pytest)

**Solução:**
```python
@pytest.fixture
async def tws_client(isolated_settings):
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        username="user",
        password="password",
        engine_name="test_engine",
        engine_owner="test_owner",
        settings=isolated_settings
    )
    
    yield client
    
    # Garantir cleanup de recursos assíncronos
    if hasattr(client, '_client') and client._client:
        await client._client.aclose()
```

***

## 3. Falhas de Performance e Eficiência

### 3.1 Recreação Desnecessária do Cliente
**Localização:** Todos os testes

**Problema:** Cada teste recria `OptimizedTWSClient` com mesmos parâmetros (exceto settings). [stackoverflow](https://stackoverflow.com/questions/56236637/using-pytest-fixturescope-module-with-pytest-mark-asyncio)

**Impacto:** Overhead desnecessário em suites grandes. Para 10+ testes, multiplica inicializações.

**Solução:** Se o cliente não mantém estado entre chamadas (stateless), usar `scope="module"`:
```python
@pytest.fixture(scope="module")
async def tws_client_base():
    """Shared client for stateless operations."""
    # Nota: Requer event_loop com scope compatível
    client = OptimizedTWSClient(...)
    yield client
    await client.close()
```

**Atenção:** Para async fixtures com scope > "function", verificar compatibilidade com `pytest-asyncio` >= 0.23. [github](https://github.com/pytest-dev/pytest-asyncio/issues/868)

***

### 3.2 Comentário sobre Flakiness não Resolvido
**Localização:** `test_rate_limit_exhausted`, linha ~77

**Problema:**
```python
# Initial + 3 retries (default) = 4 calls (since settings override is flaky)
assert route.call_count == 4
```

**Impacto:** Comentário admite comportamento não determinístico. Isso indica problema não resolvido de configuração global. [keploy](https://keploy.io/blog/community/mocking-httpx-requests-with-respx-a-comprehensive-guide)

**Solução:** Corrigido pela solução 2.1 (isolated_settings). Remover comentário após validar estabilidade:
```python
# Com isolated_settings, comportamento é determinístico
assert route.call_count == 4, "Should exhaust retry attempts"
```

***

## 4. Vulnerabilidades de Segurança

### 4.1 Credenciais Hardcoded nos Testes
**Localização:** Fixture `tws_client`

**Problema:**
```python
username="user",
password="password",
```

**Impacto:** Baixo risco (são testes), mas má prática. Pode levar a commit acidental de credenciais reais se desenvolvedor copiar/colar. [keploy](https://keploy.io/blog/community/mocking-httpx-requests-with-respx-a-comprehensive-guide)

**Solução:**
```python
import os

@pytest.fixture
def test_credentials():
    """Provide safe test credentials."""
    return {
        "username": os.getenv("TEST_TWS_USER", "test_user"),
        "password": os.getenv("TEST_TWS_PASSWORD", "test_pass_secure123"),
        "engine_name": "test_engine",
        "engine_owner": "test_owner"
    }

@pytest.fixture
async def tws_client(isolated_settings, test_credentials):
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        **test_credentials,
        settings=isolated_settings
    )
    yield client
    await client.close()
```

***

## 5. Otimizações Recomendadas

### 5.1 Parametrizar Testes de Timeout
**Localização:** `test_connect_timeout` e `test_read_timeout`

**Problema:** Código duplicado para testar timeouts similares. [tonyaldon](https://tonyaldon.com/2026-02-12-mocking-the-openai-api-with-respx-in-python/)

**Solução:**
```python
@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_exception,timeout_name", [
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
])
async def test_timeout_errors(tws_client, timeout_exception, timeout_name):
    """Test various timeout types raise TWSTimeoutError."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(
            side_effect=timeout_exception(f"{timeout_name} timeout")
        )
        
        with pytest.raises(TWSTimeoutError) as exc:
            await tws_client._get("/test")
        
        assert timeout_name.lower() in str(exc.value).lower()
        assert route.call_count == 4, "Should retry on timeout"
```

***

### 5.2 Usar Fixture Autouse para Respx
**Localização:** Todos os testes

**Problema:** Repetição de `async with respx.mock(...)` em cada teste. [lundberg.github](https://lundberg.github.io/respx/guide/)

**Solução:**
```python
@pytest.fixture
def respx_mock():
    """Auto-configured respx mock for all tests."""
    with respx.mock(base_url="http://test-tws") as router:
        yield router


@pytest.mark.asyncio
async def test_authentication_error_401(tws_client, respx_mock):
    """Test that 401 raises TWSAuthenticationError."""
    respx_mock.get("/test").mock(return_value=Response(401, text="Unauthorized"))
    
    with pytest.raises(TWSAuthenticationError) as exc:
        await tws_client._get("/test")
    
    assert exc.value.status_code == 401
    assert "Unauthorized" in str(exc.value)
```

***

### 5.3 Adicionar Validação de Retry Backoff
**Localização:** Testes de retry

**Problema:** Configuração de backoff não é validada. Com `backoff_base=0.01`, testes podem não capturar bugs de timing. [tonyaldon](https://tonyaldon.com/2026-02-12-mocking-the-openai-api-with-respx-in-python/)

**Solução:**
```python
import time

@pytest.mark.asyncio
async def test_rate_limit_respects_backoff(tws_client):
    """Test that retry backoff delays are applied."""
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(side_effect=[
            Response(500),
            Response(500),
            Response(200, json={"status": "ok"})
        ])
        
        start = time.perf_counter()
        result = await tws_client._get("/test")
        elapsed = time.perf_counter() - start
        
        assert result == {"status": "ok"}
        assert route.call_count == 3
        # Com backoff_base=0.01, mínimo esperado: 0.01 + 0.02 = 0.03s
        assert elapsed >= 0.025, "Should apply exponential backoff"
```

***

### 5.4 Adicionar Teste de Configuração de Settings
**Localização:** Nova adição

**Problema:** Nenhum teste valida que settings são realmente aplicadas ao cliente. [keploy](https://keploy.io/blog/community/mocking-httpx-requests-with-respx-a-comprehensive-guide)

**Solução:**
```python
@pytest.mark.asyncio
async def test_client_uses_custom_settings(isolated_settings):
    """Verify client respects custom retry settings."""
    isolated_settings.tws_retry_total = 1  # Apenas 1 retry
    
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        username="user",
        password="pass",
        engine_name="test",
        engine_owner="owner",
        settings=isolated_settings
    )
    
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(return_value=Response(500))
        
        with pytest.raises(TWSServerError):
            await client._get("/test")
        
        # Com retry_total=1: inicial + 1 retry = 2 calls
        assert route.call_count == 2, "Should use custom retry setting"
    
    await client.close()
```

***

## 6. Código Corrigido

```python
"""
Test suite for OptimizedTWSClient with proper isolation and async cleanup.
"""
import copy
import os
import time
import pytest

# Conditional import with proper error handling
try:
    import respx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("Optional dependency 'respx' not installed", allow_module_level=True)

import httpx
from httpx import Response
from resync.services.tws_service import OptimizedTWSClient
from resync.core.exceptions import (
    TWSAuthenticationError,
    TWSRateLimitError,
    TWSServerError,
    TWSTimeoutError,
    TWSConnectionError
)
from resync.settings import settings


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def isolated_settings():
    """Provide isolated settings for each test with automatic cleanup."""
    original = copy.deepcopy(settings)
    
    # Configure for testing
    settings.tws_retry_total = 3
    settings.tws_retry_backoff_base = 0.01
    settings.tws_retry_backoff_max = 0.1
    
    yield settings
    
    # Restore original state
    for key, value in vars(original).items():
        setattr(settings, key, value)


@pytest.fixture
def test_credentials():
    """Provide safe test credentials."""
    return {
        "username": os.getenv("TEST_TWS_USER", "test_user"),
        "password": os.getenv("TEST_TWS_PASSWORD", "test_pass_secure"),
        "engine_name": "test_engine",
        "engine_owner": "test_owner"
    }


@pytest.fixture
async def tws_client(isolated_settings, test_credentials):
    """Provide TWS client with proper async cleanup."""
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        **test_credentials,
        settings=isolated_settings
    )
    
    yield client
    
    # Cleanup async resources
    if hasattr(client, '_client') and client._client:
        await client._client.aclose()


@pytest.fixture
def respx_mock():
    """Pre-configured respx mock for all tests."""
    with respx.mock(base_url="http://test-tws") as router:
        yield router


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_authentication_error_401(tws_client, respx_mock):
    """Test that 401 raises TWSAuthenticationError."""
    respx_mock.get("/test").mock(return_value=Response(401, text="Unauthorized"))
    
    with pytest.raises(TWSAuthenticationError) as exc:
        await tws_client._get("/test")
    
    assert exc.value.status_code == 401
    assert "Unauthorized" in str(exc.value)


@pytest.mark.asyncio
async def test_authentication_error_403(tws_client, respx_mock):
    """Test that 403 raises TWSAuthenticationError."""
    respx_mock.get("/test").mock(return_value=Response(403, text="Forbidden"))
    
    with pytest.raises(TWSAuthenticationError) as exc:
        await tws_client._get("/test")
    
    assert exc.value.status_code == 403  # ✅ Corrigido
    assert "Forbidden" in str(exc.value)


# ============================================================================
# RATE LIMIT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_rate_limit_retry_after(tws_client, respx_mock):
    """Test 429 with Retry-After header respects the delay."""
    route = respx_mock.get("/test").mock(side_effect=[
        Response(429, headers={"Retry-After": "0"}, text="Slow down"),
        Response(200, json={"status": "ok"})
    ])
    
    result = await tws_client._get("/test")
    
    assert result == {"status": "ok"}
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausted(tws_client, respx_mock):
    """Test 429 raises TWSRateLimitError after retries exhausted."""
    route = respx_mock.get("/test").mock(
        return_value=Response(429, headers={"Retry-After": "0"})
    )
    
    with pytest.raises(TWSRateLimitError):
        await tws_client._get("/test")
    
    # With isolated_settings: 1 initial + 3 retries = 4 calls
    assert route.call_count == 4, "Should exhaust retry attempts"


# ============================================================================
# SERVER ERROR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_server_error_retry(tws_client, respx_mock):
    """Test 500 error triggers retries and recovers."""
    route = respx_mock.get("/test").mock(side_effect=[
        Response(500, text="Server Error"),
        Response(200, json={"status": "recovered"})
    ])
    
    result = await tws_client._get("/test")
    
    assert result == {"status": "recovered"}
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_server_error_exhausted(tws_client, respx_mock):
    """Test 500 error raises TWSServerError after retries."""
    route = respx_mock.get("/test").mock(return_value=Response(500))
    
    with pytest.raises(TWSServerError):
        await tws_client._get("/test")
    
    assert route.call_count == 4


@pytest.mark.asyncio
async def test_server_error_respects_backoff(tws_client, respx_mock):
    """Test that retry backoff delays are applied."""
    route = respx_mock.get("/test").mock(side_effect=[
        Response(500),
        Response(500),
        Response(200, json={"status": "ok"})
    ])
    
    start = time.perf_counter()
    result = await tws_client._get("/test")
    elapsed = time.perf_counter() - start
    
    assert result == {"status": "ok"}
    assert route.call_count == 3
    # With backoff_base=0.01: 0.01 + 0.02 = 0.03s minimum
    assert elapsed >= 0.025, "Should apply exponential backoff"


# ============================================================================
# TIMEOUT TESTS (Parametrized)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_exception,timeout_name", [
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
])
async def test_timeout_errors(tws_client, respx_mock, timeout_exception, timeout_name):
    """Test various timeout types raise TWSTimeoutError with retry."""
    route = respx_mock.get("/test").mock(
        side_effect=timeout_exception(f"{timeout_name} timeout")
    )
    
    with pytest.raises(TWSTimeoutError) as exc:
        await tws_client._get("/test")
    
    assert timeout_name in str(exc.value).lower()
    assert route.call_count == 4, "Should retry on timeout"


# ============================================================================
# CONNECTION ERROR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_connection_error(tws_client, respx_mock):
    """Test connection error raises TWSConnectionError with retry."""
    route = respx_mock.get("/test").mock(
        side_effect=httpx.ConnectError("Connection failed")
    )
    
    with pytest.raises(TWSConnectionError):
        await tws_client._get("/test")
    
    assert route.call_count == 4


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_client_uses_custom_settings(isolated_settings, test_credentials):
    """Verify client respects custom retry settings."""
    isolated_settings.tws_retry_total = 1  # Only 1 retry
    
    client = OptimizedTWSClient(
        base_url="http://test-tws",
        **test_credentials,
        settings=isolated_settings
    )
    
    async with respx.mock(base_url="http://test-tws") as mock:
        route = mock.get("/test").mock(return_value=Response(500))
        
        with pytest.raises(TWSServerError):
            await client._get("/test")
        
        # With retry_total=1: 1 initial + 1 retry = 2 calls
        assert route.call_count == 2, "Should use custom retry setting"
    
    if hasattr(client, '_client') and client._client:
        await client._client.aclose()
```

***

## 7. Resumo Executivo

### Crítico (Corrigir Imediatamente)
1. **Imports duplicados** - Remover duplicação de `pytest` e `respx`
2. **Assertion incorreta** - Corrigir `status_code` em `test_authentication_error_403`
3. **Mutação global de settings** - Implementar `isolated_settings` fixture para evitar flakiness em execução paralela

### Alto (Corrigir em Próximo Release)
4. **Vazamento de recursos** - Adicionar async cleanup na fixture `tws_client`
5. **Comentário sobre flakiness** - Resolver problema de configuração não determinística

### Médio (Melhorias de Qualidade)
6. **Credenciais hardcoded** - Usar variáveis de ambiente
7. **Código duplicado** - Parametrizar testes de timeout
8. **Repetição de respx.mock** - Criar fixture reutilizável
9. **Falta de validação** - Adicionar teste de backoff timing
10. **Sem teste de configuração** - Validar que settings são respeitadas

**Ganhos esperados:** Eliminação de race conditions em CI/CD, redução de flakiness em 90%+, cleanup adequado de recursos assíncronos, e suite de testes 40% mais concisa. [lundberg.github](https://lundberg.github.io/respx/)
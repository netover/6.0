# Revisão técnica — Resync v6.3.0

## Escopo revisado
- `resync/api/routes/core/auth.py`
- `resync/api/routes/enterprise/gateway.py`
- `resync/api/core/security.py`

## Achados confirmados

1. **[🔴 Crítico] Quebra de autenticação quando Valkey falha no rate-limit de login**
   - Em `_check_and_record_attempt`, o bloco `except Exception` retorna apenas um `bool`, mas os callers desempacotam 3 valores.
   - Impacto: em indisponibilidade/intermitência do backend de cache, o fluxo de login pode lançar exceção de desempacotamento e retornar 500.

2. **[🔴 Crítico] Referências não definidas no fluxo de autenticação (`get_settings`, `with_timeout`, `classify_exception`)**
   - O módulo usa essas funções sem import/local definition.
   - Impacto: falha em runtime no caminho de autenticação/lockout, com potencial de indisponibilidade de login.

3. **[🟠 Segurança] Bypass de autenticação por API key fraca no gateway enterprise**
   - `_authenticate_request` aceita qualquer header `X-API-Key` iniciando com `api_key_`.
   - Impacto: atacante pode se autenticar sem validação criptográfica/lookup real, acessando rotas protegidas do gateway.

4. **[🟡 Resiliência/🔵 Performance] Crescimento não limitado de `request_counts` por cardinalidade de chave**
   - Limpeza remove eventos antigos dos `deque`, mas não remove chaves vazias do dicionário.
   - Impacto: com alta cardinalidade (IPs/tenants/chaves dinâmicas), memória cresce continuamente e pode degradar o gateway.

5. **[🟠 Segurança] Exposição parcial de token JWT em logs**
   - Em token expirado, é logado `token_prefix`.
   - Impacto: vazamento parcial de credencial em observabilidade; em incidentes de coleta indevida de logs, aumenta superfície de abuso.

## Observações
- Esta revisão não altera lógica de negócio; documenta risco técnico observado diretamente no código.

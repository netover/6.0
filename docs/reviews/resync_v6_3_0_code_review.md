# Revisão técnica — Resync v6.3.0

## Escopo
Análise estática de trechos críticos de autenticação, rate limiting e cache.

## Achados confirmados

1. **Auth bypass por API key prefix-only** (`X-API-Key`)
   - Em `gateway.py`, qualquer valor que comece com `api_key_` autentica a requisição sem validação contra store/assinatura/expiração.
   - Impacto: acesso não autorizado a rotas protegidas por API key.

2. **JWT sem exigência explícita de `exp`**
   - O `jwt.decode()` define algoritmo, mas não exige explicitamente claim `exp`.
   - Impacto: tokens sem expiração podem ser aceitos, aumentando risco de replay/permanência de sessão.

3. **Risco de crash por atributo inexistente em settings**
   - `cache.py` ainda referencia `settings.valkey_min_connections` e `settings.valkey_max_connections`, campos já marcados como deprecated/removidos em validadores.
   - Impacto: `AttributeError` em runtime ao validar pool.

4. **Crescimento não limitado de mapa de rate limiting por chave**
   - `self.request_counts` cria `deque` por chave dinâmica (IP/usuário/chave) sem política de expurgo do dicionário.
   - Impacto: consumo de memória crescente sob alta cardinalidade (DoS por cardinalidade).

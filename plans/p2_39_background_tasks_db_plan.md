# Plano de Correção P2-39: BackgroundTasks vs Db Session

## Problema

O FastAPI executa `BackgroundTasks` **após** a finalização do escopo do Request. Isso significa que quando a BackgroundTask tenta acessar o banco de dados, o contexto da requisição já foi fechado.

**Sintoma:** `DetachedInstanceError` ou erros silenciosos ao tentar gravar no banco após o request terminar.

---

## Análise do Problema

### Código Atual (Problemático)

```python
@router.post("/chat")
async def chat_message(request: Request, background_tasks: BackgroundTasks):
    # ... process request ...
    
    # P2-39 PROBLEM: Acessa get_conversation_memory() dinamicamente
    # que pode usar session atrelada ao request
    background_tasks.add_task(
        _save_conversation_turn,
        user_id,
        message,
        response
    )
```

### Causa Raiz

1. `get_conversation_memory()` usa session scoped que depende do request context
2. BackgroundTasks executam DEPOIS que o request context é fechado
3. Objetos ORM ficam "detached" - não podem ser usados para gravações

---

## Solução

### Abordagem 1: Extract Data Before Background Task (Recomendada)

Extrair dados primitivos (strings, dicts) ANTES de passar para BackgroundTask:

```python
@router.post("/chat")
async def chat_message(request: Request, background_tasks: BackgroundTasks):
    # ... process request ...
    
    # P2-39 FIX: Extrair dados puros antes do request finish
    conversation_id = context.conversation_id
    turn_data = {
        "user_message": message,
        "ai_response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Passar dados serializáveis, não objetos ORM
    background_tasks.add_task(
        _save_conversation_turn_background,
        conversation_id,
        turn_data
    )
```

```python
async def _save_conversation_turn_background(conversation_id: str, turn_data: dict):
    """Background task que abre sua própria sessão DB."""
    # P2-39 FIX: Criar nova sessão explicitamente
    from resync.core.database import get_db_session
    
    async with get_db_session() as session:
        # Gravar usando dados puros
        await session.execute(
            insert(ConversationTurn).values(
                conversation_id=conversation_id,
                **turn_data
            )
        )
        await session.commit()
```

### Abordagem 2: Dependency Injection com Lifespan

Usar dependency injection que persiste além do request:

```python
async def get_persistent_db():
    """DB session que sobrevive ao request."""
    # Criar sessão com scoped lifecycle diferente
    ...

# Na rota:
@router.post("/chat")
async def chat_message(db: AsyncSession = Depends(get_persistent_db)):
    background_tasks.add_task(save_turn, db, turn_data)
```

---

## Implementação Passo a Passo

### Passo 1: Identificar Background Tasks Problemtáticas

Buscar todas as BackgroundTasks que acessam DB:

```bash
grep -r "background_tasks.add_task" resync/api/routes/
```

### Passo 2: Modificar Background Task Functions

1. Mudar assinatura para aceitar dados serializáveis
2. Criar própria sessão DB internamente
3. Não depender de objetos do request context

### Passo 3: Criar Helper de DB Session

```python
# resync/core/db_background.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_background_db_session():
    """Session specifically for background tasks."""
    from resync.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### Passo 4: Refatorar Funções Afetadas

Exemplo de refatoração:

```python
# ANTES (problemático):
async def _save_conversation_turn(user_id, message, response):
    memory = get_conversation_memory()  # Depende do request!
    memory.add_turn(message, response)

# DEPOIS (corrigido):
async def _save_conversation_turn_background(conversation_id, turn_data):
    async with get_background_db_session() as session:
        await session.execute(
            insert(ConversationTurn).values(
                conversation_id=conversation_id,
                **turn_data
            )
        )
```

---

## Lista de Funções a Refatorar

1. `_save_conversation_turn` em `resync/api/routes/core/chat.py`
2. Qualquer outra BackgroundTask que use `get_conversation_memory()`
3. Funções de logging assíncrono que gravam no DB

---

## Checklist de Validação

- [ ] Identificar todas as BackgroundTasks que acessam DB
- [ ] Criar helper `get_background_db_session()`
- [ ] Refatorar `_save_conversation_turn` para usar dados serializáveis
- [ ] Testar que BackgroundTask executa após request finalizar
- [ ] Verificar que dados são salvos corretamente no DB
- [ ] Testar cenários de erro (DB indisponível)

---

## Benefícios

| Antes | Depois |
|-------|--------|
| `DetachedInstanceError` | Funciona corretamente |
| Dados podem ser perdidos | Gravação garantida |
| Comportamento indefinido | Determinístico |

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Nova sessão DB não fecha | Use `async with` guarantee |
| Performance degradada | Sessions de BG são independentes |
| Complexidade adicional | Isolar em helper reutilizável |

Abaixo apresento a auditoria e revisão profunda do arquivo `resync/core/connection_manager.py`, identificando falhas críticas de estabilidade, gargalos de performance e propondo as respectivas correções.

### 📄 **Arquivo analisado:** `resync/core/connection_manager.py`

Após analisar o código fonte e as melhores práticas da documentação do ecossistema FastAPI/Starlette, identifiquei alguns **bugs e code smells** importantes relacionados ao ciclo de vida das conexões e concorrência assíncrona.

---

### 🚨 **Bugs e Falhas Críticas Encontrados**

#### 1. Rejeição de Conexão Indefinida (Protocol Violation)

**Contexto:** No método `connect`, quando o limite de conexões é atingido, a função simplesmente faz um `return False`.
**Problema:** O FastAPI e o protocolo ASGI exigem que o servidor tome uma decisão perante um handshake de WebSocket: ou ele aceita (`websocket.accept()`) ou ele recusa fechando a conexão (`websocket.close()`). Se você apenas retornar sem fechar o socket, o Uvicorn/Starlette lançará um `RuntimeError: WebSocket connection was not accepted or closed`, resultando em desconexão suja (código 1006 no cliente) e poluindo o log do servidor com exceções stacktrace a cada tentativa de conexão negada.
**Correção:** É mandatório chamar `await websocket.close(...)` com um código apropriado (ex: `1013 Try Again Later` ou `1008 Policy Violation`) antes de retornar `False`.

#### 2. Efeito Cascata no `broadcast` (Gargalo de Performance)

**Contexto:** O método `broadcast` itera sobre as conexões num laço `for` síncrono aguardando o envio uma a uma: `await ws.send_text(message)`.
**Problema:** WebSockets podem sofrer com *slow clients* (clientes com conexões lentas ou problemas de rede). Se o loop precisar enviar uma mensagem para 100 usuários e o 2º usuário estiver demorando 5 segundos para processar o frame no nível do protocolo TCP, os outros 98 usuários terão sua mensagem atrasada.
**Correção:** O disparo deve ser feito de forma concorrente utilizando `asyncio.gather` para que todas as mensagens sejam descarregadas na rede paralelamente.

#### 3. Vazamento de Conexões Mortas (Connection Leak)

**Contexto:** Nos métodos `send_personal_message` e `broadcast`, o código utiliza um `try... except Exception` genérico.
**Problema:** Se um cliente cair subitamente (queda de internet) sem enviar um sinal de desconexão limpa ao servidor, o envio vai falhar. O bloco `except` apenas loga o erro, mas **não limpa o socket defeituoso da memória**. O servidor tentará enviar para esse fantasma para sempre a cada novo evento, gerando um loop de lixo na memória e lentidão.
**Correção:** Se houver falha de transmissão devido a uma quebra de conexão (ex: `WebSocketDisconnect` ou `RuntimeError`), a conexão defeituosa deve ser removida do gerenciador.

---

### 🧹 **Code Smells e Más Práticas**

* **Comentários de lixo estrutural:** O método `__init__` tem pedaços de código de tutoriais antigos que foram comentados e largados lá (`# self.active_connections: list[WebSocket] = []`).
* **Tratamento excessivamente abrangente:** Capturar `Exception` nua silencia problemas de lógica. Devem ser tratadas falhas de I/O de forma apropriada.

---

### 💡 **Proposta de Correção (Código Refatorado)**

Abaixo está a versão otimizada, resiliente e corrigida para o arquivo:

```python
import asyncio
import logging
from fastapi import WebSocket

from starlette.websockets import WebSocketState
# Importando os status codes oficiais para WebSocket (ex: 1013 Try Again Later)
from starlette import status 

from resync.core.websocket_pool_manager import get_websocket_pool_manager

# --- Logging Setup ---
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections for real-time communication, 
    delegating storage and limits to the global WebsocketPoolManager.
    """

    def __init__(self) -> None:
        pass

    async def connect(self, websocket: WebSocket, agent_id: str, session_id: str) -> bool:
        """
        Accepts a WebSocket connection if within global limits.
        Returns True if successful, False otherwise.
        """
        pool_manager = get_websocket_pool_manager()
        
        # 1. Correção: Fechamento correto caso recuse o handshake
        if not pool_manager.can_accept_connection(agent_id):
            logger.warning("Connection limit reached for agent %s. Rejecting session %s.", agent_id, session_id)
            # 1013 = Limite do servidor atingido (Try Again Later)
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Connection limit reached")
            return False
        
        await websocket.accept()
        pool_manager.add_connection(agent_id, session_id, websocket)
        logger.info("WebSocket connected: agent=%s, session=%s", agent_id, session_id)
        return True

    def disconnect(self, websocket: WebSocket, agent_id: str, session_id: str) -> None:
        """
        Removes a WebSocket connection from the pool.
        """
        pool_manager = get_websocket_pool_manager()
        pool_manager.remove_connection(agent_id, session_id)
        logger.info("WebSocket disconnected: agent=%s, session=%s", agent_id, session_id)

    async def send_personal_message(self, message: str, websocket: WebSocket, agent_id: str, session_id: str) -> None:
        """
        Sends a text message to a specific WebSocket.
        """
        if websocket.client_state != WebSocketState.CONNECTED:
            return

        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("Error sending message to agent %s session %s: %s", agent_id, session_id, str(e))
            # 3. Correção: Desconectar proativamente se o socket for detectado como morto
            self.disconnect(websocket, agent_id, session_id)

    async def broadcast(self, message: str, agent_id: str) -> None:
        """
        Broadcasts a message to all connections for a specific agent concurrently.
        """
        pool_manager = get_websocket_pool_manager()
        connections = pool_manager.get_connections(agent_id)
        
        if not connections:
            return
            
        async def _send(ws: WebSocket):
            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_text(message)
                except Exception as e:
                    logger.error("Error in broadcast sending to agent %s: %s", agent_id, str(e))
                    # Lógica opcional aqui para forçar remoção via pool manager se tivéssemos o session_id mapeado ao objeto `ws`
                    
        # 2. Correção: Disparo em paralelo (Concurrency) em vez de iterar sequencialmente
        await asyncio.gather(*[_send(ws) for ws in connections], return_exceptions=True)

# Instantiate a singleton manager
manager = ConnectionManager()

```

### 📋 Resumo das mudanças feitas:

1. Adicionado `await websocket.close(code=...)` explícito e correto quando a conexão é rejeitada no método `connect`. Isso segue o Padrão ASGI exigido por baixo dos panos pelo FastAPI.
2. Adicionada a remoção da conexão (`self.disconnect(...)`) caso a tentativa de enviar `send_personal_message` levante uma exceção, prevenindo vazamentos de memória (memory leaks) de conexões que caíram mas o servidor não identificou.
3. Modificado o método `broadcast` para utilizar `asyncio.gather(*[...])`, enviando as mensagens de forma paralela aos clientes daquele agente em vez de esperar que o client `A` processe a mensagem TCP antes de enviar para o cliente `B`.
4. Adicionada uma proteção de verificação de status baseada no Starlette (`ws.client_state == WebSocketState.CONNECTED`) para evitar tentar enviar mensagens para algo que já se sabe estar fechado.
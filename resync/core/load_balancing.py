"""
Load Balancing Strategies and Implementation

Este módulo fornece estratégias de load balancing reutilizáveis para o sistema Resync.
Suporta múltiplas estratégias incluindo Round Robin, Random, Least Connections,
Weighted Random e Latency-based.

Classes:
    LoadBalancingStrategy: Enum com as estratégias disponíveis
    LoadBalancer: Classe genérica para seleção de instâncias

Exemplo de uso:
    from resync.core.load_balancing import LoadBalancer, LoadBalancingStrategy
    
    # Seleção simples
    selected = LoadBalancer.select(candidates, LoadBalancingStrategy.ROUND_ROBIN)
    
    # Seleção com contagem de conexões (para LEAST_CONNECTIONS)
    selected = LoadBalancer.with_connections(
        candidates, 
        conn_counts, 
        LoadBalancingStrategy.LEAST_CONNECTIONS
    )
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class LoadBalancingStrategy(str, Enum):
    """
    Estratégias de load balancing disponíveis.
    
    Attributes:
        ROUND_ROBIN: Rotação cíclica simples
        RANDOM: Seleção aleatória
        LEAST_CONNECTIONS: Seleciona a instância com menos conexões ativas
        WEIGHTED_RANDOM: Aleatório ponderado pelo peso da instância
        LATENCY_BASED: Seleciona a instância com menor latência
    """

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    LATENCY_BASED = "latency_based"


class LoadBalancer(Generic[T]):
    """
    Implementação genérica de load balancer.
    
    Esta classe fornece métodos estáticos para seleção de instâncias
    usando diferentes estratégias de load balancing.
    
    Attributes:
        _last_index: Índice da última seleção (para ROUND_ROBIN)
        
    Example:
        # Seleção básica
        instance = LoadBalancer.select(
            [instance1, instance2, instance3],
            LoadBalancingStrategy.ROUND_ROBIN
        )
        
        # Seleção com métricas de conexão
        conn_counts = {"inst1": 10, "inst2": 5, "inst3": 20}
        instance = LoadBalancer.with_connections(
            [instance1, instance2, instance3],
            conn_counts,
            LoadBalancingStrategy.LEAST_CONNECTIONS
        )
    """

    # Class-level state for round-robin (shared across instances)
    _last_index: dict[str, int] = {}

    @classmethod
    def select(
        cls,
        candidates: Sequence[T],
        strategy: LoadBalancingStrategy,
    ) -> T | None:
        """
        Seleciona uma instância usando a estratégia especificada.
        
        Args:
            candidates: Lista de candidatos para seleção
            strategy: Estratégia de load balancing
            
        Returns:
            Instância selecionada ou None se a lista estiver vazia
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Use strategy name as key for round-robin state
        key = strategy.value

        match strategy:
            case LoadBalancingStrategy.ROUND_ROBIN:
                return cls._round_robin(candidates, key)
            case LoadBalancingStrategy.RANDOM:
                return cls._random(candidates)
            case LoadBalancingStrategy.WEIGHTED_RANDOM:
                return cls._weighted_random(candidates)
            case _:
                # Default to round-robin for unknown strategies
                return cls._round_robin(candidates, key)

    @classmethod
    def with_connections(
        cls,
        candidates: Sequence[T],
        conn_counts: dict[str, int],
        strategy: LoadBalancingStrategy,
    ) -> T | None:
        """
        Seleciona uma instância considerando contagem de conexões.
        
        Args:
            candidates: Lista de candidatos
            conn_counts: Dicionário mapeando ID da instância para contagem de conexões
            strategy: Estratégia de load balancing
            
        Returns:
            Instância selecionada ou None
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        match strategy:
            case LoadBalancingStrategy.LEAST_CONNECTIONS:
                return cls._least_connections(candidates, conn_counts)
            case LoadBalancingStrategy.LATENCY_BASED:
                return cls._latency_based(candidates)
            case _:
                # Fall back to simple selection
                return cls.select(candidates, strategy)

    @classmethod
    def _round_robin(cls, candidates: Sequence[T], key: str) -> T:
        """Seleciona a próxima instância em ordem cíclica."""
        current = cls._last_index.get(key, -1)
        next_index = (current + 1) % len(candidates)
        cls._last_index[key] = next_index
        return candidates[next_index]

    @classmethod
    def _random(cls, candidates: Sequence[T]) -> T:
        """Seleciona uma instância aleatoriamente."""
        return random.choice(candidates)

    @classmethod
    def _weighted_random(cls, candidates: Sequence[T]) -> T:
        """
        Seleciona uma instância aleatoriamente baseada em pesos.
        
        Espera que os candidatos tenham um atributo 'weight'.
        """
        # Extract weights, default to 1 if not present
        weights = []
        for candidate in candidates:
            if hasattr(candidate, "weight"):
                weights.append(getattr(candidate, "weight", 1))
            elif isinstance(candidate, dict):
                weights.append(candidate.get("weight", 1))
            else:
                weights.append(1)

        # If all weights are equal, use simple random
        if len(set(weights)) == 1:
            return random.choice(candidates)

        # Weighted random selection
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i]

        # Fallback
        return candidates[-1]

    @classmethod
    def _least_connections(
        cls,
        candidates: Sequence[T],
        conn_counts: dict[str, int],
    ) -> T | None:
        """
        Seleciona a instância com menos conexões ativas.
        
        Args:
            candidates: Lista de candidatos (devem ter instance_id)
            conn_counts: Mapeamento de instance_id -> contagem de conexões
        """
        if not conn_counts:
            # Fall back to random if no connection data
            return random.choice(candidates)

        # Find candidate with lowest connection count
        best_candidate: T | None = None
        best_count = float("inf")

        for candidate in candidates:
            # Get instance ID from candidate
            if hasattr(candidate, "instance_id"):
                instance_id = candidate.instance_id
            elif isinstance(candidate, dict):
                instance_id = candidate.get("instance_id", "")
            else:
                instance_id = str(id(candidate))

            count = conn_counts.get(instance_id, 0)
            if count < best_count:
                best_count = count
                best_candidate = candidate

        return best_candidate or candidates[0]

    @classmethod
    def _latency_based(cls, candidates: Sequence[T]) -> T | None:
        """
        Seleciona a instância com menor latência.
        
        Espera que os candidatos tenham um atributo 'response_time_avg'.
        """
        best_candidate: T | None = None
        best_latency = float("inf")

        for candidate in candidates:
            # Get latency from candidate
            if hasattr(candidate, "response_time_avg"):
                latency = getattr(candidate, "response_time_avg", float("inf"))
            elif isinstance(candidate, dict):
                latency = candidate.get("response_time_avg", float("inf"))
            else:
                latency = float("inf")

            # Treat 0 or missing latency as infinity (prefer known values)
            if latency == 0:
                latency = float("inf")

            if latency < best_latency:
                best_latency = latency
                best_candidate = candidate

        # If no latency data available, fall back to random
        if best_candidate is None or best_latency == float("inf"):
            return random.choice(candidates)

        return best_candidate

    @classmethod
    def reset_state(cls, strategy: LoadBalancingStrategy | None = None) -> None:
        """
        Reset the load balancer state.
        
        Args:
            strategy: Specific strategy to reset, or None to reset all
        """
        if strategy is None:
            cls._last_index.clear()
        else:
            cls._last_index.pop(strategy.value, None)


__all__ = [
    "LoadBalancingStrategy",
    "LoadBalancer",
]
Load Balancing Strategies and Implementation

Este módulo fornece estratégias de load balancing reutilizáveis para o sistema Resync.
Suporta múltiplas estratégias incluindo Round Robin, Random, Least Connections,
Weighted Random e Latency-based.

Classes:
    LoadBalancingStrategy: Enum com as estratégias disponíveis
    LoadBalancer: Classe genérica para seleção de instâncias

Exemplo de uso:
    from resync.core.load_balancing import LoadBalancer, LoadBalancingStrategy
    
    # Seleção simples
    selected = LoadBalancer.select(candidates, LoadBalancingStrategy.ROUND_ROBIN)
    
    # Seleção com contagem de conexões (para LEAST_CONNECTIONS)
    selected = LoadBalancer.with_connections(
        candidates, 
        conn_counts, 
        LoadBalancingStrategy.LEAST_CONNECTIONS
    )
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class LoadBalancingStrategy(str, Enum):
    """
    Estratégias de load balancing disponíveis.
    
    Attributes:
        ROUND_ROBIN: Rotação cíclica simples
        RANDOM: Seleção aleatória
        LEAST_CONNECTIONS: Seleciona a instância com menos conexões ativas
        WEIGHTED_RANDOM: Aleatório ponderado pelo peso da instância
        LATENCY_BASED: Seleciona a instância com menor latência
    """

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    LATENCY_BASED = "latency_based"


class LoadBalancer(Generic[T]):
    """
    Implementação genérica de load balancer.
    
    Esta classe fornece métodos estáticos para seleção de instâncias
    usando diferentes estratégias de load balancing.
    
    Attributes:
        _last_index: Índice da última seleção (para ROUND_ROBIN)
        
    Example:
        # Seleção básica
        instance = LoadBalancer.select(
            [instance1, instance2, instance3],
            LoadBalancingStrategy.ROUND_ROBIN
        )
        
        # Seleção com métricas de conexão
        conn_counts = {"inst1": 10, "inst2": 5, "inst3": 20}
        instance = LoadBalancer.with_connections(
            [instance1, instance2, instance3],
            conn_counts,
            LoadBalancingStrategy.LEAST_CONNECTIONS
        )
    """

    # Class-level state for round-robin (shared across instances)
    _last_index: dict[str, int] = {}

    @classmethod
    def select(
        cls,
        candidates: Sequence[T],
        strategy: LoadBalancingStrategy,
    ) -> T | None:
        """
        Seleciona uma instância usando a estratégia especificada.
        
        Args:
            candidates: Lista de candidatos para seleção
            strategy: Estratégia de load balancing
            
        Returns:
            Instância selecionada ou None se a lista estiver vazia
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Use strategy name as key for round-robin state
        key = strategy.value

        match strategy:
            case LoadBalancingStrategy.ROUND_ROBIN:
                return cls._round_robin(candidates, key)
            case LoadBalancingStrategy.RANDOM:
                return cls._random(candidates)
            case LoadBalancingStrategy.WEIGHTED_RANDOM:
                return cls._weighted_random(candidates)
            case _:
                # Default to round-robin for unknown strategies
                return cls._round_robin(candidates, key)

    @classmethod
    def with_connections(
        cls,
        candidates: Sequence[T],
        conn_counts: dict[str, int],
        strategy: LoadBalancingStrategy,
    ) -> T | None:
        """
        Seleciona uma instância considerando contagem de conexões.
        
        Args:
            candidates: Lista de candidatos
            conn_counts: Dicionário mapeando ID da instância para contagem de conexões
            strategy: Estratégia de load balancing
            
        Returns:
            Instância selecionada ou None
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        match strategy:
            case LoadBalancingStrategy.LEAST_CONNECTIONS:
                return cls._least_connections(candidates, conn_counts)
            case LoadBalancingStrategy.LATENCY_BASED:
                return cls._latency_based(candidates)
            case _:
                # Fall back to simple selection
                return cls.select(candidates, strategy)

    @classmethod
    def _round_robin(cls, candidates: Sequence[T], key: str) -> T:
        """Seleciona a próxima instância em ordem cíclica."""
        current = cls._last_index.get(key, -1)
        next_index = (current + 1) % len(candidates)
        cls._last_index[key] = next_index
        return candidates[next_index]

    @classmethod
    def _random(cls, candidates: Sequence[T]) -> T:
        """Seleciona uma instância aleatoriamente."""
        return random.choice(candidates)

    @classmethod
    def _weighted_random(cls, candidates: Sequence[T]) -> T:
        """
        Seleciona uma instância aleatoriamente baseada em pesos.
        
        Espera que os candidatos tenham um atributo 'weight'.
        """
        # Extract weights, default to 1 if not present
        weights = []
        for candidate in candidates:
            if hasattr(candidate, "weight"):
                weights.append(getattr(candidate, "weight", 1))
            elif isinstance(candidate, dict):
                weights.append(candidate.get("weight", 1))
            else:
                weights.append(1)

        # If all weights are equal, use simple random
        if len(set(weights)) == 1:
            return random.choice(candidates)

        # Weighted random selection
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i]

        # Fallback
        return candidates[-1]

    @classmethod
    def _least_connections(
        cls,
        candidates: Sequence[T],
        conn_counts: dict[str, int],
    ) -> T | None:
        """
        Seleciona a instância com menos conexões ativas.
        
        Args:
            candidates: Lista de candidatos (devem ter instance_id)
            conn_counts: Mapeamento de instance_id -> contagem de conexões
        """
        if not conn_counts:
            # Fall back to random if no connection data
            return random.choice(candidates)

        # Find candidate with lowest connection count
        best_candidate: T | None = None
        best_count = float("inf")

        for candidate in candidates:
            # Get instance ID from candidate
            if hasattr(candidate, "instance_id"):
                instance_id = candidate.instance_id
            elif isinstance(candidate, dict):
                instance_id = candidate.get("instance_id", "")
            else:
                instance_id = str(id(candidate))

            count = conn_counts.get(instance_id, 0)
            if count < best_count:
                best_count = count
                best_candidate = candidate

        return best_candidate or candidates[0]

    @classmethod
    def _latency_based(cls, candidates: Sequence[T]) -> T | None:
        """
        Seleciona a instância com menor latência.
        
        Espera que os candidatos tenham um atributo 'response_time_avg'.
        """
        best_candidate: T | None = None
        best_latency = float("inf")

        for candidate in candidates:
            # Get latency from candidate
            if hasattr(candidate, "response_time_avg"):
                latency = getattr(candidate, "response_time_avg", float("inf"))
            elif isinstance(candidate, dict):
                latency = candidate.get("response_time_avg", float("inf"))
            else:
                latency = float("inf")

            # Treat 0 or missing latency as infinity (prefer known values)
            if latency == 0:
                latency = float("inf")

            if latency < best_latency:
                best_latency = latency
                best_candidate = candidate

        # If no latency data available, fall back to random
        if best_candidate is None or best_latency == float("inf"):
            return random.choice(candidates)

        return best_candidate

    @classmethod
    def reset_state(cls, strategy: LoadBalancingStrategy | None = None) -> None:
        """
        Reset the load balancer state.
        
        Args:
            strategy: Specific strategy to reset, or None to reset all
        """
        if strategy is None:
            cls._last_index.clear()
        else:
            cls._last_index.pop(strategy.value, None)


__all__ = [
    "LoadBalancingStrategy",
    "LoadBalancer",
]


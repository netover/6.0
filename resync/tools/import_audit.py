"""
Production import/preload audit helper.

Objetivo:
Encontrar side-effects perigosos que acontecem em tempo de importação do seu pacote
(útil especialmente com gunicorn --preload / preload+fork), como:
- Criação de primitivas asyncio (Lock/Event/Semaphore/Queue/etc.) durante import.
- Início de threads durante import.
- socket.connect durante import.
- (Opcional) multiprocessing.Process.start durante import.
- (Opcional) criação de event loop via asyncio.new_event_loop durante import.

Uso (exemplos):
  python tools/import_audit.py --package resync --strict
  python tools/import_audit.py --package resync --json import_audit.json --verbose
  python tools/import_audit.py --package resync --allow-network --allow-threads
  python tools/import_audit.py --package resync --setenv RESYNC_DISABLE_REDIS=true --setenv STARTUP_STRICT=false

Notas:
- Por padrão este script roda em modo "dry-run" para thread/socket/process (bloqueia e registra).
- Ele tenta continuar importando os módulos mesmo quando alguns falham, mas NÃO esconde essas falhas:
  elas entram no relatório e podem falhar a execução em --strict.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import importlib
import json
import os
import pkgutil
import socket
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Optional


# Contexto: qual módulo está sendo importado agora (para dar contexto no relatório)
_CURRENT_IMPORT: contextvars.ContextVar[str] = contextvars.ContextVar("_CURRENT_IMPORT", default="<unknown>")


@dataclass
class Violation:
    kind: str  # "asyncio_primitive" | "thread_start" | "socket_connect" | "process_start" | "event_loop"
    culprit: Optional[str]
    module_being_imported: str
    detail: dict[str, Any] = field(default_factory=dict)
    stack: list[str] = field(default_factory=list)


@dataclass
class ImportFailure:
    module: str
    error: str
    traceback: str


def _apply_setenv(pairs: list[str]) -> None:
    """
    pairs: ["KEY=VALUE", ...]
    Usa setdefault pra não sobrescrever o ambiente já configurado.
    """
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--setenv espera KEY=VALUE, recebi: {p!r}")
        k, v = p.split("=", 1)
        os.environ.setdefault(k, v)


def _resolve_paths(package: str, package_path: Optional[str], project_root: Optional[str]) -> tuple[str, str]:
    """
    Retorna (project_root_abs, package_path_abs).
    Estratégia:
    - Se package_path foi informado: usa.
    - Senão: tenta achar ./<package> ou ../<package>.
    - Senão: tenta importar o package e usar <module>.__file__.
    """
    if project_root:
        project_root_abs = os.path.abspath(project_root)
    else:
        project_root_abs = os.path.abspath(os.getcwd())

    if package_path:
        package_path_abs = os.path.abspath(package_path)
        return project_root_abs, package_path_abs

    cwd = os.path.abspath(os.getcwd())
    cand1 = os.path.join(cwd, package)
    if os.path.isdir(cand1):
        return cwd, os.path.abspath(cand1)

    parent = os.path.dirname(cwd)
    cand2 = os.path.join(parent, package)
    if os.path.isdir(cand2):
        return parent, os.path.abspath(cand2)

    # Fallback: importa e usa __file__
    try:
        m = importlib.import_module(package)
        pkg_file = getattr(m, "__file__", None)
        if not pkg_file:
            raise RuntimeError(f"Não achei __file__ do pacote {package!r} (package namespace?)")
        package_path_abs = os.path.dirname(os.path.abspath(pkg_file))
        project_root_abs = os.path.dirname(package_path_abs)
        return project_root_abs, package_path_abs
    except Exception as e:
        raise RuntimeError(
            f"Não consegui resolver caminho do pacote {package!r}. "
            f"Passe --package-path explicitamente. Erro: {e}"
        ) from e


def _is_project_file(filename: str, package_path_abs: str) -> bool:
    try:
        abs_path = os.path.abspath(filename)
    except Exception:
        return False
    return abs_path.startswith(package_path_abs)


def _capture_stack(max_frames: int) -> list[traceback.FrameSummary]:
    """
    Remove frames do próprio audit (os últimos 2 frames: _capture_stack + wrapper).
    """
    st = traceback.extract_stack(limit=max_frames + 2)
    if len(st) >= 2:
        st = st[:-2]
    return st


def _format_stack(stack: list[traceback.FrameSummary], package_path_abs: str, include_external: bool) -> list[str]:
    out: list[str] = []
    for fs in stack:
        if include_external or _is_project_file(fs.filename, package_path_abs):
            out.append(f"{fs.filename}:{fs.lineno} in {fs.name} -> {fs.line}")
    return out


def _get_culprit(stack: list[traceback.FrameSummary], package_path_abs: str) -> Optional[str]:
    """
    Pega o frame mais recente dentro do projeto (percorre reversed).
    """
    for fs in reversed(stack):
        if _is_project_file(fs.filename, package_path_abs):
            if os.path.basename(fs.filename) == "import_audit.py":
                continue
            return f"{fs.filename}:{fs.lineno} in {fs.name} -> {fs.line}"
    return None


@contextlib.contextmanager
def _monkey_patch(obj: Any, attr: str, new: Any):
    old = getattr(obj, attr)
    setattr(obj, attr, new)
    try:
        yield old
    finally:
        setattr(obj, attr, old)


class ImportAuditor:
    def __init__(
        self,
        package: str,
        package_path_abs: str,
        *,
        max_frames: int = 40,
        include_external_frames: bool = False,
        allow_threads: bool = False,
        allow_network: bool = False,
        allow_processes: bool = False,
        patch_asyncio_primitives: bool = True,
        patch_event_loop: bool = True,
        patch_process_start: bool = False,
    ):
        self.package = package
        self.package_path_abs = package_path_abs
        self.max_frames = max_frames
        self.include_external_frames = include_external_frames

        # dry-run default: bloqueia
        self.allow_threads = allow_threads
        self.allow_network = allow_network
        self.allow_processes = allow_processes

        self.patch_asyncio_primitives = patch_asyncio_primitives
        self.patch_event_loop = patch_event_loop
        self.patch_process_start = patch_process_start

        self.violations: list[Violation] = []

        # Guarda originais para desfazer
        self._patches: list[tuple[Any, str, Any]] = []

        # Import failures (por módulo)
        self.import_failures: list[ImportFailure] = []

        # Walk_packages failures (pkgutil onerror)
        self.walk_failures: list[ImportFailure] = []

        # patch importlib.import_module pra setar contexto do "módulo sendo importado"
        self._orig_import_module: Optional[Callable[..., Any]] = None

    def _record_violation(self, kind: str, *, detail: dict[str, Any]) -> None:
        st = _capture_stack(self.max_frames)
        culprit = _get_culprit(st, self.package_path_abs)
        module_being_imported = _CURRENT_IMPORT.get()
        formatted_stack = _format_stack(st, self.package_path_abs, include_external=self.include_external_frames)

        self.violations.append(
            Violation(
                kind=kind,
                culprit=culprit,
                module_being_imported=module_being_imported,
                detail=detail,
                stack=formatted_stack,
            )
        )

    def _install_patch(self, obj: Any, attr: str, new: Any) -> None:
        old = getattr(obj, attr)
        self._patches.append((obj, attr, old))
        setattr(obj, attr, new)

    def _uninstall_all(self) -> None:
        for obj, attr, old in reversed(self._patches):
            try:
                setattr(obj, attr, old)
            except Exception:
                # Last resort: continua revertendo os demais
                pass
        self._patches.clear()

        if self._orig_import_module is not None:
            importlib.import_module = self._orig_import_module  # type: ignore[assignment]
            self._orig_import_module = None

    def _install_import_context_patch(self) -> None:
        self._orig_import_module = importlib.import_module

        def patched_import_module(name: str, package: Optional[str] = None) -> Any:
            tok = _CURRENT_IMPORT.set(name)
            try:
                assert self._orig_import_module is not None
                return self._orig_import_module(name, package)
            finally:
                _CURRENT_IMPORT.reset(tok)

        importlib.import_module = patched_import_module  # type: ignore[assignment]

    def _install_thread_patch(self) -> None:
        orig = threading.Thread.start

        def patched_thread_start(thr: threading.Thread, *args: Any, **kwargs: Any) -> Any:
            self._record_violation(
                "thread_start",
                detail={"thread_name": thr.name, "daemon": getattr(thr, "daemon", None)},
            )
            if self.allow_threads:
                return orig(thr, *args, **kwargs)
            raise RuntimeError(f"[import_audit] blocked Thread.start for {thr.name!r}")

        self._install_patch(threading.Thread, "start", patched_thread_start)

    def _install_socket_patch(self) -> None:
        orig = socket.socket.connect

        def patched_socket_connect(sock: socket.socket, address: Any) -> Any:
            self._record_violation("socket_connect", detail={"address": repr(address)})
            if self.allow_network:
                return orig(sock, address)
            raise ConnectionRefusedError(f"[import_audit] blocked socket.connect to {address!r}")

        self._install_patch(socket.socket, "connect", patched_socket_connect)

    def _install_process_patch(self) -> None:
        try:
            import multiprocessing  # local import por ser opcional
        except Exception:
            return

        orig = multiprocessing.Process.start  # type: ignore[attr-defined]

        def patched_process_start(proc: Any, *args: Any, **kwargs: Any) -> Any:
            self._record_violation(
                "process_start",
                detail={"process_name": getattr(proc, "name", None), "pid": getattr(proc, "pid", None)},
            )
            if self.allow_processes:
                return orig(proc, *args, **kwargs)
            raise RuntimeError(f"[import_audit] blocked multiprocessing.Process.start for {getattr(proc, 'name', None)!r}")

        self._install_patch(multiprocessing.Process, "start", patched_process_start)  # type: ignore[arg-type]

    def _install_asyncio_primitives_patch(self) -> None:
        # Patch em __init__ das primitivas relevantes
        candidates: list[tuple[Any, str, str]] = []

        # locks
        candidates.append((asyncio.locks.Lock, "__init__", "Lock"))
        # Alguns Python têm RLock; outros não
        if hasattr(asyncio.locks, "Event"):
            candidates.append((asyncio.locks.Event, "__init__", "Event"))  # type: ignore[attr-defined]
        if hasattr(asyncio.locks, "Condition"):
            candidates.append((asyncio.locks.Condition, "__init__", "Condition"))  # type: ignore[attr-defined]
        if hasattr(asyncio.locks, "Semaphore"):
            candidates.append((asyncio.locks.Semaphore, "__init__", "Semaphore"))  # type: ignore[attr-defined]
        if hasattr(asyncio.locks, "BoundedSemaphore"):
            candidates.append((asyncio.locks.BoundedSemaphore, "__init__", "BoundedSemaphore"))  # type: ignore[attr-defined]

        # queues
        if hasattr(asyncio, "queues"):
            if hasattr(asyncio.queues, "Queue"):
                candidates.append((asyncio.queues.Queue, "__init__", "Queue"))  # type: ignore[attr-defined]
            if hasattr(asyncio.queues, "PriorityQueue"):
                candidates.append((asyncio.queues.PriorityQueue, "__init__", "PriorityQueue"))  # type: ignore[attr-defined]
            if hasattr(asyncio.queues, "LifoQueue"):
                candidates.append((asyncio.queues.LifoQueue, "__init__", "LifoQueue"))  # type: ignore[attr-defined]

        # FIX: capture `self` (the auditor) explicitly to avoid shadowing
        # by the patched method's first positional arg (the primitive instance).
        auditor = self

        for cls, method, label in candidates:
            orig = getattr(cls, method)

            def make_patched(original: Any, primitive_label: str):
                def patched(prim_self: Any, *args: Any, **kwargs: Any) -> Any:
                    auditor._record_violation("asyncio_primitive", detail={"primitive": primitive_label})
                    return original(prim_self, *args, **kwargs)

                return patched

            patched = make_patched(orig, label)
            self._install_patch(cls, method, patched)

    def _install_event_loop_patch(self) -> None:
        # new_event_loop é o mais direto para detectar criação explícita
        orig_new_event_loop = asyncio.new_event_loop

        def patched_new_event_loop(*args: Any, **kwargs: Any) -> Any:
            self._record_violation("event_loop", detail={"call": "asyncio.new_event_loop"})
            return orig_new_event_loop(*args, **kwargs)

        self._install_patch(asyncio, "new_event_loop", patched_new_event_loop)

    def install(self) -> None:
        self._install_import_context_patch()
        self._install_thread_patch()
        self._install_socket_patch()

        if self.patch_process_start:
            self._install_process_patch()

        if self.patch_asyncio_primitives:
            self._install_asyncio_primitives_patch()

        if self.patch_event_loop:
            self._install_event_loop_patch()

    def uninstall(self) -> None:
        self._uninstall_all()

    @contextlib.contextmanager
    def running(self):
        self.install()
        try:
            yield self
        finally:
            self.uninstall()

    def walk_and_import(self) -> None:
        # Garante que conseguimos caminhar
        if not os.path.isdir(self.package_path_abs):
            raise RuntimeError(f"PACKAGE_PATH não existe ou não é diretório: {self.package_path_abs}")

        prefix = f"{self.package}."
        search_path = [self.package_path_abs]

        def on_walk_error(name: str) -> None:
            # pkgutil chama onerror em falhas de walk, não necessariamente de import
            tb = traceback.format_exc()
            self.walk_failures.append(ImportFailure(module=name, error="walk_packages error", traceback=tb))

        for _, module_name, _ in pkgutil.walk_packages(search_path, prefix, onerror=on_walk_error):
            if module_name in sys.modules:
                continue
            try:
                importlib.import_module(module_name)
            except Exception as e:
                tb = traceback.format_exc()
                self.import_failures.append(ImportFailure(module=module_name, error=f"{type(e).__name__}: {e}", traceback=tb))
                # continua para não mascarar outras violações


def _build_report(auditor: ImportAuditor, *, project_root_abs: str, package_path_abs: str, elapsed_s: float) -> dict[str, Any]:
    return {
        "meta": {
            "project_root": project_root_abs,
            "package_path": package_path_abs,
            "package": auditor.package,
            "elapsed_seconds": round(elapsed_s, 4),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "counts": {
            "violations": len(auditor.violations),
            "import_failures": len(auditor.import_failures),
            "walk_failures": len(auditor.walk_failures),
        },
        "violations": [asdict(v) for v in auditor.violations],
        "import_failures": [asdict(f) for f in auditor.import_failures],
        "walk_failures": [asdict(f) for f in auditor.walk_failures],
    }


def _print_human(auditor: ImportAuditor, *, project_root_abs: str, package_path_abs: str, verbose: bool) -> None:
    print("--- Iniciando auditoria de importação ---")
    print(f"Raiz do projeto: {project_root_abs}")
    print(f"Caminho do pacote: {package_path_abs}")
    print("")

    by_kind: dict[str, int] = {}
    for v in auditor.violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1

    print("=== Resultado da Auditoria ===")
    print(f"Total violações: {len(auditor.violations)}")
    for k in sorted(by_kind):
        print(f"  - {k}: {by_kind[k]}")
    print(f"Falhas ao importar módulos: {len(auditor.import_failures)}")
    print(f"Falhas no walk_packages: {len(auditor.walk_failures)}")

    if auditor.import_failures:
        print("\n-- Import failures (NÃO foram engolidas) --")
        for f in auditor.import_failures:
            print(f"  [!] {f.module}: {f.error}")
            if verbose:
                print(f.traceback.rstrip())
                print("")

    if auditor.walk_failures:
        print("\n-- Walk failures (pkgutil.walk_packages) --")
        for f in auditor.walk_failures:
            print(f"  [!] {f.module}: {f.error}")
            if verbose:
                print(f.traceback.rstrip())
                print("")

    if auditor.violations:
        print("\n-- Violações (culpado no projeto quando encontrado) --")
        for v in auditor.violations:
            loc = v.culprit or "Culpado não identificado no projeto (talvez externo)"
            print(f"[{v.kind}] {loc} | importing={v.module_being_imported} | detail={v.detail}")
            if verbose and v.stack:
                print("  stack:")
                for line in v.stack:
                    print(f"    {line}")
                print("")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit side-effects during import/preload.")
    parser.add_argument("--package", default="resync", help="Nome do pacote raiz (default: resync).")
    parser.add_argument("--package-path", default=None, help="Caminho absoluto/relativo para a pasta do pacote.")
    parser.add_argument("--project-root", default=None, help="Raiz do projeto (para sys.path).")
    parser.add_argument("--max-frames", type=int, default=40, help="Max frames na stack capturada (default: 40).")
    parser.add_argument("--include-external-frames", action="store_true", help="Inclui frames fora do projeto na stack.")
    parser.add_argument("--verbose", action="store_true", help="Imprime stack trace das violações e falhas de import.")
    parser.add_argument("--json", default=None, help="Escreve relatório JSON (arquivo) ou '-' para stdout.")
    parser.add_argument("--setenv", action="append", default=[], help="Set default env var KEY=VALUE (repetível).")

    # Dry-run behavior (default: block)
    parser.add_argument("--allow-threads", action="store_true", help="Permite Thread.start (default: bloqueia).")
    parser.add_argument("--allow-network", action="store_true", help="Permite socket.connect (default: bloqueia).")
    parser.add_argument("--allow-processes", action="store_true", help="Permite multiprocessing.Process.start (default: bloqueia).")

    # Patch coverage
    parser.add_argument("--no-asyncio-primitives", action="store_true", help="Desliga patch de primitivas asyncio.")
    parser.add_argument("--no-event-loop", action="store_true", help="Desliga patch de asyncio.new_event_loop.")
    parser.add_argument("--patch-process-start", action="store_true", help="Ativa patch de multiprocessing.Process.start.")

    # Exit codes / CI
    parser.add_argument("--strict", action="store_true", help="Exit 1 se houver violações OU falhas de import/walk.")
    parser.add_argument("--fail-on-violations-only", action="store_true", help="Em --strict, ignora falhas de import/walk.")

    args = parser.parse_args(argv)

    if args.setenv:
        _apply_setenv(args.setenv)

    project_root_abs, package_path_abs = _resolve_paths(args.package, args.package_path, args.project_root)

    # garante sys.path com project root
    if project_root_abs not in sys.path:
        sys.path.insert(0, project_root_abs)

    auditor = ImportAuditor(
        package=args.package,
        package_path_abs=package_path_abs,
        max_frames=args.max_frames,
        include_external_frames=args.include_external_frames,
        allow_threads=args.allow_threads,
        allow_network=args.allow_network,
        allow_processes=args.allow_processes,
        patch_asyncio_primitives=not args.no_asyncio_primitives,
        patch_event_loop=not args.no_event_loop,
        patch_process_start=args.patch_process_start,
    )

    t0 = time.time()
    with auditor.running():
        auditor.walk_and_import()
    elapsed = time.time() - t0

    # Output humano
    _print_human(auditor, project_root_abs=project_root_abs, package_path_abs=package_path_abs, verbose=args.verbose)

    # Output JSON
    if args.json is not None:
        report = _build_report(auditor, project_root_abs=project_root_abs, package_path_abs=package_path_abs, elapsed_s=elapsed)
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        if args.json.strip() == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(payload + "\n")

    # Exit code
    if args.strict:
        has_violations = len(auditor.violations) > 0
        has_failures = (len(auditor.import_failures) + len(auditor.walk_failures)) > 0
        if args.fail_on_violations_only:
            return 1 if has_violations else 0
        return 1 if (has_violations or has_failures) else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

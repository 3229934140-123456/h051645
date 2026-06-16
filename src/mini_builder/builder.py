"""构建执行模块。

每条指令如何在前一层基础上叠加文件变更产生新层:
    构建器内部维护一条「层链」: [L0, L1, ..., Ln]
    其中 L0 是 FROM 指令产生的基础层 (可能为空层), L1..Ln 为后续每条指令。

    对第 i 条构建步骤:
        1. 计算缓存 key = sha256(chain_hash | step_sig)
        2. 若缓存命中 → 直接把缓存层加入层链, 更新 chain_hash, 跳过实际构建
        3. 若缓存未命中:
            a. 在前一层的基础上创建新层 Layer(parent = L_{i-1})
            b. 依据指令类型, 在新层中写入文件变更 (或标记 whiteout)
            c. 若本层无文件变更 → 标记 empty=True (仅元数据, 不写入 FS 层)
            d. 把新层加入层链, 更新 chain_hash, 写入缓存记录

这就是 Docker 的「copy-on-write 构建」过程的纯 Python 实现。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cache import LayerCache
from .context import BuildContext, EmptyContext
from .layer import Layer, LayerStore, LayeredFilesystem
from .parser import BuildStep, InstructionType, parse_dockerfile_file, parse_dockerfile


# ---------------------------------------------------------------------------
# 构建状态
# ---------------------------------------------------------------------------


@dataclass
class BuildState:
    """构建过程中可变的「环境」: env vars, workdir, user, cmd, entrypoint 等。

    这些状态随指令变化, 但不直接产生文件系统层 (有时会产生空层以记录 history)。
    """

    env: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    workdir: str = "/"
    user: str = "root"
    cmd: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    exposed_ports: List[str] = field(default_factory=list)
    volumes: List[str] = field(default_factory=list)
    args: Dict[str, str] = field(default_factory=dict)  # --build-args
    base_image: str = ""
    stage_name: Optional[str] = None
    maintainer: str = ""

    def copy(self) -> "BuildState":
        return BuildState(
            env=dict(self.env),
            labels=dict(self.labels),
            workdir=self.workdir,
            user=self.user,
            cmd=list(self.cmd) if self.cmd else None,
            entrypoint=list(self.entrypoint) if self.entrypoint else None,
            exposed_ports=list(self.exposed_ports),
            volumes=list(self.volumes),
            args=dict(self.args),
            base_image=self.base_image,
            stage_name=self.stage_name,
            maintainer=self.maintainer,
        )

    def apply_substitutions(self, s: str) -> str:
        """在字符串中替换 $VAR / ${VAR} 形式的变量。

        查找顺序: ARG → ENV
        """
        def repl(m: re.Match) -> str:
            name = m.group(2) or m.group(1)
            default = m.group(3)
            # ARG 优先? Docker 中 ENV 会覆盖 ARG; 这里 ENV 优先
            if name in self.env:
                return self.env[name]
            if name in self.args:
                return self.args[name]
            if default is not None:
                return default
            return ""

        # 匹配 ${VAR:-default} 或 $VAR
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
        prev = None
        cur = s
        # 迭代直到稳定, 防止嵌套替换导致遗漏
        for _ in range(10):
            prev = cur
            cur = pattern.sub(repl, cur)
            if cur == prev:
                break
        return cur


# ---------------------------------------------------------------------------
# 构建事件 (用于日志/进度)
# ---------------------------------------------------------------------------


@dataclass
class BuildEvent:
    kind: str
    message: str
    step_index: Optional[int] = None
    step: Optional[BuildStep] = None
    extra: dict = field(default_factory=dict)


class BuildEventSink:
    """事件回调接口, 子类可重写 on_event 来输出日志。"""

    def on_event(self, event: BuildEvent) -> None:
        pass


class PrintEventSink(BuildEventSink):
    def on_event(self, event: BuildEvent) -> None:
        prefix = f"[Step {event.step_index}] " if event.step_index is not None else "[*] "
        print(f"{prefix}{event.kind}: {event.message}")


# ---------------------------------------------------------------------------
# RUN 指令的迷你 shell 模拟器
# ---------------------------------------------------------------------------


class MiniShell:
    """为 RUN 指令提供一个简化的文件系统变更模拟器。

    我们不真正 fork/exec 子进程 (那需要容器运行时), 而是在一个
    临时目录中执行常见命令, 记录文件变更, 再把变更同步到新层。

    支持的命令 (故意做得很有限, 体现原理即可):
        - mkdir [-p] <dir>...
        - touch <file>...
        - echo <text> [> file | >> file]
        - cp <src>... <dest>
        - rm [-rf] <path>...
        - ls [path]
        - cat <file>
        - pwd
        - cd <dir>
        - export K=V
        - sh -c "..." (递归)
    """

    def __init__(self, root_fs_dir: str, env: Dict[str, str], workdir: str = "/") -> None:
        self.root = Path(root_fs_dir)
        self.env: Dict[str, str] = dict(env)
        self.cwd = "/" if not workdir else workdir
        self.stdout_lines: List[str] = []
        self._ensure_dir(self.root)

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.cwd) / p
        # 规范化
        resolved = os.path.normpath(str(p)).replace("\\", "/")
        return self.root / resolved.lstrip("/")

    def run(self, command: str) -> int:
        """执行一条或多条 (以 ; 或 && 分隔) shell 命令。"""
        # 简单拆分: 按 && / || / ; 粗粒度
        statements = self._split_statements(command)
        last_rc = 0
        mode = "then"  # then / then_if_failed
        for stmt in statements:
            if stmt in ("&&",):
                mode = "then_if_ok"
                continue
            if stmt in ("||",):
                mode = "then_if_failed"
                continue
            if stmt in (";",):
                mode = "then"
                continue
            if mode == "then_if_ok" and last_rc != 0:
                continue
            if mode == "then_if_failed" and last_rc == 0:
                continue
            last_rc = self._run_single(stmt)
            mode = "then"
        return last_rc

    def _split_statements(self, command: str) -> List[str]:
        out: List[str] = []
        buf: List[str] = []
        in_sq = False
        in_dq = False
        i = 0
        while i < len(command):
            c = command[i]
            if c == "'" and not in_dq:
                in_sq = not in_sq
                buf.append(c)
            elif c == '"' and not in_sq:
                if buf and buf[-1] == "\\":
                    buf[-1] = c
                else:
                    in_dq = not in_dq
                    buf.append(c)
            elif c in ("&", "|", ";") and not in_sq and not in_dq:
                # 检查是否 && 或 ||
                if i + 1 < len(command) and command[i + 1] == c and c in ("&", "|"):
                    if buf:
                        out.append("".join(buf).strip())
                        buf = []
                    out.append(c + c)
                    i += 1
                else:
                    if buf:
                        out.append("".join(buf).strip())
                        buf = []
                    out.append(c)
            else:
                buf.append(c)
            i += 1
        if buf:
            out.append("".join(buf).strip())
        return [s for s in out if s != ""]

    def _run_single(self, stmt: str) -> int:
        stmt = stmt.strip()
        if not stmt:
            return 0
        # 展开变量
        stmt_expanded = self._expand(stmt)
        # 用 shlex 拆词
        try:
            tokens = shlex.split(stmt_expanded, posix=True)
        except ValueError:
            tokens = stmt_expanded.split()
        if not tokens:
            return 0
        cmd = tokens[0]
        args = tokens[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            self.stdout_lines.append(f"sh: {cmd}: command not found (simulated)")
            return 127
        try:
            return handler(args)
        except Exception as e:
            self.stdout_lines.append(f"error: {e}")
            return 1

    # ---- 命令实现 ----------------------------------------------------

    def _cmd_echo(self, args: List[str]) -> int:
        # 简单处理 > / >>
        out_list: List[str] = []
        redirect_to = None
        append = False
        i = 0
        while i < len(args):
            a = args[i]
            if a in (">", ">>") and i + 1 < len(args):
                append = a == ">>"
                redirect_to = args[i + 1]
                i += 2
                continue
            out_list.append(a)
            i += 1
        line = " ".join(out_list) + "\n"
        if redirect_to is not None:
            target = self._abs(redirect_to)
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with target.open(mode, encoding="utf-8") as f:
                f.write(line)
        else:
            self.stdout_lines.append(line.rstrip("\n"))
        return 0

    def _cmd_mkdir(self, args: List[str]) -> int:
        parents = False
        for a in args:
            if a == "-p":
                parents = True
                continue
            p = self._abs(a)
            if parents:
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir()
        return 0

    def _cmd_touch(self, args: List[str]) -> int:
        for a in args:
            p = self._abs(a)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                # update mtime
                p.touch()
            else:
                p.open("w").close()
        return 0

    def _cmd_rm(self, args: List[str]) -> int:
        recursive = False
        force = False
        for a in args:
            if a.startswith("-") and all(c in "rf" for c in a.lstrip("-")):
                recursive = "r" in a or "R" in a
                force = "f" in a
                continue
            p = self._abs(a)
            try:
                if p.is_dir() and not p.is_symlink():
                    if recursive:
                        shutil.rmtree(p)
                    else:
                        if force:
                            return 0
                        raise IsADirectoryError(str(a))
                else:
                    if p.exists():
                        p.unlink()
                    elif not force:
                        raise FileNotFoundError(str(a))
            except Exception as e:
                if force:
                    continue
                self.stdout_lines.append(f"rm: {e}")
                return 1
        return 0

    def _cmd_cp(self, args: List[str]) -> int:
        if len(args) < 2:
            return 1
        sources = args[:-1]
        dest = args[-1]
        dest_abs = self._abs(dest)
        for s in sources:
            s_abs = self._abs(s)
            if dest_abs.is_dir() or (len(sources) > 1 and not dest_abs.exists()):
                dest_abs.mkdir(parents=True, exist_ok=True)
                target = dest_abs / s_abs.name
            else:
                dest_abs.parent.mkdir(parents=True, exist_ok=True)
                target = dest_abs
            if s_abs.is_dir():
                if target.exists():
                    target = target / s_abs.name
                shutil.copytree(s_abs, target, dirs_exist_ok=True)
            else:
                shutil.copy2(s_abs, target)
        return 0

    def _cmd_mv(self, args: List[str]) -> int:
        if len(args) != 2:
            return 1
        s_abs = self._abs(args[0])
        d_abs = self._abs(args[1])
        if d_abs.is_dir():
            d_abs = d_abs / s_abs.name
        d_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s_abs), str(d_abs))
        return 0

    def _cmd_ls(self, args: List[str]) -> int:
        target = args[0] if args else self.cwd
        p = self._abs(target)
        if not p.exists():
            self.stdout_lines.append(f"ls: {target}: No such file or directory")
            return 1
        if p.is_dir():
            for c in sorted(p.iterdir()):
                self.stdout_lines.append(c.name + ("/" if c.is_dir() else ""))
        else:
            self.stdout_lines.append(p.name)
        return 0

    def _cmd_cat(self, args: List[str]) -> int:
        for a in args:
            p = self._abs(a)
            if not p.is_file():
                continue
            try:
                self.stdout_lines.append(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
        return 0

    def _cmd_pwd(self, args: List[str]) -> int:
        self.stdout_lines.append(self.cwd)
        return 0

    def _cmd_cd(self, args: List[str]) -> int:
        if not args:
            self.cwd = "/"
            return 0
        target = args[0]
        new = os.path.normpath(os.path.join(self.cwd, target)).replace("\\", "/")
        if not new.startswith("/"):
            new = "/" + new
        self.cwd = new
        return 0

    def _cmd_export(self, args: List[str]) -> int:
        for a in args:
            if "=" in a:
                k, _, v = a.partition("=")
                self.env[k] = v
        return 0

    def _cmd_env(self, args: List[str]) -> int:
        for k in sorted(self.env):
            self.stdout_lines.append(f"{k}={self.env[k]}")
        return 0

    def _cmd_sh(self, args: List[str]) -> int:
        if len(args) >= 2 and args[0] == "-c":
            return self.run(args[1])
        return 0

    def _cmd_chmod(self, args: List[str]) -> int:
        # 简化: 忽略 mode, 只确保路径存在
        if len(args) < 2:
            return 1
        for a in args[1:]:
            p = self._abs(a)
            if not p.exists():
                self.stdout_lines.append(f"chmod: {a}: No such file or directory")
                return 1
        return 0

    def _cmd_chown(self, args: List[str]) -> int:
        return self._cmd_chmod(args)

    # ---- 变量展开 ------------------------------------------------------

    def _expand(self, s: str) -> str:
        def repl(m: re.Match) -> str:
            name = m.group(1) or m.group(3)
            default = m.group(2)
            if name in self.env:
                return self.env[name]
            return default if default is not None else ""
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")
        prev = None
        cur = s
        for _ in range(10):
            prev = cur
            cur = pattern.sub(repl, cur)
            if cur == prev:
                break
        return cur


# ---------------------------------------------------------------------------
# 主构建器
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    success: bool
    layers: List[Layer]
    final_state: BuildState
    chain_hash: str
    events: List[BuildEvent]
    error_message: Optional[str] = None


class Builder:
    """逐指令构建镜像的核心构建器。"""

    def __init__(
        self,
        *,
        layer_store: LayerStore,
        cache: LayerCache,
        context: Optional[BuildContext] = None,
        event_sink: Optional[BuildEventSink] = None,
        use_cache: bool = True,
        build_args: Optional[Dict[str, str]] = None,
    ) -> None:
        self.layer_store = layer_store
        self.cache = cache
        self.context = context or EmptyContext()
        self.event_sink = event_sink or BuildEventSink()
        self.use_cache = use_cache
        self.build_args: Dict[str, str] = dict(build_args or {})
        # 运行时状态
        self.layers: List[Layer] = []
        self.state = BuildState(args=dict(self.build_args))
        self.chain_hash: str = ""
        self.events: List[BuildEvent] = []

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def build_from_dockerfile(self, dockerfile_text: str) -> BuildResult:
        from .parser import parse_dockerfile
        try:
            steps = parse_dockerfile(dockerfile_text)
        except Exception as e:
            err = f"Dockerfile 解析失败: {e}"
            return BuildResult(False, [], BuildState(), "", [], err)
        return self.build(steps)

    def build_from_file(self, dockerfile_path: str) -> BuildResult:
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.build_from_dockerfile(text)

    def build(self, steps: List[BuildStep]) -> BuildResult:
        if not steps:
            return BuildResult(False, [], self.state.copy(), self.chain_hash, self.events, "没有任何构建步骤")
        # 第一条必须是 FROM
        if steps[0].type != InstructionType.FROM:
            return BuildResult(False, [], self.state.copy(), self.chain_hash, self.events, "第一条指令必须是 FROM")

        try:
            for idx, step in enumerate(steps):
                self._emit("STEP", f"执行 {step.type.value}: {step.raw[:80]}", step_index=idx + 1, step=step)
                self._execute_step(idx, step)
            return BuildResult(
                True,
                list(self.layers),
                self.state.copy(),
                self.chain_hash,
                list(self.events),
            )
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            self._emit("ERROR", err)
            return BuildResult(False, list(self.layers), self.state.copy(), self.chain_hash, list(self.events), err)

    # ------------------------------------------------------------------
    # 单步骤执行
    # ------------------------------------------------------------------

    def _execute_step(self, idx: int, step: BuildStep) -> None:
        # 缓存查找
        cached = None
        if self.use_cache and step.type not in (InstructionType.FROM, InstructionType.ARG):
            cached = self.cache.get_layer_if_cached(self.chain_hash, step, self.context)

        if cached is not None:
            layer, next_chain = cached
            self.layers.append(layer)
            self.chain_hash = next_chain
            # 元数据指令: 需要推进构建状态
            self._apply_state_effects(step, layer_empty_hint=layer.empty)
            self._emit("CACHE", f"命中缓存, 复用层 {layer.layer_id[:12]}", step_index=idx + 1)
            return

        # 缓存未命中: 创建新层并执行
        parent = self.layers[-1] if self.layers else None
        new_layer = self.layer_store.create_layer(parent=parent)
        new_layer.created_by = step.raw

        # 执行具体指令
        layer_changed = self._dispatch_execute(idx, step, new_layer)

        # 没有任何文件变更 → 空层 (优化: 相同空层也能缓存)
        if not layer_changed:
            new_layer.empty = True

        # 存储元数据
        self.layer_store.store_layer(new_layer)

        # 更新 chain_hash 与缓存
        self.chain_hash = self.cache.record(self.chain_hash, step, self.context, new_layer)
        self.layers.append(new_layer)
        self._emit("BUILD", f"创建层 {new_layer.layer_id[:12]}{' (空层)' if new_layer.empty else ''}", step_index=idx + 1)

    def _dispatch_execute(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        """执行单条指令并返回「本层是否有文件变更」。"""
        handler = getattr(self, f"_exec_{step.type.value.lower()}", None)
        if handler is None:
            # 未知指令 → 作为空层 (仅元数据)
            self._apply_state_effects(step)
            return False
        return handler(idx, step, layer)

    # ------------------------------------------------------------------
    # 各类指令的具体执行
    # ------------------------------------------------------------------

    def _exec_from(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        image_ref = step.args[0]
        self.state.base_image = image_ref
        if "as" in step.kwargs:
            self.state.stage_name = step.kwargs["as"]
        self._emit("FROM", f"基础镜像: {image_ref}", step_index=idx + 1)
        # FROM 层: 我们无法下载真实镜像, 所以创建一个「模拟的基础层」
        # 在基础层里放入几个常见目录, 表示 rootfs 骨架
        for d in ("bin", "etc", "usr", "tmp", "var", "home", "root", "opt"):
            layer.add_string_content(f"{d}/.keep", "")
        # 默认 /etc/hostname, /etc/resolv.conf
        layer.add_string_content("etc/hostname", "builder-container\n")
        layer.add_string_content("etc/resolv.conf", "nameserver 8.8.8.8\n")
        layer.add_string_content("etc/passwd", "root:x:0:0:root:/root:/bin/sh\n")
        return True

    def _exec_run(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        # 准备一个临时目录作为 MiniShell 的 rootfs, 先把当前层链 materialize 进去
        fs_dir = tempfile.mkdtemp(prefix="mib-run-root-")
        try:
            current_layers = list(self.layers) + [layer]  # 其实不应该加, 但先放着
            lfs = LayeredFilesystem(self.layers)
            lfs.materialize(fs_dir)

            # 合并 env: state.env + 已有的 shell env 里的 PATH 等
            run_env = dict(self.state.env)
            run_env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")

            shell = MiniShell(fs_dir, env=run_env, workdir=self.state.workdir)
            command_text = step.args[0] if step.args else ""

            if "exec" in step.flags:
                # JSON exec 形式: ["echo", "hi"]
                try:
                    arr = json.loads(command_text)
                    # 转成 shell 形式
                    cmd_line = " ".join(shlex.quote(a) for a in arr)
                except Exception:
                    cmd_line = command_text
            else:
                # RUN 默认用 /bin/sh -c <cmd>
                cmd_line = command_text

            self._emit("RUN-EXEC", f"执行: {cmd_line[:120]}", step_index=idx + 1)
            rc = shell.run(cmd_line)
            if shell.stdout_lines:
                for line in shell.stdout_lines[-30:]:  # 只显示最后 30 行
                    self._emit("RUN-OUT", line, step_index=idx + 1)
            if rc != 0:
                raise RuntimeError(f"RUN 指令返回非零退出码: {rc}")

            # 对比 fs_dir 和原来的层链, 找出变更, 写入 layer
            # 简单做法: 把整个 fs_dir 复制进 layer (会包含基础层的所有文件),
            # 更优做法: 计算 diff。这里用更精确的 diff 方法。
            return self._diff_into_layer(fs_dir, layer)
        finally:
            shutil.rmtree(fs_dir, ignore_errors=True)

    def _diff_into_layer(self, fs_dir: str, layer: Layer) -> bool:
        """把 (fs_dir 相对当前层链的差异) 写入 layer。"""
        lfs = LayeredFilesystem(self.layers)
        changed = False
        for dirpath, dirnames, filenames in os.walk(fs_dir):
            rel_dir = os.path.relpath(dirpath, fs_dir)
            if rel_dir == ".":
                rel_dir = ""
            for dn in dirnames:
                rel = os.path.join(rel_dir, dn).replace("\\", "/") if rel_dir else dn
                existing, _ = lfs.resolve_path(rel)
                if existing is None:
                    # 新增目录: 放一个 .keep
                    layer.add_string_content(f"{rel}/.keep", "")
                    changed = True
            for fn in filenames:
                rel = os.path.join(rel_dir, fn).replace("\\", "/") if rel_dir else fn
                abs_new = os.path.join(dirpath, fn)
                existing, _ = lfs.resolve_path(rel)
                if existing is None:
                    # 新增
                    layer.add_file(rel, abs_new)
                    changed = True
                else:
                    # 比较内容
                    with open(existing, "rb") as f1, open(abs_new, "rb") as f2:
                        same = True
                        while True:
                            b1 = f1.read(1 << 16)
                            b2 = f2.read(1 << 16)
                            if b1 != b2:
                                same = False
                                break
                            if not b1:
                                break
                    if not same:
                        layer.add_file(rel, abs_new)
                        changed = True
        # 检测删除: 遍历层链中存在的文件, 是否在 fs_dir 中消失
        all_existing_rel: List[str] = []
        for old_layer in self.layers:
            for rel, is_dir, is_special in old_layer.walk_changes():
                if is_special or is_dir:
                    continue
                all_existing_rel.append(rel)
        # 去重
        for rel in sorted(set(all_existing_rel)):
            abs_expected = os.path.join(fs_dir, rel.lstrip("/"))
            resolved, _ = lfs.resolve_path(rel)
            if resolved is None:
                continue
            if not os.path.exists(abs_expected):
                # 被 RUN 删除了
                layer.remove_path(rel)
                changed = True
        return changed

    def _exec_copy(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        return self._do_copy_or_add(idx, step, layer, is_add=False)

    def _exec_add(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        return self._do_copy_or_add(idx, step, layer, is_add=True)

    def _do_copy_or_add(self, idx: int, step: BuildStep, layer: Layer, *, is_add: bool) -> bool:
        if len(step.args) < 2:
            raise RuntimeError("COPY/ADD 缺少参数")
        srcs = step.args[:-1]
        dest = step.args[-1]
        # 处理相对 workdir
        if not dest.startswith("/"):
            wd = self.state.workdir.rstrip("/") or ""
            dest = f"{wd}/{dest}" if wd else "/" + dest
        dest = dest.replace("\\", "/")

        # 从上下文中收集文件
        matched = self.context.collect_paths(srcs)
        if not matched and not is_add:
            raise FileNotFoundError(f"COPY 没有匹配任何文件: {srcs}")

        self._emit(
            "COPY",
            f"复制 {len(matched)} 个条目 → {dest}",
            step_index=idx + 1,
            extra={"matched": matched},
        )
        changed = False
        for rel in matched:
            ctx_entry = self.context.manifest.get(rel)
            if ctx_entry is None:
                continue
            src_abs = ctx_entry.abs_path
            # 目标相对 dest 计算: 如果 dest 以 / 结尾或是目录, 则文件进入 dest/<basename>
            dest_is_dir = dest.endswith("/") or (len(matched) > 1)
            if ctx_entry.is_dir:
                # 目录: 把目录树拷进 dest/<name>/
                target_dir_rel = dest.rstrip("/") + "/" + os.path.basename(rel) if dest_is_dir else dest
                layer.add_directory_tree(target_dir_rel, src_abs)
                changed = True
            else:
                if dest_is_dir:
                    target_rel = dest.rstrip("/") + "/" + os.path.basename(rel)
                else:
                    target_rel = dest
                layer.add_file(target_rel, src_abs)
                changed = True
        # ADD 额外功能: 自动解压 tar / 下载 URL (这里只实现 tar 解压)
        if is_add and changed:
            self._auto_extract_local_tars(layer, matched)
        return changed

    def _auto_extract_local_tars(self, layer: Layer, matched: List[str]) -> None:
        """识别并解压刚拷入 layer 的 .tar / .tar.gz 文件。"""
        import tarfile
        import gzip

        for rel in list(matched):
            basename = os.path.basename(rel).lower()
            if not (basename.endswith(".tar") or basename.endswith(".tar.gz") or basename.endswith(".tgz")):
                continue
            # 在 layer 的 root 中找到它
            # 简化: 直接尝试用 context 里的 src_abs 解压
            ctx_entry = self.context.manifest.get(rel)
            if ctx_entry is None or ctx_entry.is_dir:
                continue
            try:
                extract_dir = tempfile.mkdtemp(prefix="mib-add-extract-")
                mode = "r:gz" if basename.endswith((".tar.gz", ".tgz")) else "r:"
                with tarfile.open(ctx_entry.abs_path, mode) as tf:
                    tf.extractall(extract_dir)
                dest_dir = "/".join(["/"] + os.path.dirname(rel).split("/")[-1:]) if "/" in rel else "/"
                layer.add_directory_tree(dest_dir if dest_dir != "/" else "", extract_dir)
                shutil.rmtree(extract_dir, ignore_errors=True)
                # 删除原来的 tar 文件
                layer.remove_path(rel)
            except Exception:
                pass

    def _exec_env(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        for k, v in step.kwargs.items():
            self.state.env[k] = self.state.apply_substitutions(v)
        self._emit("ENV", f"设置 {len(step.kwargs)} 个环境变量", step_index=idx + 1)
        return False

    def _exec_label(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        for k, v in step.kwargs.items():
            self.state.labels[k] = v
        self._emit("LABEL", f"设置 {len(step.kwargs)} 个标签", step_index=idx + 1)
        return False

    def _exec_workdir(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        target = step.args[0] if step.args else "/"
        target = self.state.apply_substitutions(target)
        if not target.startswith("/"):
            wd = self.state.workdir.rstrip("/") or ""
            target = f"{wd}/{target}" if wd else "/" + target
        target = os.path.normpath(target).replace("\\", "/")
        self.state.workdir = target
        # WORKDIR 还会创建目录
        layer.add_string_content(target.lstrip("/") + "/.keep", "")
        self._emit("WORKDIR", f"工作目录: {target}", step_index=idx + 1)
        return True

    def _exec_user(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.user = step.args[0] if step.args else "root"
        self._emit("USER", f"用户: {self.state.user}", step_index=idx + 1)
        return False

    def _exec_expose(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.exposed_ports.extend(step.args)
        self._emit("EXPOSE", f"端口: {', '.join(step.args)}", step_index=idx + 1)
        return False

    def _exec_volume(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.volumes.extend(step.args)
        self._emit("VOLUME", f"挂载点: {', '.join(step.args)}", step_index=idx + 1)
        return False

    def _exec_cmd(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.cmd = self._parse_exec_or_shell_arg(step)
        self._emit("CMD", f"{self.state.cmd}", step_index=idx + 1)
        return False

    def _exec_entrypoint(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.entrypoint = self._parse_exec_or_shell_arg(step)
        self._emit("ENTRYPOINT", f"{self.state.entrypoint}", step_index=idx + 1)
        return False

    def _exec_arg(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        name = step.kwargs.get("name", "")
        default = step.kwargs.get("default")
        if name and name not in self.state.args:
            if default is not None:
                self.state.args[name] = self.state.apply_substitutions(default)
        self._emit("ARG", f"{name}={self.state.args.get(name, '<未设置>')}", step_index=idx + 1)
        return False

    def _exec_maintainer(self, idx: int, step: BuildStep, layer: Layer) -> bool:
        self.state.maintainer = step.args[0] if step.args else ""
        return False

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _parse_exec_or_shell_arg(self, step: BuildStep) -> Optional[List[str]]:
        raw = step.args[0] if step.args else ""
        if "exec" in step.flags and raw.startswith("["):
            try:
                return list(json.loads(raw))
            except Exception:
                pass
        return ["/bin/sh", "-c", raw]

    def _apply_state_effects(self, step: BuildStep, layer_empty_hint: Optional[bool] = None) -> None:
        """缓存命中时, 仍然要推进 state, 但不产生文件层。"""
        dummy = Layer(layer_id="dummy", parent_id=None, root="")
        # 用已有的 exec 方法 (它们不会 touch 文件系统, 因为 root 是空字符串)
        # 但要避免 WORKDIR 等操作访问 root。简单做法: 只处理非文件型指令。
        state_only = {
            InstructionType.ENV, InstructionType.LABEL, InstructionType.USER,
            InstructionType.EXPOSE, InstructionType.VOLUME, InstructionType.CMD,
            InstructionType.ENTRYPOINT, InstructionType.ARG, InstructionType.MAINTAINER,
            InstructionType.WORKDIR,
        }
        if step.type in state_only:
            handler = getattr(self, f"_exec_{step.type.value.lower()}", None)
            if handler is not None:
                try:
                    # WORKDIR 需要特殊处理, 不真的写文件
                    if step.type == InstructionType.WORKDIR:
                        target = step.args[0] if step.args else "/"
                        target = self.state.apply_substitutions(target)
                        if not target.startswith("/"):
                            wd = self.state.workdir.rstrip("/") or ""
                            target = f"{wd}/{target}" if wd else "/" + target
                        self.state.workdir = os.path.normpath(target).replace("\\", "/")
                    else:
                        handler(-1, step, dummy)
                except Exception:
                    pass

    def _emit(self, kind: str, message: str, step_index: Optional[int] = None, step: Optional[BuildStep] = None, extra: Optional[dict] = None) -> None:
        ev = BuildEvent(kind=kind, message=message, step_index=step_index, step=step, extra=extra or {})
        self.events.append(ev)
        self.event_sink.on_event(ev)

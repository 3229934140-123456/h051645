"""Dockerfile 解析模块。

工作流程:
    Lexer  ──→ Token list  ──→ Parser  ──→ BuildStep list
    (字符流)     (词法单元)      (语法规则)     (有序构建步骤)

构建指令如何解析成有序步骤:
    1. Lexer 按行扫描, 处理反斜杠续行, 识别注释与空行
    2. Parser 按指令关键字 (FROM / RUN / COPY 等) 分派到各 parse_* 方法
    3. 每个 parse_* 方法将指令参数解析为强类型 BuildStep 对象
    4. 按原文顺序返回 BuildStep 列表, 保证后续构建器可顺序执行
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class InstructionType(str, Enum):
    """所有支持的 Dockerfile 指令类型。"""

    FROM = "FROM"
    RUN = "RUN"
    COPY = "COPY"
    ADD = "ADD"
    ENV = "ENV"
    WORKDIR = "WORKDIR"
    USER = "USER"
    EXPOSE = "EXPOSE"
    CMD = "CMD"
    ENTRYPOINT = "ENTRYPOINT"
    LABEL = "LABEL"
    ARG = "ARG"
    VOLUME = "VOLUME"
    MAINTAINER = "MAINTAINER"


@dataclass
class BuildStep:
    """解析后的单条构建步骤, 对应 Dockerfile 中一条指令。

    Attributes:
        type:       指令类型枚举
        raw:        原始文本 (不含注释与续行), 用于缓存哈希
        line_no:    原始行号 (第一条续行所在行), 用于错误报告
        args:       位置参数 (例如 RUN 的 shell 命令字符串)
        kwargs:     命名参数 (例如 COPY 的 --from=..., --chown=...)
        flags:      布尔标志
    """

    type: InstructionType
    raw: str
    line_no: int
    args: List[str] = field(default_factory=list)
    kwargs: Dict[str, str] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def cache_key(self) -> str:
        """生成用于缓存的 key 基础部分。

        不同指令会混入额外信息 (如文件内容哈希), 但指令本身的
        规范化文本 + 类型 是最小的指纹。
        """
        norm = " ".join(self.raw.split())
        return f"{self.type.value}|{norm}"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


@dataclass
class Token:
    kind: str
    value: str
    line_no: int


class Lexer:
    """简单的基于行的 Lexer。

    Dockerfile 语法核心:
      - # 开头为注释 (仅当行首为 # 时)
      - 行尾 \ 表示续行, 续行后行首空白会被折叠为一个空格
      - 指令名不区分大小写, 但输出时统一大写
      - 空行或纯空白行被忽略
    """

    _COMMENT_RE = re.compile(r"^\s*#")
    _CONTINUATION_RE = re.compile(r"\\\s*$")
    _INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z]+)\b(.*)$", re.DOTALL)

    def __init__(self, source: str) -> None:
        self.source = source
        self._lines: List[str] = source.splitlines()

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        i = 0
        total = len(self._lines)
        while i < total:
            raw_line = self._lines[i]
            start_line = i + 1

            if self._COMMENT_RE.match(raw_line) or raw_line.strip() == "":
                i += 1
                continue

            # 处理续行: 累积所有以 \ 结尾的行
            buf = raw_line
            while self._CONTINUATION_RE.search(buf):
                # 去掉末尾的反斜杠
                buf = self._CONTINUATION_RE.sub("", buf).rstrip()
                i += 1
                if i >= total:
                    break
                next_raw = self._lines[i]
                # 跳过续行中的注释行 (Docker 行为)
                if self._COMMENT_RE.match(next_raw):
                    continue
                buf = buf + " " + next_raw.lstrip()

            m = self._INSTRUCTION_RE.match(buf)
            if not m:
                raise ParseError(f"第 {start_line} 行: 无法解析指令: {buf!r}")
            instr_name = m.group(1).upper()
            rest = m.group(2).strip()
            tokens.append(Token("INSTRUCTION", instr_name, start_line))
            tokens.append(Token("ARGUMENTS", rest, start_line))
            i += 1
        return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """解析异常。"""


class Parser:
    """将 Token 列表解析为 BuildStep 有序列表。"""

    _VALID_INSTRUCTIONS = {t.value for t in InstructionType}

    def __init__(self, tokens: List[Token], *, dockerfile_text: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self._raw_text = dockerfile_text

    def parse(self) -> List[BuildStep]:
        steps: List[BuildStep] = []
        while self.pos < len(self.tokens):
            name_tok = self._expect("INSTRUCTION")
            arg_tok = self._expect("ARGUMENTS")
            instr_name = name_tok.value
            if instr_name not in self._VALID_INSTRUCTIONS:
                raise ParseError(f"第 {name_tok.line_no} 行: 未知指令 {instr_name}")
            itype = InstructionType(instr_name)
            raw_line = f"{instr_name} {arg_tok.value}".rstrip()
            step = self._dispatch(itype, arg_tok.value, raw_line, arg_tok.line_no)
            steps.append(step)
        return steps

    def _expect(self, kind: str) -> Token:
        if self.pos >= len(self.tokens):
            raise ParseError(f"期望 token {kind}, 但已到达文件末尾")
        tok = self.tokens[self.pos]
        if tok.kind != kind:
            raise ParseError(f"第 {tok.line_no} 行: 期望 {kind}, 实际 {tok.kind}")
        self.pos += 1
        return tok

    # ------------------------------------------------------------------
    # 指令分派
    # ------------------------------------------------------------------

    def _dispatch(
        self, itype: InstructionType, arguments: str, raw: str, line_no: int
    ) -> BuildStep:
        method = getattr(self, f"_parse_{itype.value.lower()}", self._parse_generic)
        args, kwargs, flags = method(arguments)
        return BuildStep(
            type=itype,
            raw=raw,
            line_no=line_no,
            args=args,
            kwargs=kwargs,
            flags=flags,
        )

    # ---- 各类指令的专用解析 -------------------------------------------

    def _parse_from(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # FROM [--platform=<platform>] <image> [AS <name>]
        args: List[str] = []
        kwargs: Dict[str, str] = {}
        parts = self._split_flags(arguments)
        iter_parts = iter(parts)
        for p in iter_parts:
            if p.startswith("--"):
                k, _, v = p[2:].partition("=")
                kwargs[k] = v if _ else ""
            elif p.upper() == "AS":
                try:
                    kwargs["as"] = next(iter_parts)
                except StopIteration:
                    raise ParseError("FROM ... AS 后缺少阶段名")
            else:
                args.append(p)
        if not args:
            raise ParseError("FROM 指令缺少镜像参数")
        return args, kwargs, []

    def _parse_run(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # RUN 有两种: shell 形式 (整串字符串) 和 exec 形式 (JSON 数组)
        # 这里统一把原始内容放进 args[0], 构建器再区分处理
        stripped = arguments.strip()
        if stripped.startswith("["):
            args = [stripped]
            flags = ["exec"]
        else:
            args = [stripped]
            flags = ["shell"]
        return args, {}, flags

    def _parse_copy(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        return self._parse_src_dest_with_flags(arguments)

    def _parse_add(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        return self._parse_src_dest_with_flags(arguments)

    def _parse_src_dest_with_flags(
        self, arguments: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        # COPY [--from=...] [--chown=...] <src>... <dest>
        kwargs: Dict[str, str] = {}
        positional: List[str] = []
        parts = self._split_flags(arguments)
        for p in parts:
            if p.startswith("--"):
                k, _, v = p[2:].partition("=")
                kwargs[k] = v if _ else ""
            else:
                positional.append(p)
        if len(positional) < 2:
            raise ParseError("COPY/ADD 至少需要 <src> 和 <dest> 两个参数")
        return positional, kwargs, []

    def _parse_env(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # ENV 有两种形式:
        #   ENV KEY=VALUE KEY2=VALUE2  (推荐)
        #   ENV KEY VALUE              (遗留, 整行作为值)
        kwargs: Dict[str, str] = {}
        stripped = arguments.strip()
        if "=" in stripped.split(maxsplit=1)[0] if stripped else False:
            # 形式 1: 多个 K=V
            tokens = shlex_split_preserve(stripped)
            for tok in tokens:
                if "=" not in tok:
                    raise ParseError(f"ENV 无法解析 {tok!r}, 期望 KEY=VALUE")
                k, _, v = tok.partition("=")
                kwargs[k] = v
        else:
            # 形式 2: 遗留
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                kwargs[parts[0]] = parts[1]
            elif len(parts) == 1:
                kwargs[parts[0]] = ""
        return [], kwargs, []

    def _parse_label(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # LABEL 与 ENV 形式 1 相同: LABEL k=v k2=v2
        kwargs: Dict[str, str] = {}
        tokens = shlex_split_preserve(arguments)
        for tok in tokens:
            if "=" not in tok:
                raise ParseError(f"LABEL 无法解析 {tok!r}, 期望 KEY=VALUE")
            k, _, v = tok.partition("=")
            kwargs[k] = v
        return [], kwargs, []

    def _parse_arg(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # ARG <name>[=<default>]
        parts = arguments.strip().split("=", 1)
        name = parts[0].strip()
        default = parts[1] if len(parts) == 2 else None
        kwargs: Dict[str, str] = {"name": name}
        if default is not None:
            kwargs["default"] = default
        return [], kwargs, []

    def _parse_workdir(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        return [arguments.strip()], {}, []

    def _parse_user(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        return [arguments.strip()], {}, []

    def _parse_expose(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        ports = arguments.strip().split()
        return ports, {}, []

    def _parse_volume(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        # VOLUME ["/data"] 或 VOLUME /data /data2
        stripped = arguments.strip()
        if stripped.startswith("["):
            import json

            try:
                paths = list(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ParseError(f"VOLUME JSON 解析失败: {e}")
        else:
            paths = stripped.split()
        return paths, {}, []

    def _parse_cmd(self, arguments: str) -> Tuple[List[str], Dict[str, str], List[str]]:
        return self._parse_exec_or_shell(arguments)

    def _parse_entrypoint(
        self, arguments: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        return self._parse_exec_or_shell(arguments)

    def _parse_exec_or_shell(
        self, arguments: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        stripped = arguments.strip()
        if stripped.startswith("["):
            return [stripped], {}, ["exec"]
        return [stripped], {}, ["shell"]

    def _parse_maintainer(
        self, arguments: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        return [arguments.strip()], {}, []

    def _parse_generic(
        self, arguments: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        return arguments.strip().split(), {}, []

    # ---- 工具函数 ----------------------------------------------------

    @staticmethod
    def _split_flags(s: str) -> List[str]:
        """按空白拆分但保留被引用部分, 使用 shlex。"""
        return shlex_split_preserve(s)


def shlex_split_preserve(s: str) -> List[str]:
    """shlex.split 的包装, 避免 Windows 路径等问题。"""
    try:
        return shlex.split(s, posix=True)
    except ValueError as e:
        raise ParseError(f"参数解析失败: {e}")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def parse_dockerfile(text: str) -> List[BuildStep]:
    """解析 Dockerfile 文本, 返回有序的 BuildStep 列表。

    这是模块最主要的入口函数。
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens, dockerfile_text=text)
    return parser.parse()


def parse_dockerfile_file(path: str) -> List[BuildStep]:
    """从文件路径解析 Dockerfile。"""
    with open(path, "r", encoding="utf-8") as f:
        return parse_dockerfile(f.read())

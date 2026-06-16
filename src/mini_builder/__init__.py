"""Mini Image Builder - 一个轻量级容器镜像构建工具。

模块结构:
    parser:     Dockerfile 解析, 将构建指令转换为有序步骤
    layer:      层管理, 文件系统层叠加与 whiteout 标记
    cache:      层缓存, 基于指令哈希与上下文的缓存命中
    builder:    构建执行, 逐指令在前一层基础上叠加变更
    packager:   镜像打包, 生成符合 OCI/Docker 规范的镜像结构
    context:    构建上下文, 管理传入的文件目录
    cli:        命令行入口
"""

__version__ = "0.1.0"

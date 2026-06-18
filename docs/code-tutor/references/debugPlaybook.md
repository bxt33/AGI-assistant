## 新手排错 SOP

当用户遇到报错时，按此流程引导排查。

### 第一步：看现象
- 完整报错信息是什么？
- 什么操作触发的？
- 能复现吗？

### 第二步：定位置
- 报错指向哪个文件、哪一行？
- 是编译时还是运行时？
- 是本地报错还是线上报错？

### 第三步：查上下文
- 这个文件/函数是做什么的？
- 最近改过什么？
- 有没有相关的 git diff？

### 第四步：找根因
- 是语法错误？
- 是类型错误？
- 是依赖缺失？
- 是环境配置问题？
- 是业务逻辑问题？

### 第五步：验证假设
- 加日志/print 确认变量值
- 注释掉可疑代码看是否消失
- 查官方文档或搜索报错信息

### 第六步：修复并复盘
- 写出修复方案
- 评估影响范围
- 跑测试验证
- 总结"为什么错 + 怎么避免"

---

## Python/FastAPI 常见报错速查

| 报错关键词 | 可能原因 | 排查方向 |
|-----------|---------|---------|
| `ImportError: No module named 'xxx'` | 依赖没装或路径错 | `pip install xxx`、检查 sys.path、检查 __init__.py |
| `ModuleNotFoundError: No module named 'src.xxx'` | 包路径不对 | 检查是否从项目根目录运行、检查 PYTHONPATH |
| `TypeError: __init__() missing required positional argument` | 缺少必传参数 | 检查 dataclass/类 构造函数参数是否传全 |
| `AttributeError: 'xxx' object has no attribute 'yyy'` | 属性/方法不存在 | 检查拼写、检查是不是 None（NoneType 报错） |
| `TypeError: 'NoneType' object is not callable` | 方法返回了 None | 检查函数是否忘记 return、检查变量覆盖 |
| `ValueError: too many values to unpack` | 解包数量不匹配 | 检查返回值和接收变量数量是否一致 |
| `KeyError: 'xxx'` | 字典 key 不存在 | 用 `dict.get()` 代替 `dict[]`、检查 key 拼写 |
| `IndentationError` | 缩进错误 | Tab 和空格混用、缩进层级不对 |
| `NameError: name 'xxx' is not defined` | 变量/函数未定义 | 检查拼写、检查导入、是否在定义前使用 |
| `RecursionError: maximum recursion depth exceeded` | 递归过深/死循环 | 检查退出条件、检查循环引用 |
| `RuntimeError: cannot schedule new futures after shutdown` | asyncio 事件循环已关闭 | 检查事件循环生命周期、避免多次关闭 |
| `ConnectionRefusedError` | 服务拒绝连接 | 检查目标服务是否启动、检查端口是否正确 |
| `ConnectionError: Max retries exceeded` | 连接失败（多次重试） | 检查网络、检查目标 URL、检查 DNS |
| `TimeoutError` | 操作超时 | 检查下游服务是否正常、检查超时配置 |
| `FileNotFoundError` | 文件/配置不存在 | 检查路径、检查当前工作目录 `os.getcwd()` |
| `PermissionError: [Errno 13]` | 权限不足 | 检查文件权限、检查是否其他进程占用 |
| `IsADirectoryError` | 把目录当文件操作了 | read/write 到了目录而不是文件 |
| `MemoryError` | 内存不足 | 数据量太大、检查是否存在内存泄漏 |
| `threading.RLock cannot be released` | 锁释放错误 | 检查是否有未加锁就释放的情况 |

## Go/tRPC 常见报错速查（遗留/混合项目参考）

| 报错关键词 | 可能原因 | 排查方向 |
|-----------|---------|---------|
| `undefined: xxx` | 函数/变量未定义 | 检查拼写、检查导入、检查是否在正确的包里 |
| `cannot use xxx as type yyy` | 类型不匹配 | 检查接口实现、检查参数类型 |
| `nil pointer dereference` | 空指针 | 检查变量是否初始化、检查返回值是否为 nil |
| `deadline exceeded` | 超时 | 检查下游服务是否正常、检查超时配置 |
| `connection refused` | 连接被拒 | 检查目标服务是否启动、检查地址和端口 |

---

## Python 项目排错工具箱

| 工具/方法 | 用途 | 使用场景 |
|-----------|------|---------|
| `logging.getLogger(__name__)` | 打印日志 | 在可疑位置加日志，看变量值和执行路径 |
| `print(obj.__dict__)` | 打印 dataclass 详情 | 看完整的数据对象内容 |
| `python -m pytest test_xxx.py` | 跑单个测试 | 验证某个函数的行为 |
| `python -c "import x; from pathlib import Path"` | 快速验证导入 | 检查模块能否正常导入 |
| `python -m py_compile file.py` | 语法编译检查 | 快速发现语法错误 |
| `python -m ruff check .` | 静态分析 | 发现潜在的代码问题 |
| `python -m mypy src/` | 类型检查 | 发现类型不匹配 |
| `python -c "import pdb; pdb.pm()"` | 交互式断点调试 | 在任意位置加 `breakpoint()` 进入调试 |
| `git diff` | 看改了什么 | 对比改动前后的差异 |
| `git log --oneline -10` | 看最近提交 | 找到最近谁改了什么 |
| `grep -r "关键词" .` | 全局搜索 | 找到某个函数/变量在哪里用了 |

---

## 环境问题排查

| 问题 | 排查步骤 |
|------|---------|
| 依赖拉不下来 | 1. 检查 pip 镜像源 → 2. 检查网络/VPN → 3. 检查 requirements.txt 中版本要求 |
| 服务启动失败 | 1. 看报错日志 → 2. 检查 config.yaml 路径 → 3. 检查端口是否被占用 |
| 本地跑不起来 | 1. 检查 Python 版本（>= 3.10） → 2. `pip install -r requirements.txt` → 3. 检查 config.yaml 中的连接地址 |
| 数据库连接失败 | 1. 检查 postgres/milvus/es 是否启动 → 2. 检查 docker-compose 状态 → 3. 检查 config.yaml 中 host:port |
| 调用 LLM 失败 | 1. 检查 API Key → 2. 检查 BaseURL → 3. 检查网络连通性 |

---

## 通用报错速查

| 报错关键词 | 可能原因 | 排查方向 |
|-----------|---------|---------|
| `undefined is not a function` | 调用了不存在的方法 | 检查拼写、检查导入 |
| `Cannot read property of null` | 访问了空对象的属性 | 检查数据是否正确加载 |
| `Module not found` | 依赖没装或路径错 | 检查 import 路径、运行 npm install |
| `CORS error` | 跨域被拦截 | 检查后端 CORS 配置 |
| `404 Not Found` | 接口地址错或服务没启动 | 检查 URL、检查服务状态 |
| `500 Internal Server Error` | 后端代码崩了 | 看后端日志 |
| `TypeError` | 类型不匹配 | 检查变量类型、检查接口返回 |
| `SyntaxError` | 语法写错了 | 检查括号、引号、逗号 |

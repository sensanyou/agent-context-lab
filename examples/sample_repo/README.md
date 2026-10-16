# demo_pkg

一个用于演示「代码仓库理解」工具的最小示例包。

分三层：

- `models.py` —— 数据模型 `Note`
- `storage.py` —— 内存存储 `InMemoryStore`
- `service.py` —— 业务逻辑 `NoteService` 与几个纯函数

可以试着让 ContextForge 回答：

- 这个项目的目录结构是什么样的？
- `NoteService` 里有哪些方法？分别在哪些文件里被调用？
- `rank_notes` 依赖了哪些模块？

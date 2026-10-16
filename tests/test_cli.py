"""CLI 层测试。

CLI 是用户接触项目的第一层，也是最容易被"最后一次改动"打破而没人发现的一层。
这里覆盖三类风险：

* 子命令是否真的注册成功（typo 不会在 import 时报错，只会在运行时炸）；
* 输出格式是否稳定（`--format json` 要能被程序解析）；
* 无 API Key 时是否依然可用（这是整个项目的核心承诺）。
"""

from __future__ import annotations

import json

import pytest

from contextforge.cli import build_parser, main
from contextforge.graph.workflow import NODE_ORDER


def test_all_subcommands_are_registered():
    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "command")
    names = set(action.choices)
    assert {
        "status",
        "workflow",
        "ingest",
        "search",
        "ask",
        "chat",
        "bench",
        "serve",
        "demo",
    } <= names


def test_workflow_mermaid_output(capsys, tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    exit_code = main(["workflow"])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert output.startswith("flowchart")
    assert "build_context" in output
    assert "finalize --> END" in output


def test_workflow_text_output_lists_every_node(capsys, tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    exit_code = main(["workflow", "--format", "text"])
    assert exit_code == 0

    output = capsys.readouterr().out
    for node in NODE_ORDER:
        assert node in output
    assert "条件边" in output


def test_workflow_json_output_is_parseable(capsys, tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    exit_code = main(["workflow", "--format", "json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["entry"] == "prepare"
    assert set(payload["nodes"]) == set(NODE_ORDER)
    # 工具节点必须能回到 build_context，否则上下文不会重建
    assert payload["conditional"]["tools"]["build_context"] == "build_context"


def test_unknown_subcommand_exits_nonzero(capsys):
    with pytest.raises(SystemExit):
        main(["not-a-command"])


def test_status_reports_offline_model(capsys, tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    exit_code = main(["status"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["llm"]["online"] is False
    assert payload["context"]["token_budget"] > 0


# --------------------------------------------------------------------------- #
def _isolate(monkeypatch, tmp_path) -> None:
    """把 CLI 的数据目录指向临时目录，避免污染仓库里的 data/。"""
    import dataclasses

    from contextforge.config import get_settings

    base = get_settings(refresh=True)
    scoped = dataclasses.replace(
        base,
        data_dir=tmp_path,
        docs_dir=tmp_path / "documents",
        db_path=tmp_path / "cli.db",
        trace_dir=tmp_path / "traces",
        model=dataclasses.replace(base.model, api_key=""),
        context=dataclasses.replace(base.context),
        retrieval=dataclasses.replace(base.retrieval),
        agent=dataclasses.replace(base.agent),
    )
    monkeypatch.setattr("contextforge.cli.get_settings", lambda refresh=False: scoped)
    monkeypatch.setattr("contextforge.config.get_settings", lambda refresh=False: scoped)
    scoped.ensure_dirs()

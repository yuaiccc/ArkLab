from arklab.web import demo_report, render_index


def test_web_index_contains_product_and_demo_metrics() -> None:
    html = render_index()

    assert "ArkLab" in html
    assert "RAG evaluation orchestration" in html
    assert "japanese-verb-master search eval" in html
    assert "arklab eval-jvm" in html


def test_demo_report_shape() -> None:
    report = demo_report()

    assert report["summary"]["cases"] == 75
    assert report["root_causes"]["retrieval_failure"] == 3
    assert report["failing_cases"]


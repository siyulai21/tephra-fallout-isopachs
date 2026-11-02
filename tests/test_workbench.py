import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from src.probabilistic_isopach.workbench import Workspace, handler_for, prepare_upload, preview_upload


def test_example_preview_excludes_pseudo_without_creating_a_run(tmp_path):
    result = preview_upload({"dataset": "mazama"}, tmp_path)
    assert result["measurements"] == 45
    assert result["excluded_pseudo"] == 7
    assert result["original_sha256"] == result["analysis_sha256"]
    assert list(tmp_path.iterdir()) == []


def test_vent_fields_can_be_added_without_altering_original_upload():
    original = "Lon,Lat,Thickness\n1,2,3\n"
    raw, analysis, name = prepare_upload({"csv": original, "name": "Example", "vent_lon": 0, "vent_lat": 1})
    assert raw.decode() == original
    assert b"Vent Lon,Vent Lat" in analysis
    assert name == "Example"
    with pytest.raises(ValueError, match="disagree"):
        prepare_upload({"dataset": "mazama", "vent_lon": 0, "vent_lat": 0})


def test_preview_reports_invalid_observations(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        preview_upload({"csv": "Lon,Lat,Thickness,Vent Lon,Vent Lat\n1,2,0,0,0\n"}, tmp_path)


def test_http_preview_and_local_origin_boundary(tmp_path):
    workspace = Workspace(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(workspace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"Preview measurements" in response.read()
        connection.request("POST", "/api/preview", json.dumps({"dataset": "tambora"}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["measurements"] == 26
        export = tmp_path/"test-run/report/isopachs.geojson"
        export.parent.mkdir(parents=True)
        export.write_text('{"type":"FeatureCollection","features":[]}')
        connection.request("GET", "/runs/test-run/report/isopachs.geojson")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["type"] == "FeatureCollection"
        connection.request("GET", "/atlas/../../README.md")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.request("POST", "/api/runs", "{}", {"Origin": "https://unrelated.example"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/runs/../outside/report/index.html")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

import pytest
from fastapi.testclient import TestClient

from app.api import app
from unittest.mock import patch

client = TestClient(app)


def mock_llm_generate(prompt: str, system_prompt: str = "", **kwargs):
    prompt_lower = prompt.lower()
    system_prompt.lower()

    # 1. Ghost Document
    if "zorglax" in prompt_lower:
        return "The CEO of Google is a purple alien named Zorglax."

    # 2. Needle in a Haystack
    if "alpha-charlie-99" in prompt_lower:
        return "The Server Backup Code is Alpha-Charlie-99."

    # 4. Hallucination Traps
    if "javascript" in prompt_lower:
        if (
            "python and rust" in prompt_lower
            and "javascript" not in prompt_lower.replace("javascript", "", 1)
        ):
            # Note: the query contains javascript, so we check if the context contains python and rust.
            return "KnowledgeForge AI only supports Python and Rust. It does not support JavaScript."

    if "quantum physics" in prompt_lower:
        return "I don't know the answer because the provided context does not mention quantum physics."

    # Default fallback
    return "I don't know the answer based on the context provided."


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Wipe DB and mock LLM before and after the brutal tests."""
    patcher = patch("app.api._llm.generate", new=mock_llm_generate)
    patcher.start()
    client.post("/reset_db")
    yield
    client.post("/reset_db")
    patcher.stop()


class TestBrutalAssault:

    def test_01_ghost_document_destruction(self, tmp_path):
        """Test if the system truly deletes a document and forgets it."""
        fake_doc = tmp_path / "zorglax.txt"
        fake_doc.write_text("The CEO of Google is a purple alien named Zorglax.")

        with open(fake_doc, "rb") as f:
            resp = client.post(
                "/upload", files=[("files", (fake_doc.name, f, "text/plain"))]
            )
        assert resp.status_code == 201

        query_resp = client.post(
            "/query", json={"query": "Who is the CEO of Google?", "top_k": 3}
        )
        assert query_resp.status_code == 200
        answer = query_resp.json()["answer"]
        assert "zorglax" in answer.lower(), f"Expected Zorglax in answer, got: {answer}"

        del_resp = client.delete(f"/documents/{fake_doc.name}")
        assert del_resp.status_code == 200

        query_resp_2 = client.post(
            "/query", json={"query": "Who is the CEO of Google?", "top_k": 3}
        )
        answer_2 = query_resp_2.json()["answer"]
        assert (
            "zorglax" not in answer_2.lower()
        ), f"GHOST DOCUMENT FOUND! Answer still contains Zorglax: {answer_2}"

    def test_02_needle_in_a_haystack(self, tmp_path):
        """Push Hybrid Search and Reranker with lots of noise."""
        files_to_upload = []
        for i in range(3):
            noise_file = tmp_path / f"haystack_{i}.txt"
            noise_file.write_text(
                "This is a random sentence that has nothing to do with anything. " * 500
            )
            files_to_upload.append(noise_file)

        needle_file = tmp_path / "needle.txt"
        needle_file.write_text("Server Backup Code: Alpha-Charlie-99")
        files_to_upload.append(needle_file)

        opened_files = [
            ("files", (f.name, open(f, "rb"), "text/plain")) for f in files_to_upload
        ]

        try:
            resp = client.post("/upload", files=opened_files)
            assert resp.status_code == 201

            q_resp = client.post(
                "/query", json={"query": "What is the Server Backup Code?", "top_k": 5}
            )
            answer = q_resp.json()["answer"]
            assert (
                "alpha-charlie-99" in answer.lower()
            ), f"Failed to retrieve needle! Answer: {answer}"
        finally:
            for _, f_tuple in opened_files:
                f_tuple[1].close()

    def test_03_api_stress_test(self, tmp_path):
        """Test API limits and graceful error handling."""
        big_file = tmp_path / "massive.txt"
        with open(big_file, "wb") as f:
            f.seek((50 * 1024 * 1024) + 1024)
            f.write(b"\0")

        with open(big_file, "rb") as f:
            resp = client.post(
                "/upload", files=[("files", (big_file.name, f, "text/plain"))]
            )
        assert resp.status_code == 413, f"Expected 413, got {resp.status_code}"

        query_resp = client.post("/query", json={"query": "", "top_k": 3})
        assert query_resp.status_code == 400

        del_resp = client.delete("/documents/fake_document_that_does_not_exist.pdf")
        assert del_resp.status_code == 404

    def test_04_hallucination_traps(self, tmp_path):
        """Test if Gemma model strictly obeys Context Builder boundaries."""
        rules_doc = tmp_path / "rules.txt"
        rules_doc.write_text("KnowledgeForge AI only supports Python and Rust.")

        with open(rules_doc, "rb") as f:
            client.post("/upload", files=[("files", (rules_doc.name, f, "text/plain"))])

        resp1 = client.post(
            "/query",
            json={"query": "Does KnowledgeForge support JavaScript?", "top_k": 3},
        )
        answer1 = resp1.json()["answer"].lower()

        # It should say no.
        assert "yes" not in answer1, f"LLM Hallucinated! Said yes to JS: {answer1}"
        assert "no" in answer1 or "not" in answer1, f"LLM failed to deny JS: {answer1}"

        resp2 = client.post(
            "/query", json={"query": "Explain quantum physics.", "top_k": 3}
        )
        answer2 = resp2.json()["answer"].lower()

        valid_rejections = [
            "don't know",
            "does not contain",
            "does not mention",
            "cannot answer",
            "no information",
            "not provided",
        ]
        assert (
            any(rej in answer2 for rej in valid_rejections)
            or "quantum physics" not in answer2
        ), f"LLM Hallucinated outside context! {answer2}"

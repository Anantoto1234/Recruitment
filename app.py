from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form
from pydantic import AliasChoices, BaseModel, Field
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
MATCH_THRESHOLD = 0.75


def resolve_model_path() -> str:
    base_dir = Path(__file__).resolve().parent
    direct_path = base_dir / "fine_tuned_sbert"
    nested_path = direct_path / "fine_tuned_sbert"

    if (direct_path / "modules.json").exists():
        return str(direct_path)
    if (nested_path / "modules.json").exists():
        return str(nested_path)

    raise FileNotFoundError("Could not find a valid fine_tuned_sbert model directory.")


# LOAD MODEL
model = SentenceTransformer(resolve_model_path())


class MatchRequest(BaseModel):
    cv_text: str
    jd_text: str = Field(
        validation_alias=AliasChoices("jd_text", "job_detail"),
        serialization_alias="jd_text",
    )


def predict_match(cv_text, jd_text):
    emb1 = model.encode(cv_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    is_match = score >= MATCH_THRESHOLD

    return {
        "score": round(score, 4),
        "is_match": is_match,
        "threshold": MATCH_THRESHOLD,
    }


@app.post("/match")
def match_cv_jd(data: MatchRequest):
    result = predict_match(data.cv_text, data.jd_text)
    return result


@app.post("/match-form")
def match_cv_jd_form(
    cv_text: Annotated[str, Form(...)],
    job_detail: Annotated[str, Form(...)],
):
    result = predict_match(cv_text, job_detail)
    return result

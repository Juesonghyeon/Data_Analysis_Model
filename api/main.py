"""
직업 추천 API 서버
한국직업정보 재직자조사 데이터 기반 코사인 유사도 직업 추천 + 대학/학과 연계

데이터 파일 위치: Data_Analysis_Model/data/
  - 직업별_진로특성_프로필.csv
  - 전국대학별학과정보.csv

실행: uvicorn main:app --reload --port 8001
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

# 피처 순서: 프론트엔드 mapScoresToFeatures() 반환 순서와 동일해야 함
FEATURES = ['성취/노력', '인내', '책임성과 진취성', '리더십', '혁신', '타인에대한 배려']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PROFILE_CSV = os.path.join(DATA_DIR, '직업별_진로특성_프로필.csv')
RAW_CSV = os.path.join(DATA_DIR, '한국직업정보 재직자조사 전처리 완료.csv')
UNI_CSV = os.path.join(DATA_DIR, '전국대학별학과정보.csv')

df_job: pd.DataFrame | None = None
df_uni: pd.DataFrame | None = None


def _load_profile() -> pd.DataFrame | None:
    """직업 프로필 CSV 로드. 없으면 원본에서 직접 생성."""
    if os.path.exists(PROFILE_CSV):
        df = pd.read_csv(PROFILE_CSV)
        df.columns = df.columns.str.strip()
        return df

    if os.path.exists(RAW_CSV):
        print("[INFO] 직업별_진로특성_프로필.csv 없음 → 원본 CSV에서 직접 생성")
        raw = pd.read_csv(RAW_CSV, low_memory=False)
        raw.columns = raw.columns.str.strip()

        # 원본 컬럼명 확인 후 매핑 (타인에 대한 배려 공백 차이 처리)
        col_map = {}
        for col in raw.columns:
            normalized = col.replace(' ', '')
            for feat in FEATURES:
                if feat.replace(' ', '') == normalized:
                    col_map[col] = feat
        raw = raw.rename(columns=col_map)

        available = [f for f in ['직업명'] + FEATURES if f in raw.columns]
        missing = [f for f in FEATURES if f not in available]
        if missing:
            print(f"[WARN] 원본 CSV에서 피처를 찾을 수 없음: {missing}")
            return None

        df = raw[available].dropna()
        df = df.groupby('직업명')[FEATURES].mean().reset_index()
        df[FEATURES] = df[FEATURES].round(2)
        print(f"[INFO] 원본에서 {len(df)}개 직업 프로필 생성 완료")
        return df

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df_job, df_uni

    df_job = _load_profile()
    if df_job is not None:
        print(f"[OK] 직업 프로필 로드 완료: {len(df_job)}개 직업")
    else:
        print("[WARN] 직업 데이터 없음 — data 폴더에 CSV 파일을 넣어주세요")

    if os.path.exists(UNI_CSV):
        df_uni = pd.read_csv(UNI_CSV, low_memory=False)
        df_uni.columns = df_uni.columns.str.strip()
        print(f"[OK] 대학 데이터 로드 완료: {len(df_uni):,}개 학과")
    else:
        print("[WARN] 전국대학별학과정보.csv 없음 — 대학 추천 기능 비활성")

    yield


app = FastAPI(title="직업 추천 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    user_scores: list[float]  # [성취/노력, 인내, 책임성과진취성, 리더십, 혁신, 배려] 각 1~5
    top_n: int = 4


@app.get("/health")
def health():
    return {
        "status": "ok",
        "jobs_loaded": df_job is not None,
        "jobs_count": len(df_job) if df_job is not None else 0,
        "uni_loaded": df_uni is not None,
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if df_job is None:
        return {"error": "직업 데이터가 로드되지 않았습니다. data 폴더에 CSV 파일을 넣어주세요."}

    if len(req.user_scores) != 6:
        return {"error": f"user_scores는 6개여야 합니다. (받은 개수: {len(req.user_scores)})"}

    job_matrix = df_job[FEATURES].values
    user_vector = np.array(req.user_scores, dtype=float).reshape(1, -1)
    sims = cosine_similarity(user_vector, job_matrix).flatten()
    top_indices = sims.argsort()[::-1][: req.top_n]

    result = []
    for rank, idx in enumerate(top_indices, start=1):
        job_name = df_job.iloc[idx]["직업명"]
        score = round(float(sims[idx]) * 100, 1)

        schools = []
        if df_uni is not None:
            matched = df_uni[
                df_uni["관련직업명"].astype(str).str.contains(job_name, na=False, regex=False)
            ]
            if not matched.empty:
                unique = (
                    matched[["시도명", "학교명", "학과명"]]
                    .drop_duplicates()
                    .head(4)
                )
                schools = unique.to_dict("records")

        result.append(
            {
                "rank": rank,
                "job_name": job_name,
                "similarity_score": score,
                "schools": schools,
            }
        )

    return result

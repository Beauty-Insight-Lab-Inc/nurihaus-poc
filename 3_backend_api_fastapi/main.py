import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# 1. FastAPI 앱 초기화
app = FastAPI(title="Nurihaus PoC AI API", description="Creator Matching & ROI Prediction")

# 2. 학습된 모델 로드 (서버 시작 시 한 번만 로드)
# 스크립트의 현재 위치를 기준으로 모델 파일의 절대 경로를 계산합니다.
# 이렇게 하면 어떤 위치에서 서버를 실행하더라도 항상 정확한 경로를 찾을 수 있습니다.
try:
    # 현재 파일(main.py)의 절대 경로
    current_file_path = os.path.abspath(__file__)
    # 현재 파일이 속한 디렉토리 (3_backend_api_fastapi)
    current_dir = os.path.dirname(current_file_path)
    # 프로젝트 루트 디렉토리 (current_dir의 상위 폴더)
    project_root = os.path.dirname(current_dir)
    # 프로젝트 루트에서부터 모델 파일까지의 전체 경로 조합
    MODEL_PATH = os.path.join(project_root, "2_recommendation_model", "saved_models", "roi_predictor.joblib")
    
    # 경로가 실제로 존재하는지 확인
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at the calculated path: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print(f">>> Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    print(f">>> FATAL: Failed to load model. Error: {e}")
    model = None

# 3. 요청 데이터 구조 정의 (Pydantic)
class CampaignRequest(BaseModel):
    follower_count: int
    niche: str
    platform: str
    budget: int  # 예산은 ROI 계산 후 매출 추정에 사용

# 4. 헬스 체크 엔드포인트 (서버 상태 확인용)
@app.get("/")
def read_root():
    return {"status": "active", "service": "Nurihaus AI PoC"}

# 5. 추천 및 예측 엔드포인트 (핵심)
@app.post("/predict")
def predict_roi(request: CampaignRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # 입력 데이터를 모델이 이해할 수 있는 DataFrame 형태로 변환
    input_data = pd.DataFrame([{
        'follower_count': request.follower_count,
        'niche': request.niche,
        'platform': request.platform
    }])

    try:
        # AI 예측 실행 (예상 ROI)
        predicted_roi = model.predict(input_data)[0]
        
        # 비즈니스 로직: 예상 매출 계산 (ROI * 예산)
        # ROI가 5.0이면 예산의 5배 효율이라는 뜻
        estimated_revenue = request.budget * predicted_roi

        return {
            "input_info": {
                "niche": request.niche,
                "platform": request.platform
            },
            "ai_analysis": {
                "predicted_roi": round(predicted_roi, 2),
                "estimated_revenue": round(estimated_revenue, 0),
                "confidence_score": "Low (Synthetic Data)" # 데모용 문구
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# 실행 방법 (터미널): uvicorn 3_backend_api_fastapi.main:app --reload
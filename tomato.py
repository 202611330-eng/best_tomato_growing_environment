import streamlit as st
import pandas as pd
import joblib  # 호환성이 더 좋은 joblib으로 변경
import pickle

# 웹화면 UI 구성
st.title("🌱 착과율 예측 프로그램")
st.write("아래 정보를 입력하시면 랜덤포레스트 모델을 통해 예측된 착과율을 보여줍니다.")
st.divider()

st.subheader("📊 환경 데이터 입력")
col1, col2, col3 = st.columns(3)

with col1:
    humidity = st.number_input("내부습도 (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
with col2:
    soil_temp = st.number_input("외부온도 (°C)", min_value=-50.0, max_value=60.0, value=20.0, step=0.1)
with col3:
    ground_temp = st.number_input("지온 (°C)", min_value=-50.0, max_value=60.0, value=15.0, step=0.1)

st.divider()

# 모델 로드 함수 (버전 호환성 해결을 위해 2단계 검증 적용)
@st.cache_resource
def load_model():
    try:
        # 1안: joblib으로 먼저 로드 시도 (버전 불일치 에러 해결용)
        return joblib.load('tomato_model.pkl')
    except Exception:
        # 2안: 안되면 기존 pickle 방식으로 백업 로드
        with open('tomato_model.pkl', 'rb') as f:
            return pickle.load(f)

# 예측 실행 버튼
if st.button("착과율 예측하기", type="primary"):
    try:
        # 모델 로드
        rf_model = load_model()
        
        # 입력 데이터를 DataFrame으로 변환
        input_data = pd.DataFrame(
            [[humidity, soil_temp, ground_temp]], 
            columns=['내부습도', '외부온도', '지온']
        )
        
        # 예측 및 결과 출력
        predicted = rf_model.predict(input_data)
        st.subheader("🔮 예측 결과")
        st.success(f"예측 착과율은 **{predicted[0]:.1f}%** 입니다.")
        
    except FileNotFoundError:
        st.error("❌ 'tomato_model.pkl' 파일을 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"❌ 호환성 에러 발생: {e}\n\n모델을 저장할 때 사용한 패키지와 현재 서버 버전에 차이가 있습니다.")
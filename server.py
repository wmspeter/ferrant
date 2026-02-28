import os
import json
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache
from collections import Counter

# --- CẤU HÌNH BIẾN MÔI TRƯỜNG ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("LỖI: Chưa cấu hình biến môi trường GOOGLE_API_KEY!")
    
genai.configure(api_key=GOOGLE_API_KEY)

db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="career_jobs_collection")

# --- Tải dữ liệu JSON gốc lên bộ nhớ để tra cứu siêu tốc ---
RAW_DATA_FILE = "all_jobs_data.json" 
jobs_database = {}

if os.path.exists(RAW_DATA_FILE):
    with open(RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        raw_list = json.load(f)
        for job in raw_list:
            job_id = job.get('job_id') or job.get('id')
            if job_id:
                jobs_database[str(job_id)] = job

app = FastAPI(title="Ferrant API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "message": "Xin chào! Ferrant API đang hoạt động cực kỳ ổn định."}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@lru_cache(maxsize=1000)
def enhance_query_with_gemini(user_query):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash') # Cập nhật lên 2.0 theo yêu cầu của bạn
        prompt = f"""Ngữ cảnh: Tìm việc IT.
        Yêu cầu người dùng: "{user_query}"
        
        Nhiệm vụ: Trích xuất và mở rộng các từ khóa chuyên ngành IT từ yêu cầu trên.
        Quy tắc: 
        1. Chỉ trả về chuỗi từ khóa (VD: frontend, React, HTML). Tuyệt đối không giải thích.
        2. Nếu câu hỏi là lời chào (hello, hi) hoặc không có từ khóa IT nào, hãy trả về chính xác câu gốc của người dùng.
        """
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=40, temperature=0.1)
        )
        if not response.parts: return user_query
        enhanced_query = response.text.strip()
        if not enhanced_query: return user_query
        print(f"[CACHE MISS] Vừa hỏi Gemini: '{user_query}' ---> '{enhanced_query}'")
        return enhanced_query
    except Exception as e:
        print(f"Lỗi Gemini: {e}")
        return user_query

def get_query_embedding(query_text):
    result = genai.embed_content(
        model="models/gemini-embedding-001", content=query_text, task_type="retrieval_query" 
    )
    return result['embedding']

# =======================================================
# ENDPOINT PING (ĐỂ LOADING SCREEN ĐÁNH THỨC SERVER)
# =======================================================
@app.get("/ping")
async def ping():
    return {"status": "ready"}

# =======================================================
# API TÌM KIẾM VÀ TÍNH TOÁN (GIỮ NGUYÊN LOGIC CỦA BẠN)
# =======================================================
@app.post("/api/search")
async def search_jobs_api(request: SearchRequest):
    try:
        smart_query = enhance_query_with_gemini(request.query)
        query_vector = get_query_embedding(smart_query)
        
        # MẸO: Quét dư số lượng (x2) để dự phòng cho các job bị loại bỏ do thiếu title
        results = collection.query(query_embeddings=[query_vector], n_results=request.top_k * 2)
        
        if not results['ids'][0]:
            return {"status": "success", "message": "Không tìm thấy", "data": []}

        jobs_data = []
        skill_counter = Counter()
        location_stats = {}

        for i in range(len(results['ids'][0])):
            if len(jobs_data) >= request.top_k:
                break

            job_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            
            raw_job = jobs_database.get(job_id, {})
            job_title = raw_job.get("job_title", "")
            
            if not job_title or str(job_title).strip() == "" or str(job_title).lower() == "nan":
                continue
                
            location = raw_job.get("location", "")
            if not location or str(location).strip() == "" or str(location).lower() == "nan":
                location = "Thỏa thuận"
            else:
                location = str(location).strip()

            # Xử lý lương
            est_min = metadata.get('estimated_min', 0)
            est_max = metadata.get('estimated_max', 0)
            
            raw_avg = 0
            if est_min > 0 and est_max > 0:
                display_salary = f"${int(est_min)} - ${int(est_max)} / năm"
                raw_avg = (est_min + est_max) / 2
            elif est_min > 0:
                display_salary = f"Từ ${int(est_min)} / năm"
                raw_avg = est_min
            else:
                display_salary = metadata.get('salary_original', 'Thỏa thuận')

            # Biểu đồ lương
            if raw_avg > 0:
                short_loc = location.split(',')[0].split('-')[0].strip() 
                if not short_loc:
                    short_loc = "Thỏa thuận"
                    
                if short_loc not in location_stats:
                    location_stats[short_loc] = {"total": 0, "count": 0}
                location_stats[short_loc]["total"] += raw_avg
                location_stats[short_loc]["count"] += 1

            # Biểu đồ skill
            skills_str = metadata.get('skills', '')
            if skills_str:
                job_skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                for skill in job_skills:
                    skill_counter[skill] += 1

            job_info = {
                "job_id": job_id,
                "match_score": round(results['distances'][0][i], 4),
                "salary": display_salary, 
                "experience_level": metadata.get('experience_level', ''),
                "skills": skills_str,
                "job_title": job_title,
                "job_description": raw_job.get("job_description", "Không có mô tả chi tiết."),
                "url": raw_job.get("url", "#"),
                "location": location
            }
            jobs_data.append(job_info)

        top_skills = skill_counter.most_common(10)
        skills_chart_data = [{"skill": skill, "count": count} for skill, count in top_skills]

        salary_chart_data = []
        for loc, stats in location_stats.items():
            if stats["count"] > 0:
                salary_chart_data.append({
                    "location": loc, 
                    "average_salary": round(stats["total"] / stats["count"])
                })
        salary_chart_data.sort(key=lambda x: x["average_salary"], reverse=True)

        return {
            "status": "success", 
            "data": jobs_data, 
            "skills_chart_data": skills_chart_data,
            "salary_chart_data": salary_chart_data
        }

    except Exception as e:
        print(f"LỖI NGẦM TRÊN SERVER: {str(e)}") 
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}

import os
import json
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache
from collections import Counter

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

# =======================================================
# API 1: TRẢ VỀ DỮ LIỆU BIỂU ĐỒ SKILL (CHART.JS)
# =======================================================
@app.get("/api/skills")
async def get_top_skills(query: str = ""):
    try:
        if not query.strip():
            return {"status": "success", "data": []}

        smart_query = enhance_query_with_gemini(query)
        query_vector = get_query_embedding(smart_query)
        results = collection.query(query_embeddings=[query_vector], n_results=50)
        
        if not results['metadatas'] or not results['metadatas'][0]:
            return {"status": "success", "data": []}
            
        skill_counter = Counter()
        
        for metadata in results['metadatas'][0]:
            skills_str = metadata.get('skills', '')
            if skills_str:
                job_skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                for skill in job_skills:
                    skill_counter[skill] += 1
                    
        top_skills = skill_counter.most_common(10)
        response_data = [{"skill": skill, "count": count} for skill, count in top_skills]
        
        return {"status": "success", "data": response_data}
        
    except Exception as e:
        print(f"LỖI NGẦM TẠI API SKILLS: {str(e)}")
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}

# =======================================================
# API 2: TRẢ VỀ BIỂU ĐỒ LƯƠNG TRUNG BÌNH THEO KHU VỰC
# =======================================================
@app.get("/api/salary-by-location")
async def get_salary_by_location(query: str = ""):
    try:
        if not query.strip():
            return {"status": "success", "data": []}

        smart_query = enhance_query_with_gemini(query)
        query_vector = get_query_embedding(smart_query)
        results = collection.query(query_embeddings=[query_vector], n_results=50)

        if not results['metadatas'] or not results['metadatas'][0]:
            return {"status": "success", "data": []}

        location_stats = {}

        for i in range(len(results['ids'][0])):
            job_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]

            # Lấy vị trí từ DB trên RAM
            location = "Khác"
            if job_id in jobs_database:
                location = jobs_database[job_id].get("location", "Khác")
            if not location or location.strip() == "":
                location = "Khác"
            
            # Nếu chuỗi địa điểm dài kiểu "Hà Nội, Hồ Chí Minh", chỉ lấy tỉnh đầu tiên cho biểu đồ bớt rối
            location = location.split(',')[0].split('-')[0].strip()

            est_min = metadata.get('estimated_min', 0)
            est_max = metadata.get('estimated_max', 0)

            # Bỏ qua các job không thể estimate ra số
            if est_min <= 0 and est_max <= 0:
                continue

            # Tính mức lương trung bình của job này
            if est_min > 0 and est_max > 0:
                job_avg_salary = (est_min + est_max) / 2
            else:
                job_avg_salary = est_min if est_min > 0 else est_max

            if location not in location_stats:
                location_stats[location] = {"total": 0, "count": 0}

            location_stats[location]["total"] += job_avg_salary
            location_stats[location]["count"] += 1

        # Tính trung bình cho từng tỉnh
        response_data = []
        for loc, stats in location_stats.items():
            if stats["count"] > 0:
                avg_province_salary = stats["total"] / stats["count"]
                response_data.append({
                    "location": loc, 
                    "average_salary": round(avg_province_salary)
                })

        # Sắp xếp biểu đồ theo mức lương giảm dần
        response_data.sort(key=lambda x: x["average_salary"], reverse=True)

        return {"status": "success", "data": response_data}
        
    except Exception as e:
        print(f"LỖI TẠI API SALARY: {str(e)}")
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}

# =======================================================
# API 3: TÌM KIẾM AI CHÍNH
# =======================================================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@lru_cache(maxsize=1000)
def enhance_query_with_gemini(user_query):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Ngữ cảnh: Tìm việc IT.
        Yêu cầu người dùng: "{user_query}"
        
        Nhiệm vụ: Trích xuất và mở rộng các từ khóa chuyên ngành IT từ yêu cầu trên.
        Quy tắc: 
        1. Chỉ trả về chuỗi từ khóa (VD: frontend, React, HTML). Tuyệt đối không giải thích.
        2. Nếu câu hỏi là lời chào (hello, hi) hoặc không có từ khóa IT nào, hãy trả về chính xác câu gốc của người dùng.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=40,
                temperature=0.1,
            )
        )
        
        if not response.parts:
            return user_query
            
        enhanced_query = response.text.strip()
        if not enhanced_query:
            return user_query
            
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

@app.post("/api/search")
async def search_jobs_api(request: SearchRequest):
    try:
        smart_query = enhance_query_with_gemini(request.query)
        query_vector = get_query_embedding(smart_query)
        results = collection.query(query_embeddings=[query_vector], n_results=request.top_k)
        
        if not results['ids'][0]:
            return {"status": "success", "message": "Không tìm thấy", "data": []}

        jobs_data = []
        for i in range(len(results['ids'][0])):
            job_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            
            est_min = metadata.get('estimated_min', 0)
            est_max = metadata.get('estimated_max', 0)
            
            if est_min > 0 and est_max > 0:
                display_salary = f"${int(est_min)} - ${int(est_max)} / năm"
            elif est_min > 0:
                display_salary = f"Từ ${int(est_min)} / năm"
            else:
                display_salary = metadata.get('salary_original', 'Thỏa thuận')

            job_info = {
                "job_id": job_id,
                "match_score": round(results['distances'][0][i], 4),
                "salary": display_salary, 
                "experience_level": metadata.get('experience_level', ''),
                "skills": metadata.get('skills', ''),
                "job_title": "Chưa cập nhật",
                "job_description": "Không có mô tả chi tiết.",
                "url": "#",
                "location": "Chưa cập nhật địa điểm"
            }
            
            if job_id in jobs_database:
                raw_job = jobs_database[job_id]
                job_info["job_title"] = raw_job.get("job_title", job_info["job_title"])
                job_info["job_description"] = raw_job.get("job_description", job_info["job_description"])
                job_info["url"] = raw_job.get("url", job_info["url"])
                job_info["location"] = raw_job.get("location", "Chưa cập nhật địa điểm")

            jobs_data.append(job_info)
            
        return {"status": "success", "data": jobs_data}
    except Exception as e:
        print(f"LỖI NGẦM TRÊN SERVER: {str(e)}") 
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}

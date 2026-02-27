import os
import json
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functools import lru_cache

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
            jobs_database[job.get('id')] = job

app = FastAPI(title="Ferrant API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

# Cổng chào mở cho cả GET và HEAD để Render không báo lỗi 405
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "message": "Xin chào! Ferrant API đang hoạt động cực kỳ ổn định."}

# =======================================================
# MỚI: API TRẢ VỀ DỮ LIỆU BIỂU ĐỒ CHO CHART.JS
# =======================================================
@app.get("/api/skills")
async def get_top_skills():
    return [
        { "skill": "Python", "count": 320 },
        { "skill": "ReactJS", "count": 180 },
        { "skill": "JavaScript", "count": 290 },
        { "skill": "SQL", "count": 250 },
        { "skill": "AWS", "count": 120 },
        { "skill": "Java", "count": 210 },
        { "skill": "Node.js", "count": 160 },
        { "skill": "C++", "count": 140 },
        { "skill": "Docker", "count": 110 },
        { "skill": "Golang", "count": 90 }
    ]

# =======================================================
# API TÌM KIẾM AI
# =======================================================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@lru_cache(maxsize=1000)
def enhance_query_with_gemini(user_query):
    """Dùng Gemini Flash với Cache và bắt lỗi 'cạn lời'"""
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
        
        # Chặn đứng lỗi "cạn lời": Kiểm tra xem nó có nhả ra cái "Part" nào không
        if not response.parts:
            return user_query
            
        enhanced_query = response.text.strip()
        
        # Tránh trường hợp nó nhả ra chuỗi rỗng
        if not enhanced_query:
            return user_query
            
        print(f"[CACHE MISS] Vừa hỏi Gemini: '{user_query}' ---> '{enhanced_query}'")
        return enhanced_query
        
    except Exception as e:
        # Nếu có bất kỳ lỗi gì khác, cứ âm thầm dùng lại câu gốc
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
        # 1. Nhờ Gemini Flash "dịch" và mở rộng câu hỏi
        smart_query = enhance_query_with_gemini(request.query)
        
        # 2. Mang câu hỏi đã được độ đi biến thành Vector
        query_vector = get_query_embedding(smart_query)
        
        # 3. Tìm kiếm trong ChromaDB
        results = collection.query(query_embeddings=[query_vector], n_results=request.top_k)
        
        if not results['ids'][0]:
            return {"status": "success", "message": "Không tìm thấy", "data": []}

        jobs_data = []
        for i in range(len(results['ids'][0])):
            job_id = results['ids'][0][i]
            
            # Khởi tạo data cơ bản từ DB
            job_info = {
                "job_id": job_id,
                "match_score": round(results['distances'][0][i], 4),
                "salary": results['metadatas'][0][i].get('salary'),
                "experience_level": results['metadatas'][0][i].get('experience_level'),
                "skills": results['metadatas'][0][i].get('skills'),
                "job_title": "Chưa cập nhật",
                "job_description": "Không có mô tả chi tiết.",
                "url": "#"
            }
            
            # Ghép nối data từ file gốc vào
            if job_id in jobs_database:
                raw_job = jobs_database[job_id]
                job_info["job_title"] = raw_job.get("job_title", job_info["job_title"])
                job_info["job_description"] = raw_job.get("job_description", job_info["job_description"])
                job_info["url"] = raw_job.get("url", job_info["url"])

            jobs_data.append(job_info)
            
        return {"status": "success", "data": jobs_data}
    except Exception as e:
        # ÉP TRẢ VỀ LỖI BẰNG JSON ĐỂ TRÌNH DUYỆT KHÔNG BÁO CORS
        print(f"LỖI NGẦM TRÊN SERVER: {str(e)}") 
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}

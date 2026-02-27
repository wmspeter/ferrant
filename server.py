import os
import json
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("LỖI: Chưa cấu hình biến môi trường GOOGLE_API_KEY!")
    
genai.configure(api_key=GOOGLE_API_KEY)

db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="career_jobs_collection")

# --- ĐIỂM MỚI: Tải dữ liệu JSON gốc lên bộ nhớ để tra cứu siêu tốc ---
# Đổi tên file này cho đúng với file chứa data scrape của bạn nhé
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
# Cổng chào để Render Health Check không bị lỗi 404
# Cổng chào mở cho cả GET và HEAD để Render không báo lỗi 405
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "message": "Xin chào! Ferrant API đang hoạt động cực kỳ ổn định."}
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
def enhance_query_with_gemini(user_query):
    """Dùng Gemini Flash để sửa chính tả và mở rộng từ khóa chuyên ngành IT"""
    try:
        # Dùng bản flash cho tốc độ xử lý siêu nhanh (chỉ tốn khoảng 0.5s - 1s)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f""'Bạn là một chuyên gia Headhunter mảng IT. 
        Khách hàng đang tìm việc với yêu cầu: "{user_query}"
        
        Nhiệm vụ của bạn:
        1. Sửa lỗi chính tả nếu có.
        2. Chuyển đổi các từ lóng sang ngôn ngữ chuyên ngành (Ví dụ: "giao diện" -> "frontend, UI/UX, React, Web", "mới ra trường" -> "fresher, junior, không yêu cầu kinh nghiệm").
        3. CHỈ TRẢ VỀ một chuỗi các từ khóa cách nhau bằng dấu phẩy, KHÔNG giải thích, KHÔNG chào hỏi.
        """
        
        response = model.generate_content(prompt)
        enhanced_query = response.text.strip()
        
        print(f"[AI Tối ưu] Gốc: '{user_query}' ---> Mới: '{enhanced_query}'")
        return enhanced_query
    except Exception as e:
        print(f"Lỗi khi gọi Gemini enhance: {e}")
        # Nếu lỗi (mạng/API), trả về câu gốc để hệ thống không bị sập
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
        print(f"LỖI NGẦM TRÊN SERVER: {str(e)}") # Dòng này in ra log Render
        return {"status": "error", "message": f"Lỗi server: {str(e)}", "data": []}






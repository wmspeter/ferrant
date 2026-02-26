import os
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Bảo mật API Key: Lấy từ cấu hình bí mật của Render, KHÔNG hardcode
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("LỖI: Chưa cấu hình biến môi trường GOOGLE_API_KEY trên Render!")
    
genai.configure(api_key=GOOGLE_API_KEY)

# 2. Sửa lại đường dẫn DB: Dùng đường dẫn tương đối "./chroma_db"
# Nó sẽ tự động tìm thư mục chroma_db nằm ngay cạnh file server.py
db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="career_jobs_collection")

# 3. Khởi tạo FastAPI
app = FastAPI(title="CareerMap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

def get_query_embedding(query_text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query_text,
        task_type="retrieval_query" 
    )
    return result['embedding']

@app.post("/api/search")
async def search_jobs_api(request: SearchRequest):
    try:
        query_vector = get_query_embedding(request.query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=request.top_k
        )
        
        if not results['ids'][0]:
            return {"status": "success", "message": "Không tìm thấy", "data": []}

        jobs_data = []
        for i in range(len(results['ids'][0])):
            jobs_data.append({
                "job_id": results['ids'][0][i],
                "match_score": round(results['distances'][0][i], 4),
                "salary": results['metadatas'][0][i].get('salary'),
                "experience_level": results['metadatas'][0][i].get('experience_level'),
                "skills": results['metadatas'][0][i].get('skills')
            })
            
        return {"status": "success", "data": jobs_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bỏ phần if __name__ == "__main__": vì Render sẽ dùng Uvicorn để gọi thẳng app
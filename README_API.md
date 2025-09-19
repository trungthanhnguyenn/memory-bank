# Memory Block API Documentation

## Tổng quan
  
API này cung cấp 2 endpoints chính để quản lý memory blocks theo đúng logic bạn yêu cầu:

1. **Endpoint 1**: Tạo/cập nhật memory blocks 
2. **Endpoint 2**: Lấy memory blocks đã lưu

## Khởi động API Server

```bash
cd /path/to/memory-bank
python memory_bank/src/api/memory_endpoints.py --host 0.0.0.0 --port 8000
```

## Endpoint 1: Tạo/Cập nhật Memory Block

### URL
```
POST /api/v1/memory-block
```

### Logic hoạt động
1. **Lần gọi đầu tiên** (chỉ có `user_id` và `session_id`):
   - Kiểm tra postgres_session đã tồn tại chưa
   - Nếu chưa có: tạo mới postgres_session 
   - Lấy conversation history từ docman dựa trên session_id ban đầu
   - Tạo summary bằng LLM
   - Lưu summary như role="user" vào docman sử dụng postgres_session
   - Trả về postgres_session để tracking

2. **Lần gọi thứ hai** (có thêm `answer`):
   - Sử dụng postgres_session đã có
   - Thêm answer như role="assistant" vào history
   - Hoàn thiện memory block

### Request Body

#### Lần 1 - Tạo summary:
```json
{
  "user_id": "user123",
  "session_id": "session456"
}
```

#### Lần 2 - Thêm answer:
```json
{
  "user_id": "user123", 
  "session_id": "session456",
  "answer": "Đây là câu trả lời từ các agents downstream"
}
```

### Response

#### Lần 1:
```json
{
  "success": true,
  "message": "Summary block created successfully, waiting for answer",
  "data": [
    {
      "info_docman": {
        "user_id": "user123",
        "session_id": "session456", 
        "postgres_session": "temp_session_789"
      },
      "summary_block": {
        "conversation_summary": "Tóm tắt conversation...",
        "answer": "",
        "is_complete": false
      },
      "postgres_session": "temp_session_789",
      "status": "pending_answer"
    }
  ],
  "postgres_session": "temp_session_789"
}
```

#### Lần 2:
```json
{
  "success": true,
  "message": "Memory block completed successfully",
  "data": [
    {
      "info_docman": {
        "user_id": "user123",
        "session_id": "session456",
        "postgres_session": "temp_session_789" 
      },
      "summary_block": {
        "conversation_summary": "Tóm tắt conversation...",
        "answer": "Đây là câu trả lời từ các agents downstream",
        "is_complete": true
      },
      "postgres_session": "temp_session_789",
      "status": "complete"
    }
  ],
  "postgres_session": "temp_session_789"
}
```

## Endpoint 2: Lấy Memory Blocks

### URL
```
POST /api/v1/memory-block/retrieve
```

### Logic hoạt động
1. Tìm postgres_session dựa trên user_id + session_id trong PostgresDB
2. Nếu không tìm thấy: trả về empty
3. Nếu tìm thấy: lấy conversation history từ docman bằng postgres_session
4. Phân tích history thành các memory blocks (pairs of user/assistant)

### Request Body
```json
{
  "user_id": "user123",
  "session_id": "session456"
}
```

### Response

#### Có memory blocks:
```json
{
  "success": true,
  "message": "Retrieved 2 memory block(s)",
  "postgres_session": "temp_session_789",
  "memory_blocks": [
    {
      "conversation_summary": "Tóm tắt conversation 1...",
      "answer": "Câu trả lời 1...", 
      "is_complete": true,
      "timestamp": null
    },
    {
      "conversation_summary": "Tóm tắt conversation 2...",
      "answer": "Câu trả lời 2...",
      "is_complete": true, 
      "timestamp": null
    }
  ]
}
```

#### Không có memory blocks:
```json
{
  "success": true,
  "message": "No memory blocks found - postgres session does not exist",
  "postgres_session": null,
  "memory_blocks": []
}
```

## Health Check

### URL
```
GET /api/v1/health
```

### Response
```json
{
  "status": "healthy",
  "service": "Memory Block API"
}
```

## Ví dụ sử dụng với curl

### 1. Tạo summary block
```bash
curl -X POST "http://localhost:8000/api/v1/memory-block" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "session456"
  }'
```

### 2. Cập nhật với answer
```bash
curl -X POST "http://localhost:8000/api/v1/memory-block" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123", 
    "session_id": "session456",
    "answer": "Đây là câu trả lời từ downstream agents"
  }'
```

### 3. Lấy memory blocks
```bash
curl -X POST "http://localhost:8000/api/v1/memory-block/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "session456"
  }'
```

## Lưu ý quan trọng

1. **Environment Variables**: Đảm bảo có đủ các biến môi trường cho LLM và Postgres:
   ```bash
   API_KEY=your_openai_key
   BASE_URL=https://api.openai.com/v1
   MODEL_NAME=gpt-3.5-turbo
   POSTGRES_DB=your_db
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

2. **Dependencies**: Chạy `pip install -r requirements.txt` trước khi start API

3. **Postgres Database**: Đảm bảo PostgresDB đã running và có thể kết nối được

4. **Docman API**: Đảm bảo BASE_URL đã đúng và docman API hoạt động bình thường

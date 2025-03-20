curl -X POST "http://localhost:23123/detect-face-attention" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@in.jpeg" \
     -F "face_id=user123" \
     -F "lecture_id=lecture123" \
     -F "timestamp=2025-03-06T23:17:05.664Z"

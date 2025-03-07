# Attention Detection

## Overview
The **Attention Detection API** is a machine learning-based system that analyzes images of students to determine their attention levels in a classroom setting. It processes images using a deep learning model and stores results in a PostgreSQL database.

## Features
- Detects whether a person is **focused** or **unfocused** from an image.
- Uses a pre-trained **Gazelle model** for gaze detection.
- Stores attention records in a **PostgreSQL database** via **Prisma ORM**.
- Built with **FastAPI** for high-performance API endpoints.
- Containerized with **Docker** and **Docker Compose**.

## Project Structure
```
📂 attention-detection
├── Dockerfile                 # Defines the container setup
├── docker-compose.yml         # Orchestrates services (API + Database)
├── app.py                     # Main FastAPI application
├── prisma.schema              # Database schema definition
├── requirements.txt           # Python dependencies
├── workflows
│   ├── build-test.yml         # GitHub Actions CI/CD workflow
└── README.md                  # Project documentation
```

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.10+**
- **Docker & Docker Compose**
- **PostgreSQL** (optional, if not using Docker)

### Setup & Run
1. **Clone the repository**
   ```sh
   git clone https://github.com/your-repo/attention-detection.git
   cd attention-detection
   ```

2. **Set up environment variables**
   Create a `.env` file with:
   ```sh
   DATABASE_URL=postgresql://user:password@db:5432/student_attention
   ```

3. **Start the application with Docker**
   ```sh
   docker-compose up --build
   ```

4. **Access the API**
   - API Docs: [http://localhost:5000/docs](http://localhost:5000/docs)
   - Health Check: [http://localhost:5000/](http://localhost:5000/)

## API Endpoints
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/detect-face-attention` | Analyze attention from an image |
| `GET`  | `/` | Health check |

## Attention Status Values
- **FOCUSED**: Student is looking at the teacher or relevant material.
- **UNFOCUSED**: Student is distracted, looking away, or not engaged.

## Error Handling
The API includes comprehensive error handling for:
- Invalid file types
- File size limits (max 10MB)
- Invalid timestamp formats
- Database connection issues
- Rate limiting (10 requests per minute)

## Database Schema
The database uses **Prisma ORM** with PostgreSQL.
```prisma
model AttentionSchema {
  studentId  String   @id
  lectureId  String   @id
  timestamp  DateTime @id
  attention  Float
}
```

## Security Considerations
### Environment Variables:
- Never commit `.env` files to version control.
- Use secure credentials in production.
- Add `.env` to `.gitignore`.

### Rate Limiting:
- Configured to 10 requests per minute per IP.
- Can be adjusted in the code if needed.

### File Upload Security:
- File size limited to 10MB.
- Only image files (JPEG/PNG) are accepted.
- File content validation before processing.

## Troubleshooting
### Database Connection Issues:
- Verify PostgreSQL container is running.
- Check `DATABASE_URL` in `.env`.
- Ensure database exists and is accessible.

### Image Processing Errors:
- Verify image format (JPEG/PNG only).
- Check image is not corrupted.
- Ensure image size is within limits.

### Attention Detection Issues:
- Ensure good lighting in images.
- Check if students' faces are clearly visible.
- Verify confidence thresholds in `.env`.

## GitHub Actions CI/CD
This project includes a **GitHub Actions** workflow to automate building and testing the container:
```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```
### Steps:
- Checkout repository
- Set up Docker Buildx
- Build the Docker image
- Run the container for testing


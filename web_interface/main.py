from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from typing import Annotated

app = FastAPI()
templates = Jinja2Templates(directory="./")
app.mount("/static", StaticFiles(directory="./"), name="static")


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/files")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/upload-file")
async def create_upload_file(file: UploadFile):
    allowed_file_types = ["image/jpeg", "image/png"]  # Validate file types
    if file.content_type not in allowed_file_types:
        return {"error": "Unsupported file type. Please upload a JPEG or PNG image."}

    # Validate file size
    max_file_size = 5 * 1024 * 1024
    file_content = await file.read()
    file_size = len(file_content)
    if file_size > max_file_size:
        return {"error": "File size exceeds the 5MB limit."}

    await file.seek(0) 
    return {"filename": file.filename, "file_size": file_size}

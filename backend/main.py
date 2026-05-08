from fastapi import FastAPI, HTTPException, Depends, Response, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import bcrypt
from jose import JWTError, jwt
import pymupdf
import re
import pymupdf
import subprocess
import tempfile
import httpx
import secrets
import json

app = FastAPI(title="Health Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.getenv("DB_PATH", "/data/health.db")
UPLOADS_PATH = os.getenv("UPLOADS_PATH", "/data/uploads")
SECRET_KEY=secrets.token_hex(32)
ALGORITHM = "HS256"
COOKIE_NAME = "health_session"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.2:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# ==================== DATABASE ====================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(UPLOADS_PATH).mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            first_name TEXT,
            last_name TEXT,
            birth_date TEXT,
            blood_type TEXT,
            height_cm REAL,
            weight_kg REAL,
            allergies TEXT,
            chronic_conditions TEXT,
            doctor_name TEXT,
            doctor_phone TEXT,
            notes TEXT,
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lab_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            lab_name TEXT,
            notes TEXT,
            file_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lab_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            value REAL,
            value_text TEXT,
            unit TEXT,
            ref_min REAL,
            ref_max REAL,
            ref_text TEXT,
            is_abnormal INTEGER DEFAULT 0,
            FOREIGN KEY (result_id) REFERENCES lab_results(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS medical_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'inne',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            category TEXT DEFAULT 'inne',
            description TEXT,
            date TEXT,
            tags TEXT,
            entity_type TEXT,
            entity_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS history_files (
            history_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            PRIMARY KEY (history_id, file_id),
            FOREIGN KEY (history_id) REFERENCES medical_history(id) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ==================== AUTH ====================

def create_token(user_id: int, username: str, is_admin: bool) -> str:
    return jwt.encode(
        {"sub": str(user_id), "username": username, "is_admin": is_admin},
        SECRET_KEY, algorithm=ALGORITHM
    )

def get_current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"id": int(payload["sub"]), "username": payload["username"], "is_admin": payload.get("is_admin", False)}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_admin(user=Depends(get_current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin required")
    return user

def user_upload_dir(user_id: int) -> Path:
    p = Path(UPLOADS_PATH) / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ==================== MODELS ====================

class AuthIn(BaseModel):
    username: str
    password: str

class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str

class AdminSetPasswordIn(BaseModel):
    new_password: str

class CreateUserIn(BaseModel):
    username: str
    password: str
    is_admin: bool = False

class ProfileIn(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birth_date: Optional[str] = None
    blood_type: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_phone: Optional[str] = None
    notes: Optional[str] = None

class LabParameterIn(BaseModel):
    name: str
    value: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    ref_text: Optional[str] = None
    is_abnormal: bool = False

class LabResultIn(BaseModel):
    date: str
    lab_name: Optional[str] = None
    notes: Optional[str] = None
    parameters: List[LabParameterIn] = []

class HistoryIn(BaseModel):
    date: str
    title: str
    description: Optional[str] = None
    category: str = "inne"
    file_ids: List[int] = []

class FileMetaIn(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None
    date: Optional[str] = None
    tags: Optional[str] = None

# ==================== SETUP / AUTH ENDPOINTS ====================

@app.get("/auth/setup-required")
def setup_required():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return {"setup_required": count == 0}

@app.post("/auth/setup")
def setup(data: AuthIn, response: Response):
    conn = get_db()
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
        conn.close()
        raise HTTPException(status_code=400, detail="Setup already completed")
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, 1, ?)",
        (data.username, bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode(), datetime.now().isoformat())
    )
    conn.commit()
    token = create_token(cur.lastrowid, data.username, True)
    conn.close()
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=365*24*3600)
    return {"username": data.username, "is_admin": True}

@app.post("/auth/login")
def login(data: AuthIn, response: Response):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (data.username,)).fetchone()
    conn.close()
    if not user or not bcrypt.checkpw(data.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło")
    token = create_token(user["id"], user["username"], bool(user["is_admin"]))
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=365*24*3600)
    return {"username": user["username"], "is_admin": bool(user["is_admin"])}

@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}

@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"], "is_admin": user["is_admin"]}

@app.post("/auth/change-password")
def change_password(data: ChangePasswordIn, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
    if not bcrypt.checkpw(data.current_password.encode(), row["password_hash"].encode()):
        conn.close()
        raise HTTPException(status_code=401, detail="Nieprawidłowe obecne hasło")
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
        (bcrypt.hashpw(data.new_password.encode(), bcrypt.gensalt()).decode(), user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True}

# ==================== ADMIN ====================

@app.get("/admin/users")
def list_users(user=Depends(require_admin)):
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id").fetchall()
    result = []
    for u in users:
        result.append({"id": u["id"], "username": u["username"], "is_admin": bool(u["is_admin"]), "created_at": u["created_at"]})
    conn.close()
    return result

@app.post("/admin/users", status_code=201)
def create_user(data: CreateUserIn, user=Depends(require_admin)):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
            (data.username, bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode(), int(data.is_admin), datetime.now().isoformat())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Użytkownik '{data.username}' już istnieje")
    conn.close()
    return {"username": data.username}

@app.delete("/admin/users/{user_id}")
def delete_user(user_id: int, user=Depends(require_admin)):
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Nie możesz usunąć własnego konta")
    conn = get_db()
    if not conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    # delete files from disk
    upload_dir = Path(UPLOADS_PATH) / str(user_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"deleted": user_id}

@app.put("/admin/users/{user_id}/password")
def admin_set_password(user_id: int, data: AdminSetPasswordIn, user=Depends(require_admin)):
    conn = get_db()
    if not conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
        (bcrypt.hashpw(data.new_password.encode(), bcrypt.gensalt()).decode(), user_id))
    conn.commit()
    conn.close()
    return {"ok": True}

# ==================== PROFILE ====================

@app.get("/api/profile")
def get_profile(user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user["id"],)).fetchone()
    conn.close()
    return dict(row) if row else {}

@app.put("/api/profile")
def save_profile(data: ProfileIn, user=Depends(get_current_user)):
    conn = get_db()
    existing = conn.execute("SELECT id FROM profiles WHERE user_id = ?", (user["id"],)).fetchone()
    now = datetime.now().isoformat()
    if existing:
        conn.execute("""UPDATE profiles SET first_name=?, last_name=?, birth_date=?, blood_type=?,
            height_cm=?, weight_kg=?, allergies=?, chronic_conditions=?, doctor_name=?, doctor_phone=?, notes=?, updated_at=?
            WHERE user_id=?""",
            (data.first_name, data.last_name, data.birth_date, data.blood_type,
             data.height_cm, data.weight_kg, data.allergies, data.chronic_conditions,
             data.doctor_name, data.doctor_phone, data.notes, now, user["id"]))
    else:
        conn.execute("""INSERT INTO profiles (user_id, first_name, last_name, birth_date, blood_type,
            height_cm, weight_kg, allergies, chronic_conditions, doctor_name, doctor_phone, notes, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (user["id"], data.first_name, data.last_name, data.birth_date, data.blood_type,
             data.height_cm, data.weight_kg, data.allergies, data.chronic_conditions,
             data.doctor_name, data.doctor_phone, data.notes, now))
    conn.commit()
    conn.close()
    return {"ok": True}

# ==================== LAB RESULTS ====================

@app.get("/api/lab-results")
def get_lab_results(user=Depends(get_current_user)):
    conn = get_db()
    results = conn.execute(
        "SELECT * FROM lab_results WHERE user_id = ? ORDER BY date DESC", (user["id"],)
    ).fetchall()
    out = []
    for r in results:
        params = conn.execute("SELECT * FROM lab_parameters WHERE result_id = ? ORDER BY name", (r["id"],)).fetchall()
        file_info = None
        if r["file_id"]:
            f = conn.execute("SELECT id, original_name, stored_name, mime_type FROM files WHERE id = ?", (r["file_id"],)).fetchone()
            if f: file_info = dict(f)
        out.append({**dict(r), "parameters": [dict(p) for p in params], "file": file_info})
    conn.close()
    return out

@app.post("/api/lab-results", status_code=201)
def create_lab_result(data: LabResultIn, user=Depends(get_current_user)):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO lab_results (user_id, date, lab_name, notes, created_at) VALUES (?, ?, ?, ?, ?)",
        (user["id"], data.date, data.lab_name, data.notes, datetime.now().isoformat())
    )
    result_id = cur.lastrowid
    for p in data.parameters:
        conn.execute(
            "INSERT INTO lab_parameters (result_id, name, value, value_text, unit, ref_min, ref_max, ref_text, is_abnormal) VALUES (?,?,?,?,?,?,?,?,?)",
            (result_id, p.name, p.value, p.value_text, p.unit, p.ref_min, p.ref_max, p.ref_text, int(p.is_abnormal))
        )
    conn.commit()
    conn.close()
    return {"id": result_id}

@app.put("/api/lab-results/{result_id}")
def update_lab_result(result_id: int, data: LabResultIn, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT id FROM lab_results WHERE id = ? AND user_id = ?", (result_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE lab_results SET date=?, lab_name=?, notes=? WHERE id=?",
        (data.date, data.lab_name, data.notes, result_id))
    conn.execute("DELETE FROM lab_parameters WHERE result_id = ?", (result_id,))
    for p in data.parameters:
        conn.execute(
            "INSERT INTO lab_parameters (result_id, name, value, value_text, unit, ref_min, ref_max, ref_text, is_abnormal) VALUES (?,?,?,?,?,?,?,?,?)",
            (result_id, p.name, p.value, p.value_text, p.unit, p.ref_min, p.ref_max, p.ref_text, int(p.is_abnormal))
        )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/api/lab-results/{result_id}")
def delete_lab_result(result_id: int, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT id FROM lab_results WHERE id = ? AND user_id = ?", (result_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("DELETE FROM lab_results WHERE id = ?", (result_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/api/lab-results/trends")
def get_trends(user=Depends(get_current_user)):
    """Returns all parameter names with their values over time for charting."""
    conn = get_db()
    rows = conn.execute("""
        SELECT lp.name, lp.value, lp.unit, lp.ref_min, lp.ref_max, lr.date
        FROM lab_parameters lp
        JOIN lab_results lr ON lp.result_id = lr.id
        WHERE lr.user_id = ? AND lp.value IS NOT NULL
        ORDER BY lp.name, lr.date ASC
    """, (user["id"],)).fetchall()
    conn.close()
    trends = {}
    for row in rows:
        name = row["name"]
        if name not in trends:
            trends[name] = {"name": name, "unit": row["unit"], "ref_min": row["ref_min"], "ref_max": row["ref_max"], "points": []}
        trends[name]["points"].append({"date": row["date"], "value": row["value"]})
    return list(trends.values())

# ==================== ATTACH FILE TO LAB RESULT ====================

@app.put("/api/lab-results/{result_id}/file/{file_id}")
def attach_file_to_lab(result_id: int, file_id: int, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT id FROM lab_results WHERE id = ? AND user_id = ?", (result_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE lab_results SET file_id = ? WHERE id = ?", (file_id, result_id))
    conn.commit()
    conn.close()
    return {"ok": True}

# ==================== MEDICAL HISTORY ====================

@app.get("/api/history")
def get_history(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM medical_history WHERE user_id = ? ORDER BY date DESC", (user["id"],)
    ).fetchall()
    out = []
    for r in rows:
        files = conn.execute("""
            SELECT f.id, f.original_name, f.stored_name, f.mime_type, f.size_bytes
            FROM files f JOIN history_files hf ON f.id = hf.file_id
            WHERE hf.history_id = ?
        """, (r["id"],)).fetchall()
        out.append({**dict(r), "files": [dict(f) for f in files]})
    conn.close()
    return out

@app.post("/api/history", status_code=201)
def create_history(data: HistoryIn, user=Depends(get_current_user)):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO medical_history (user_id, date, title, description, category, created_at) VALUES (?,?,?,?,?,?)",
        (user["id"], data.date, data.title, data.description, data.category, datetime.now().isoformat())
    )
    history_id = cur.lastrowid
    for fid in data.file_ids:
        f = conn.execute("SELECT id FROM files WHERE id = ? AND user_id = ?", (fid, user["id"])).fetchone()
        if f:
            conn.execute("INSERT INTO history_files (history_id, file_id) VALUES (?, ?)", (history_id, fid))
    conn.commit()
    conn.close()
    return {"id": history_id}

@app.put("/api/history/{history_id}")
def update_history(history_id: int, data: HistoryIn, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT id FROM medical_history WHERE id = ? AND user_id = ?", (history_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("UPDATE medical_history SET date=?, title=?, description=?, category=? WHERE id=?",
        (data.date, data.title, data.description, data.category, history_id))
    conn.execute("DELETE FROM history_files WHERE history_id = ?", (history_id,))
    for fid in data.file_ids:
        f = conn.execute("SELECT id FROM files WHERE id = ? AND user_id = ?", (fid, user["id"])).fetchone()
        if f:
            conn.execute("INSERT INTO history_files (history_id, file_id) VALUES (?, ?)", (history_id, fid))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/api/history/{history_id}")
def delete_history(history_id: int, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT id FROM medical_history WHERE id = ? AND user_id = ?", (history_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.execute("DELETE FROM medical_history WHERE id = ?", (history_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

# ==================== FILES ====================

ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/webp"}

@app.post("/api/files", status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("inne"),
    description: str = Form(""),
    date: str = Form(""),
    tags: str = Form(""),
    user=Depends(get_current_user)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Dozwolone formaty: PDF, JPG, PNG")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Plik za duży (max 50MB)")

    ext = Path(file.filename).suffix.lower()
    stored_name = f"{uuid.uuid4()}{ext}"
    dest = user_upload_dir(user["id"]) / stored_name
    dest.write_bytes(content)

    conn = get_db()
    cur = conn.execute(
        "INSERT INTO files (user_id, original_name, stored_name, mime_type, size_bytes, category, description, date, tags, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (user["id"], file.filename, stored_name, file.content_type, len(content),
         category, description, date or None, tags, datetime.now().isoformat())
    )
    file_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": file_id, "original_name": file.filename, "stored_name": stored_name, "mime_type": file.content_type}

@app.get("/api/files")
def list_files(user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM files WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.delete("/api/files/{file_id}")
def delete_file(file_id: int, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ? AND user_id = ?", (file_id, user["id"])).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    path = user_upload_dir(user["id"]) / row["stored_name"]
    if path.exists():
        path.unlink()
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/api/files/{file_id}/view")
def view_file(file_id: int, user=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM files WHERE id = ? AND user_id = ?", (file_id, user["id"])).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    path = user_upload_dir(user["id"]) / row["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path, media_type=row["mime_type"], filename=row["original_name"])

# ==================== OCR ====================

@app.post("/api/ocr")
async def ocr_pdf(file: UploadFile = File(...), user=Depends(get_current_user)):
    if file.content_type not in ("application/pdf", "image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Tylko PDF, JPG, PNG")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Plik za duży")

    try:
        full_text = extract_text_from_pdf(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd OCR: {str(e)}")

    print(f"=== OCR TEXT LENGTH: {len(full_text)} ===")
    print(f"=== OCR TEXT PREVIEW: {full_text[:500]} ===")
    parameters = await parse_with_ollama(full_text)
    return {"raw_text": full_text[:3000], "parameters": parameters}

def extract_text_from_pdf(content: bytes) -> str:
    # Próba 1: wyciągnij tekst wektorowy
    doc = pymupdf.open(stream=content, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Jeśli tekst jest śmieciami lub pusty – użyj Tesseract przez konwersję stron na obrazy
    if len(full_text.strip()) < 50 or looks_like_garbage(full_text):
        full_text = ocr_with_tesseract(content)
    
    return full_text

def looks_like_garbage(text: str) -> bool:
    # Heurystyka: za dużo znaków specjalnych = śmieci
    if not text.strip():
        return True
    printable = sum(1 for c in text if c.isprintable() and c.isascii())
    ratio = printable / max(len(text), 1)
    return ratio < 0.6

def ocr_with_tesseract(content: bytes) -> str:
    """Konwertuje PDF na obrazy i puszcza przez Tesseract."""
    doc = pymupdf.open(stream=content, filetype="pdf")
    full_text = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, page in enumerate(doc):
            # Renderuj stronę jako obraz 300 DPI
            mat = pymupdf.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_path = f"{tmpdir}/page_{i}.png"
            pix.save(img_path)
            # Tesseract
            result = subprocess.run(
                ["tesseract", img_path, "stdout", "-l", "pol+eng", "--psm", "6"],
                capture_output=True, text=True, timeout=30
            )
            full_text += result.stdout + "\n"
    doc.close()
    return full_text

async def parse_with_ollama(text: str) -> list:
    prompt = f"""Jesteś asystentem medycznym. Z poniższego tekstu wyników badań laboratoryjnych wyciągnij wszystkie parametry.
Odpowiedz TYLKO jako JSON array, bez żadnego tekstu przed ani po. Format:
[{{"name": "Nazwa parametru", "value": 5.4, "unit": "jednostka", "ref_min": 4.0, "ref_max": 6.0}}]
Jeśli brak zakresu referencyjnego, użyj null. Wartości liczbowe jako liczby, nie stringi.

Tekst wyników:
{text}

JSON:"""

    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        })
        r.raise_for_status()
        raw = r.json()["response"].strip()
        # Wyciągnij JSON z odpowiedzi
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start == -1 or end == 0:
            return []
        data = json.loads(raw[start:end])
        # Dodaj is_abnormal
        for p in data:
            v = p.get("value")
            rmin = p.get("ref_min")
            rmax = p.get("ref_max")
            p["is_abnormal"] = bool(v and rmin and rmax and (v < rmin or v > rmax))
            p.setdefault("value_text", None)
            p.setdefault("ref_text", "")
        return data

# ==================== STATIC ====================

app.mount("/", StaticFiles(directory="/frontend", html=True), name="frontend")
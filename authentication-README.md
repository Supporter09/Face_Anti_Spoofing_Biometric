
## 1. Clone the Repository

## 2. PostgreSQL Setup

### Install PostgreSQL (WSL / Ubuntu)

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo service postgresql start
```

### Create database and user

```bash
sudo -u postgres psql
```

Inside the psql shell:

```sql
CREATE USER face_auth WITH PASSWORD 'face_auth_pass';
CREATE DATABASE face_anti_spoofing OWNER face_auth;
GRANT ALL PRIVILEGES ON DATABASE face_anti_spoofing TO face_auth;
\q
```

> The `face_templates` table is created **automatically** when the backend starts for the first time.

---

## 3. Backend Setup

### Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install psycopg2-binary python-dotenv
```

### Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://face_auth:face_auth_pass@localhost:5432/face_anti_spoofing
FACE_AUTH_THRESHOLD=0.50
```

> `FACE_AUTH_THRESHOLD` controls how strict face matching is (0.0–1.0). Default is `0.50`.

### Start the backend

```bash
source .venv/bin/activate
uvicorn services.api.app:app --reload
```

Backend runs at: `http://127.0.0.1:8000`

---

## 4. Frontend Setup

### Install dependencies

```bash
cd apps/web
npm install
```

### Start the frontend

```bash
npm run dev
```

Frontend runs at: `http://127.0.0.1:5173`

---

## 5. Register Your First User

1. Open `http://127.0.0.1:5173` in your browser
2. Click **📋 Đăng ký người dùng**
3. Enter a username (e.g. `student001`)
4. Click **📷 Bắt đầu đăng ký** — look at the camera and hold still during the 3-second countdown
5. Your face embedding is stored in PostgreSQL

---

## 6. Run Liveness + Authentication

1. On the main page, click **Bắt đầu kiểm tra**
2. Follow the on-screen instructions:
   - Look straight at the camera
   - Turn your head left or right as instructed
   - Return to center
   - Turn the other way
3. If liveness passes → your identity is checked against all enrolled users
4. Result screen shows: **THẬT** (real) with your username, or **GIẢ MẠO** (spoof detected)


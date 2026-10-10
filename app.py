import csv
import getpass
import io
import os
import secrets
import sqlite3
import sys
import calendar
import base64
import shutil
import random
import time

import cv2
import numpy as np
from datetime import datetime, date
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

from flask import (Flask, render_template, request, redirect, url_for, flash,
                   make_response, g, session, abort)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "attendance.db"
DAILY_STANDARD_HOURS = 8.0
FACE_DATA_DIR = BASE_DIR / "face_data"
FACE_DATA_DIR.mkdir(exist_ok=True)
FACE_MATCH_THRESHOLD = float(os.environ.get("HUB pulse_FACE_MATCH_THRESHOLD", "75.0"))  # Lower is stricter; calibrate with real enrolled samples.
FACE_AMBIGUITY_MARGIN = float(os.environ.get("HUB pulse_FACE_AMBIGUITY_MARGIN", "8.0"))  # Require a clear gap from the next-best employee.
FACE_MIN_SAMPLES = 3
FACE_CHALLENGE_MIN_SHIFT = 0.07  # Minimum horizontal face-center movement as a fraction of frame width.
FACE_ATTEMPT_COOLDOWN_SECONDS = 20

app = Flask(__name__)
app.secret_key = os.environ.get("HUB pulse_SECRET_KEY", "dev-only-change-this-secret-before-deployment")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("HUB pulse_HTTPS", "0") == "1",
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
)


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, timeout=10)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            department TEXT NOT NULL DEFAULT 'General',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            work_date TEXT NOT NULL,
            clock_in TEXT NOT NULL,
            clock_out TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'employee')),
            employee_id INTEGER UNIQUE,
            created_at TEXT NOT NULL,
            FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS login_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            attempted_at INTEGER NOT NULL,
            successful INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS attendance_breaks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance_id INTEGER NOT NULL,
            break_start TEXT NOT NULL,
            break_end TEXT,
            FOREIGN KEY(attendance_id) REFERENCES attendance(id) ON DELETE CASCADE
        );
        
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            leave_type TEXT NOT NULL
                CHECK(leave_type IN ('Casual', 'Sick', 'Personal', 'Other')),
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending'
                CHECK(status IN ('Pending', 'Approved', 'Rejected', 'Cancelled')),
            admin_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_by_user_id INTEGER,
            reviewed_at TEXT,
            FOREIGN KEY(employee_id)
                REFERENCES employees(id) ON DELETE CASCADE,
            FOREIGN KEY(reviewed_by_user_id)
                REFERENCES users(id) ON DELETE SET NULL,
            CHECK(start_date <= end_date)
        );
        
        CREATE TABLE IF NOT EXISTS payroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            period_month TEXT NOT NULL,
            total_regular_minutes INTEGER NOT NULL DEFAULT 0,
            total_overtime_minutes INTEGER NOT NULL DEFAULT 0,
            hourly_rate REAL NOT NULL,
            gross_pay REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Draft' CHECK(status IN ('Draft', 'Paid')),
            created_at TEXT NOT NULL,
            FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE,
            UNIQUE(employee_id, period_month)
        );
        
        CREATE INDEX IF NOT EXISTS idx_leave_employee_dates
            ON leave_requests(employee_id, start_date, end_date);
        CREATE INDEX IF NOT EXISTS idx_leave_status
            ON leave_requests(status);
        CREATE INDEX IF NOT EXISTS idx_attendance_day ON attendance(work_date);
        CREATE INDEX IF NOT EXISTS idx_attendance_employee_day ON attendance(employee_id, work_date);
        CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempted_at);
        CREATE INDEX IF NOT EXISTS idx_payroll_period ON payroll(period_month);
    """)
    # Additive migrations: existing employee and attendance records are preserved.
    employee_columns = {row[1] for row in db.execute("PRAGMA table_info(employees)").fetchall()}
    if "shift_start" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN shift_start TEXT NOT NULL DEFAULT '09:00'")
    if "shift_end" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN shift_end TEXT NOT NULL DEFAULT '17:00'")
    if "grace_minutes" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN grace_minutes INTEGER NOT NULL DEFAULT 10")
    if "hourly_rate" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN hourly_rate REAL NOT NULL DEFAULT 0.0")
        
    attendance_columns = {row[1] for row in db.execute("PRAGMA table_info(attendance)").fetchall()}

    if "casual_leave" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN casual_leave INTEGER NOT NULL DEFAULT 15")
        db.execute("ALTER TABLE employees ADD COLUMN sick_leave INTEGER NOT NULL DEFAULT 10")
        db.execute("ALTER TABLE employees ADD COLUMN personal_leave INTEGER NOT NULL DEFAULT 5")
        
    if "overtime_multiplier" not in employee_columns:
        db.execute("ALTER TABLE employees ADD COLUMN overtime_multiplier REAL NOT NULL DEFAULT 1.5")
        
    attendance_columns = {row[1] for row in db.execute("PRAGMA table_info(attendance)").fetchall()}
    if "late_minutes" not in attendance_columns:
        db.execute("ALTER TABLE attendance ADD COLUMN late_minutes INTEGER NOT NULL DEFAULT 0")
    
    # Add the permanent-admin column to existing databases.
    user_columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(users)").fetchall()
    }

    if "is_permanent_admin" not in user_columns:
        db.execute("""
            ALTER TABLE users
            ADD COLUMN is_permanent_admin INTEGER NOT NULL DEFAULT 0
        """)

    # Protect an existing administrator.
    existing_permanent = db.execute("""
        SELECT id FROM users
        WHERE is_permanent_admin = 1
        LIMIT 1
    """).fetchone()

    if not existing_permanent:
        existing_admin = db.execute("""
            SELECT id FROM users
            WHERE role = 'admin'
            ORDER BY id
            LIMIT 1
        """).fetchone()

        if existing_admin:
            db.execute("""
                UPDATE users
                SET is_permanent_admin = 1
                WHERE id = ?
            """, (existing_admin["id"],))

    db.execute("CREATE INDEX IF NOT EXISTS idx_breaks_attendance ON attendance_breaks(attendance_id)")
    db.commit()


@app.before_request
def before_request():
    init_db()
    # Simple session-based CSRF protection for every state-changing form, including login/logout.
    if request.method == "POST":
        submitted = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not submitted or not expected or not secrets.compare_digest(submitted, expected):
            abort(400, description="Security token missing or expired. Refresh the page and try again.")
    session.setdefault("csrf_token", secrets.token_urlsafe(32))


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db().execute("""
        SELECT u.id, u.username, u.role, u.employee_id,
               e.name AS employee_name, e.email AS employee_email, e.department AS employee_department
        FROM users u LEFT JOIN employees e ON e.id = u.employee_id
        WHERE u.id = ?
    """, (user_id,)).fetchone()


@app.context_processor
def inject_globals():
    return {"current_user": get_current_user(), "csrf_token": session.get("csrf_token", "")}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        user = get_current_user()
        if user is None:
            session.clear()
            flash("Your account is unavailable. Please sign in again.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if not user or user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def current_day():
    return date.today().isoformat()


def duration_minutes(clock_in, clock_out):
    if not clock_in:
        return 0
    try:
        start = datetime.fromisoformat(clock_in)
        end = datetime.fromisoformat(clock_out) if clock_out else datetime.now()
        return max(0, int((end - start).total_seconds() // 60))
    except (ValueError, TypeError):
        return 0


def duration_text(clock_in, clock_out):
    if not clock_in:
        return "—"
    minutes = duration_minutes(clock_in, clock_out)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def hours_text(minutes):
    return f"{minutes // 60}h {minutes % 60:02d}m"


def break_minutes_for_attendance(attendance_id):
    """Return total break minutes, including a currently open break."""
    now = datetime.now()
    rows = get_db().execute(
        "SELECT break_start, break_end FROM attendance_breaks WHERE attendance_id=?",
        (attendance_id,),
    ).fetchall()
    total = 0
    for item in rows:
        try:
            start = datetime.fromisoformat(item["break_start"])
            end = datetime.fromisoformat(item["break_end"]) if item["break_end"] else now
            total += max(0, int((end - start).total_seconds() // 60))
        except (ValueError, TypeError):
            continue
    return total


def worked_minutes(row):
    gross = duration_minutes(row["clock_in"], row["clock_out"])
    try:
        breaks = break_minutes_for_attendance(row["id"])
    except (KeyError, IndexError):
        breaks = 0
    return max(0, gross - breaks)


def worked_duration_text(row):
    minutes = worked_minutes(row)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def late_minutes_for_employee(employee, clock_in):
    """Minutes after scheduled start; grace period is applied for late status."""
    try:
        actual = datetime.fromisoformat(clock_in)
        scheduled = datetime.combine(actual.date(), datetime.strptime(employee["shift_start"], "%H:%M").time())
        return max(0, int((actual - scheduled).total_seconds() // 60))
    except (ValueError, TypeError, KeyError, IndexError):
        return 0


def active_attendance(employee_id, work_date=None):
    work_date = work_date or current_day()
    return get_db().execute(
        "SELECT * FROM attendance WHERE employee_id=? AND work_date=? AND clock_out IS NULL ORDER BY clock_in DESC LIMIT 1",
        (employee_id, work_date),
    ).fetchone()


@app.template_filter("duration")
def duration_filter(row):
    return worked_duration_text(row)


def month_bounds(month_value):
    try:
        parsed = datetime.strptime(month_value, "%Y-%m").date()
    except (ValueError, TypeError):
        parsed = date.today().replace(day=1)
    first = parsed.replace(day=1)
    last = parsed.replace(day=calendar.monthrange(first.year, first.month)[1])
    return first, last


def report_filters(db):
    start = request.args.get("start", date.today().replace(day=1).isoformat())
    end = request.args.get("end", current_day())
    employee_id = request.args.get("employee_id", "all")
    try:
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        start, end = date.today().replace(day=1).isoformat(), current_day()
    if end < start:
        start, end = end, start
    sql = """SELECT a.*, e.name, e.email, e.department
             FROM attendance a JOIN employees e ON e.id = a.employee_id
             WHERE a.work_date BETWEEN ? AND ?"""
    params = [start, end]
    if employee_id != "all":
        try:
            employee_id = str(int(employee_id))
            sql += " AND e.id = ?"
            params.append(int(employee_id))
        except (ValueError, TypeError):
            employee_id = "all"
    sql += " ORDER BY a.work_date DESC, e.name COLLATE NOCASE, a.clock_in DESC"
    return start, end, employee_id, db.execute(sql, params).fetchall()


def summarize_daily(records):
    grouped = {}
    for row in records:
        key = (row["work_date"], row["employee_id"])
        if key not in grouped:
            grouped[key] = {"work_date": row["work_date"], "employee_id": row["employee_id"],
                            "name": row["name"], "email": row["email"], "department": row["department"],
                            "minutes": 0, "shifts": 0}
        grouped[key]["minutes"] += worked_minutes(row)
        grouped[key]["shifts"] += 1
    threshold = int(DAILY_STANDARD_HOURS * 60)
    result = []
    for item in grouped.values():
        item["hours"] = hours_text(item["minutes"])
        item["overtime_minutes"] = max(0, item["minutes"] - threshold)
        item["overtime"] = hours_text(item["overtime_minutes"])
        result.append(item)
    result.sort(key=lambda item: (item["work_date"], item["name"].lower()), reverse=True)
    return result


def decode_face_frame(data_url):
    """Decode one webcam JPEG, require one face, and return normalized crop + horizontal center."""
    if not data_url or not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Capture a frame from the webcam first.")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/") or len(encoded) > 4_000_000:
        raise ValueError("Invalid image. Please capture a new webcam frame.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("The webcam image could not be decoded.") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The webcam image is invalid. Please try again.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50)
        )
    if len(faces) == 0:
        raise ValueError(
            "No face detected. Move closer to the camera, face forward, "
            "and keep your full face visible."
        )
    if len(faces) != 1:
        raise ValueError("Please ensure only one face is visible in the camera.")
    x, y, w, h = faces[0]
    crop = gray[y:y+h, x:x+w]
    crop = cv2.equalizeHist(crop)
    normalized = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)
    center_x = (x + w / 2) / max(1, image.shape[1])
    return normalized, center_x


def decode_and_crop_face(data_url):
    """Decode a browser-captured JPEG and return one normalized grayscale face crop."""
    crop, _center_x = decode_face_frame(data_url)
    return crop


def verify_head_turn_challenge(baseline_data, challenge_data, direction):
    """Basic challenge-response check; useful as friction, not reliable biometric liveness proof."""
    if direction not in ("left", "right"):
        raise ValueError("The live-check challenge expired. Refresh the page and try again.")
    _baseline_crop, baseline_x = decode_face_frame(baseline_data)
    final_crop, final_x = decode_face_frame(challenge_data)
    shift = final_x - baseline_x
    required = FACE_CHALLENGE_MIN_SHIFT
    # The client mirrors the captured frame to match the displayed webcam preview.
    passed = shift <= -required if direction == "left" else shift >= required
    if not passed:
        raise ValueError(f"Live check not confirmed. Move your face toward the {direction.upper()} side of the video frame and try again.")
    return final_crop


def face_sample_files(employee_id):
    folder = FACE_DATA_DIR / str(int(employee_id))
    if not folder.exists():
        return []
    return sorted(folder.glob("*.png"))


def face_sample_counts():
    db = get_db()
    employees = db.execute("SELECT id FROM employees").fetchall()
    return {row["id"]: len(face_sample_files(row["id"])) for row in employees}


def train_face_model():
    """Train an in-memory LBPH recognizer using enrolled local face crops."""
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("OpenCV face recognition is unavailable. Install opencv-contrib-python as shown in the setup guide.")
    images, labels = [], []
    for folder in FACE_DATA_DIR.iterdir():
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        employee_id = int(folder.name)
        files = sorted(folder.glob("*.png"))
        if len(files) < FACE_MIN_SAMPLES:
            continue
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is not None and image.shape == (200, 200):
                images.append(image)
                labels.append(employee_id)
    if len(images) < 2 or len(set(labels)) < 1:
        raise ValueError(f"No trained face profiles are available yet. Enroll at least {FACE_MIN_SAMPLES} samples for an employee.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer


def recognize_employee(face_crop):
    """Match an enrolled face only when the best employee is clear and confident.

    LBPH returns a distance-like score: lower is a closer match. We compare the
    query's LBPH histogram against every enrolled sample, summarize the closest
    few samples per employee, and reject near-ties between different employees.
    """
    recognizer = train_face_model()

    # Build the same LBPH descriptor for the query crop using the recognizer's
    # default parameters. The resulting histogram can be compared with the
    # histograms held by the trained model.
    query_model = cv2.face.LBPHFaceRecognizer_create()
    query_model.train([face_crop], np.asarray([0], dtype=np.int32))
    query_histograms = query_model.getHistograms()
    if not query_histograms:
        raise RuntimeError("Could not calculate the face descriptor. Try capturing the face again.")
    query_hist = query_histograms[0]

    enrolled_histograms = recognizer.getHistograms()
    enrolled_labels = np.asarray(recognizer.getLabels()).reshape(-1)
    if len(enrolled_histograms) != len(enrolled_labels) or not len(enrolled_histograms):
        raise RuntimeError("The enrolled face model is incomplete. Re-enroll the affected employees.")

    # Collect the closest distances for each employee. Use up to three closest
    # samples so a single unusually similar sample does not dominate the score.
    distances_by_employee = {}
    for hist, raw_label in zip(enrolled_histograms, enrolled_labels):
        employee_id = int(raw_label)
        distance = float(cv2.compareHist(
            np.asarray(query_hist, dtype=np.float32),
            np.asarray(hist, dtype=np.float32),
            cv2.HISTCMP_CHISQR_ALT,
        ))
        if np.isfinite(distance):
            distances_by_employee.setdefault(employee_id, []).append(distance)

    employee_scores = []
    for employee_id, distances in distances_by_employee.items():
        closest = sorted(distances)[:3]
        score = float(np.median(closest))
        employee_scores.append((score, employee_id))

    if not employee_scores:
        raise ValueError("No usable enrolled face samples were found. Re-enroll the employee.")

    employee_scores.sort(key=lambda item: item[0])
    best_score, best_employee_id = employee_scores[0]

    if best_score > FACE_MATCH_THRESHOLD:
        raise ValueError(
            "Face match is not confident enough. Please try again or use regular attendance."
        )

    if len(employee_scores) > 1:
        second_score, _second_employee_id = employee_scores[1]
        score_gap = second_score - best_score
        if score_gap < FACE_AMBIGUITY_MARGIN:
            raise ValueError(
                "Face match is ambiguous between enrolled employees. "
                "No attendance was recorded; please retry or use regular attendance."
            )

    db = get_db()
    employee = db.execute(
        "SELECT * FROM employees WHERE id = ?", (int(best_employee_id),)
    ).fetchone()
    if employee is None or len(face_sample_files(int(best_employee_id))) < FACE_MIN_SAMPLES:
        raise ValueError("No enrolled employee matches this face.")
    return employee, best_score


def record_face_attendance(employee_id):
    """Toggle today's attendance for a recognized employee, enforcing break and late rules."""
    db = get_db()
    today = current_day()
    employee = db.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone()
    if employee is None:
        raise ValueError("Employee profile no longer exists.")
    opened = active_attendance(employee_id, today)
    now = datetime.now().isoformat(timespec="seconds")
    if opened:
        # Close any active break before clocking out.
        db.execute("UPDATE attendance_breaks SET break_end=? WHERE attendance_id=? AND break_end IS NULL",
                   (now, opened["id"]))
        db.execute("UPDATE attendance SET clock_out=? WHERE id=?", (now, opened["id"]))
        db.commit()
        return f"Clocked out {employee['name']} successfully."
    late = late_minutes_for_employee(employee, now)
    db.execute("INSERT INTO attendance (employee_id, work_date, clock_in, late_minutes) VALUES (?, ?, ?, ?)",
               (employee_id, today, now, late))
    db.commit()
    if late > int(employee["grace_minutes"] or 0):
        return f"Clocked in {employee['name']}. Late arrival recorded ({late} minutes after scheduled start)."
    return f"Clocked in {employee['name']} successfully."


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        user = get_current_user()
        if user:
            return redirect(url_for("dashboard" if user["role"] == "admin" else "attendance"))
        session.clear()
        session["csrf_token"] = secrets.token_urlsafe(32)

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        now = int(datetime.now().timestamp())
        db = get_db()
        # Five failed attempts per username in a 15-minute window trigger a temporary block.
        recent_failures = db.execute("""
            SELECT COUNT(*) FROM login_attempts
            WHERE username = ? COLLATE NOCASE AND successful = 0 AND attempted_at > ?
        """, (username, now - 900)).fetchone()[0]
        if recent_failures >= 5:
            flash("Too many failed sign-in attempts. Wait 15 minutes and try again.", "error")
            return render_template("login.html"), 429
        user = db.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        valid = bool(user and check_password_hash(user["password_hash"], password))
        db.execute("INSERT INTO login_attempts (username, attempted_at, successful) VALUES (?, ?, ?)",
                   (username, now, 1 if valid else 0))
        db.execute("DELETE FROM login_attempts WHERE attempted_at < ?", (now - 86400,))
        db.commit()
        if not valid:
            flash("Invalid username or password.", "error")
            return render_template("login.html"), 401
        session.clear()  # prevent session fixation; issue fresh CSRF token after authentication
        session["user_id"] = user["id"]
        session["csrf_token"] = secrets.token_urlsafe(32)
        session.permanent = True
        target = request.args.get("next", "")
        # Only allow local relative paths for redirects after login.
        parsed = urlparse(target)
        if target.startswith("/") and not target.startswith("//") and not parsed.netloc:
            return redirect(target)
        return redirect(url_for("dashboard" if user["role"] == "admin" else "attendance"))
    return render_template("login.html")


@app.post("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))

@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        user = get_current_user()
        db = get_db()
        
        # Verify the current password
        user_record = db.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
        
        if not check_password_hash(user_record["password_hash"], current_password):
            flash("Incorrect current password.", "error")
        elif len(new_password) < 10:
            flash("New password must be at least 10 characters long.", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "error")
        else:
            # Hash and save the new password
            db.execute("UPDATE users SET password_hash = ? WHERE id = ?", 
                       (generate_password_hash(new_password), user["id"]))
            db.commit()
            
            # Keep the user logged in and redirect
            flash("Your password has been changed successfully.", "success")
            return redirect(url_for("dashboard" if user["role"] == "admin" else "attendance"))
            
    return render_template("change_password.html")

@app.errorhandler(403)
def forbidden(error):
    return render_template("error.html", code=403, message="You do not have permission to access this page."), 403


@app.route("/")
@login_required
def dashboard():
    user = get_current_user()
    if user["role"] != "admin":
        return redirect(url_for("attendance"))
    db = get_db()
    today = current_day()
    total_employees = db.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    present_today = db.execute("SELECT COUNT(DISTINCT employee_id) FROM attendance WHERE work_date = ?", (today,)).fetchone()[0]
    open_shifts = db.execute("SELECT COUNT(*) FROM attendance WHERE work_date = ? AND clock_out IS NULL", (today,)).fetchone()[0]
    today_records = db.execute("""SELECT a.*, e.name, e.email, e.department FROM attendance a
                          JOIN employees e ON e.id=a.employee_id WHERE a.work_date=?
                          ORDER BY a.clock_in DESC""", (today,)).fetchall()
    today_daily_summary = summarize_daily(today_records)
    today_total_minutes = sum(item["minutes"] for item in today_daily_summary)
    today_total_overtime_minutes = sum(item["overtime_minutes"] for item in today_daily_summary)
    recent = today_records[:8]
    month_value = request.args.get("month", date.today().strftime("%Y-%m"))
    month_start, month_end = month_bounds(month_value)
    month_value = month_start.strftime("%Y-%m")
    month_records = db.execute("""SELECT a.*, e.name, e.email, e.department FROM attendance a
        JOIN employees e ON e.id=a.employee_id WHERE a.work_date BETWEEN ? AND ? ORDER BY a.work_date, e.name COLLATE NOCASE""",
        (month_start.isoformat(), month_end.isoformat())).fetchall()
    daily_summary = summarize_daily(month_records)
    days_in_month = calendar.monthrange(month_start.year, month_start.month)[1]
    daily_map = {}
    for item in daily_summary:
        day_num = int(item["work_date"][8:10])
        daily_map.setdefault(day_num, {"people": set(), "minutes": 0, "overtime": 0})
        daily_map[day_num]["people"].add(item["employee_id"])
        daily_map[day_num]["minutes"] += item["minutes"]
        daily_map[day_num]["overtime"] += item["overtime_minutes"]
    chart_labels = list(range(1, days_in_month + 1))
    chart_attendance = [len(daily_map.get(n, {}).get("people", set())) for n in chart_labels]
    chart_hours = [round(daily_map.get(n, {}).get("minutes", 0) / 60, 2) for n in chart_labels]
    chart_overtime = [round(daily_map.get(n, {}).get("overtime", 0) / 60, 2) for n in chart_labels]
    employee_totals = {}
    for item in daily_summary:
        eid = item["employee_id"]
        if eid not in employee_totals:
            employee_totals[eid] = {"name": item["name"], "department": item["department"], "days": 0, "minutes": 0, "overtime_minutes": 0}
        employee_totals[eid]["days"] += 1
        employee_totals[eid]["minutes"] += item["minutes"]
        employee_totals[eid]["overtime_minutes"] += item["overtime_minutes"]
    employee_summary = sorted(employee_totals.values(), key=lambda x: x["name"].lower())
    for item in employee_summary:
        item["hours"] = hours_text(item["minutes"])
        item["overtime"] = hours_text(item["overtime_minutes"])
    return render_template("dashboard.html", total_employees=total_employees, present_today=present_today,
        open_shifts=open_shifts, recent=recent, today=today, month_value=month_value,
        month_label=month_start.strftime("%B %Y"), month_end_value=month_end.isoformat(),
        chart_labels=chart_labels, chart_attendance=chart_attendance, chart_hours=chart_hours,
        chart_overtime=chart_overtime, employee_summary=employee_summary,
        month_total_hours=hours_text(sum(x["minutes"] for x in daily_summary)),
        month_total_overtime=hours_text(sum(x["overtime_minutes"] for x in daily_summary)),
        today_total_hours=hours_text(today_total_minutes),
        today_total_overtime=hours_text(today_total_overtime_minutes),
        daily_standard_hours=DAILY_STANDARD_HOURS)


@app.route("/employees", methods=["GET", "POST"])
@admin_required
def employees():
    db = get_db()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        department = request.form.get("department", "").strip() or "General"
        password = request.form.get("password", "")
        hourly_rate_raw = request.form.get("hourly_rate", "0.0").strip()
        
        try:
            hourly_rate = float(hourly_rate_raw)
        except ValueError:
            hourly_rate = 0.0

        if not name or not email or not password:
            flash("Name, email, and an initial password are required.", "error")
        elif len(password) < 10:
            flash("Employee passwords must be at least 10 characters long.", "error")
        else:
            try:
                cur = db.execute("INSERT INTO employees (name,email,department,hourly_rate,created_at) VALUES (?,?,?,?,?)",
                    (name, email, department, hourly_rate, datetime.now().isoformat(timespec="seconds")))
                employee_id = cur.lastrowid
                db.execute("INSERT INTO users (username,password_hash,role,employee_id,created_at) VALUES (?,?,?,?,?)",
                    (email, generate_password_hash(password), "employee", employee_id, datetime.now().isoformat(timespec="seconds")))
                db.commit()
                flash(f"Employee {name} and their login account were created. Username: {email}", "success")
                return redirect(url_for("employees"))
            except sqlite3.IntegrityError:
                db.rollback()
                flash("That email address or username is already registered.", "error")
    rows = db.execute("""SELECT e.*, u.username AS login_username FROM employees e
                         LEFT JOIN users u ON u.employee_id=e.id ORDER BY e.name COLLATE NOCASE""").fetchall()
    return render_template("employees.html", employees=rows)


@app.post("/employees/<int:employee_id>/create-login")
@admin_required
def create_employee_login(employee_id):
    db = get_db()
    employee = db.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone()
    if not employee:
        abort(404)
    password = request.form.get("password", "")
    if len(password) < 10:
        flash("Password must be at least 10 characters long.", "error")
    else:
        existing = db.execute("SELECT id FROM users WHERE employee_id=?", (employee_id,)).fetchone()
        if existing:
            flash("This employee already has a login account.", "error")
        else:
            try:
                db.execute("INSERT INTO users (username,password_hash,role,employee_id,created_at) VALUES (?,?,?,?,?)",
                    (employee["email"].lower(), generate_password_hash(password), "employee", employee_id, datetime.now().isoformat(timespec="seconds")))
                db.commit()
                flash(f"Login created for {employee['name']}. Username: {employee['email']}", "success")
            except sqlite3.IntegrityError:
                db.rollback()
                flash("That email is already used as a login username.", "error")
    return redirect(url_for("employees"))


@app.post("/employees/<int:employee_id>/delete")
@admin_required
def delete_employee(employee_id):
    db = get_db()
    db.execute("DELETE FROM employees WHERE id=?", (employee_id,))
    db.commit()
    face_folder = FACE_DATA_DIR / str(employee_id)
    if face_folder.exists():
        shutil.rmtree(face_folder, ignore_errors=True)
    flash("Employee, login account, attendance records, and local face samples deleted.", "success")
    return redirect(url_for("employees"))
@app.route("/employees/<int:employee_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_employee(employee_id):
    db = get_db()
    employee = db.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone()
    
    if not employee:
        abort(404)
        
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        department = request.form.get("department", "").strip() or "General"
        hourly_rate_raw = request.form.get("hourly_rate", "0.0").strip()
        overtime_multiplier_raw = request.form.get("overtime_multiplier", "1.5").strip() # NEW
        
        try:
            hourly_rate = float(hourly_rate_raw)
            overtime_multiplier = float(overtime_multiplier_raw) # NEW
        except ValueError:
            hourly_rate = 0.0
            overtime_multiplier = 1.5

        if not name or not email:
            flash("Name and email are required.", "error")
        else:
            try:
                # Update the employee profile (Added overtime_multiplier)
                db.execute("""
                    UPDATE employees 
                    SET name=?, email=?, department=?, hourly_rate=?, overtime_multiplier=? 
                    WHERE id=?
                """, (name, email, department, hourly_rate, overtime_multiplier, employee_id))
                
                # Automatically update their login username to match the new email
                db.execute("UPDATE users SET username=? WHERE employee_id=?", (email, employee_id))
                
                db.commit()
                flash(f"Updated profile for {name}.", "success")
                return redirect(url_for("employees"))
            except sqlite3.IntegrityError:
                db.rollback()
                flash("That email address is already in use by another employee.", "error")
                
    return render_template("edit_employee.html", employee=employee)
@app.route("/shifts", methods=["GET", "POST"])
@admin_required
def shifts():
    db = get_db()
    if request.method == "POST":
        employee_id = request.form.get("employee_id", type=int)
        start = request.form.get("shift_start", "").strip()
        end = request.form.get("shift_end", "").strip()
        grace_raw = request.form.get("grace_minutes", "10").strip()
        employee = db.execute("SELECT id, name FROM employees WHERE id=?", (employee_id,)).fetchone() if employee_id else None
        try:
            datetime.strptime(start, "%H:%M")
            datetime.strptime(end, "%H:%M")
            grace = int(grace_raw)
            if not 0 <= grace <= 120:
                raise ValueError
        except ValueError:
            flash("Use valid 24-hour times and a grace period between 0 and 120 minutes.", "error")
            return redirect(url_for("shifts"))
        if not employee:
            flash("Choose a valid employee.", "error")
        elif end <= start:
            flash("Shift end must be later than shift start. Overnight shifts are not supported in this version.", "error")
        else:
            db.execute("UPDATE employees SET shift_start=?, shift_end=?, grace_minutes=? WHERE id=?",
                       (start, end, grace, employee_id))
            db.commit()
            flash(f"Shift schedule updated for {employee['name']}.", "success")
        return redirect(url_for("shifts"))
    employees = db.execute("SELECT id, name, email, department, shift_start, shift_end, grace_minutes FROM employees ORDER BY name COLLATE NOCASE").fetchall()
    return render_template("shifts.html", employees=employees)


@app.route("/attendance", methods=["GET", "POST"])
@login_required
def attendance():
    db = get_db()
    user = get_current_user()
    today = current_day()
    is_admin = user["role"] == "admin"
    if request.method == "POST":
        if is_admin:
            employee_id = request.form.get("employee_id", type=int)
        else:
            employee_id = user["employee_id"]  # Never trust a client-supplied employee ID for employee accounts.
        action = request.form.get("action")
        employee = db.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone() if employee_id else None
        now = datetime.now().isoformat(timespec="seconds")
        if not employee:
            flash("No employee profile is linked to this account. Contact the administrator.", "error")
        elif action == "in":
            opened = active_attendance(employee_id, today)
            if opened:
                flash(f"{employee['name']} is already clocked in.", "error")
            else:
                late = late_minutes_for_employee(employee, now)
                db.execute("INSERT INTO attendance (employee_id,work_date,clock_in,late_minutes) VALUES (?,?,?,?)",
                    (employee_id,today,now,late))
                db.commit()
                if late > int(employee["grace_minutes"] or 0):
                    flash(f"Clocked in. Late arrival recorded: {late} minutes after scheduled start.", "error")
                else:
                    flash("Clocked in successfully.", "success")
        elif action == "out":
            opened = active_attendance(employee_id, today)
            if not opened:
                flash("There is no active shift to clock out from.", "error")
            else:
                db.execute("UPDATE attendance_breaks SET break_end=? WHERE attendance_id=? AND break_end IS NULL", (now, opened["id"]))
                db.execute("UPDATE attendance SET clock_out=? WHERE id=?", (now,opened["id"]))
                db.commit(); flash("Clocked out successfully. Any active break was closed automatically.", "success")
        elif action in ("break_start", "break_end"):
            opened = active_attendance(employee_id, today)
            if not opened:
                flash("Clock in before recording a break.", "error")
            elif action == "break_start":
                current_break = db.execute("SELECT id FROM attendance_breaks WHERE attendance_id=? AND break_end IS NULL", (opened["id"],)).fetchone()
                if current_break:
                    flash("A break is already active.", "error")
                else:
                    db.execute("INSERT INTO attendance_breaks (attendance_id, break_start) VALUES (?,?)", (opened["id"], now))
                    db.commit(); flash("Break started. Break time is excluded from worked hours.", "success")
            else:
                current_break = db.execute("SELECT id FROM attendance_breaks WHERE attendance_id=? AND break_end IS NULL ORDER BY break_start DESC LIMIT 1", (opened["id"],)).fetchone()
                if not current_break:
                    flash("There is no active break to end.", "error")
                else:
                    db.execute("UPDATE attendance_breaks SET break_end=? WHERE id=?", (now, current_break["id"]))
                    db.commit(); flash("Break ended.", "success")
        else:
            flash("Choose a valid attendance action.", "error")
        return redirect(url_for("attendance"))

    if is_admin:
        staff = db.execute("SELECT * FROM employees ORDER BY name COLLATE NOCASE").fetchall()
        records = db.execute("""SELECT a.*,e.name,e.department FROM attendance a JOIN employees e ON e.id=a.employee_id
            WHERE a.work_date=? ORDER BY a.clock_in DESC""", (today,)).fetchall()
        active_ids = {r["employee_id"] for r in db.execute("SELECT employee_id FROM attendance WHERE work_date=? AND clock_out IS NULL",(today,)).fetchall()}
    else:
        if not user["employee_id"]:
            abort(403)
        staff = [db.execute("SELECT * FROM employees WHERE id=?", (user["employee_id"],)).fetchone()]
        records = db.execute("""SELECT a.*,e.name,e.department FROM attendance a JOIN employees e ON e.id=a.employee_id
            WHERE a.work_date=? AND a.employee_id=? ORDER BY a.clock_in DESC""", (today,user["employee_id"])).fetchall()
        active_ids = {user["employee_id"]} if any(r["clock_out"] is None for r in records) else set()
    # Sum all shifts for each employee today; overtime starts after the configured daily standard.
    daily_totals = {}
    for row in records:
        employee_id = row["employee_id"]
        if employee_id not in daily_totals:
            daily_totals[employee_id] = {"minutes": 0, "overtime_minutes": 0}
        daily_totals[employee_id]["minutes"] += worked_minutes(row)
    standard_minutes = int(DAILY_STANDARD_HOURS * 60)
    for item in daily_totals.values():
        item["overtime_minutes"] = max(0, item["minutes"] - standard_minutes)
        item["hours"] = hours_text(item["minutes"])
        item["overtime"] = hours_text(item["overtime_minutes"])
    today_total_minutes = sum(item["minutes"] for item in daily_totals.values())
    today_total_overtime_minutes = sum(item["overtime_minutes"] for item in daily_totals.values())
    attendance_meta = {}
    for row in records:
        open_break = db.execute("SELECT break_start FROM attendance_breaks WHERE attendance_id=? AND break_end IS NULL ORDER BY break_start DESC LIMIT 1", (row["id"],)).fetchone()
        attendance_meta[row["id"]] = {
            "break_minutes": break_minutes_for_attendance(row["id"]),
            "on_break": bool(open_break),
            "break_start": open_break["break_start"] if open_break else None,
            "late": int(row["late_minutes"] or 0) > int(db.execute("SELECT grace_minutes FROM employees WHERE id=?", (row["employee_id"],)).fetchone()[0] or 0),
        }
    return render_template("attendance.html", employees=staff, records=records, today=today,
                           active_ids=active_ids, is_admin=is_admin, daily_totals=daily_totals,
                           attendance_meta=attendance_meta,
                           today_total_hours=hours_text(today_total_minutes),
                           today_total_overtime=hours_text(today_total_overtime_minutes),
                           daily_standard_hours=DAILY_STANDARD_HOURS)


@app.route("/face-enrollment", methods=["GET", "POST"])
@admin_required
def face_enrollment():
    db = get_db()
    if request.method == "POST":
        employee_id = request.form.get("employee_id", type=int)
        if request.form.get("consent_confirmed") != "yes":
            flash("Confirm that the employee has agreed to biometric enrollment before saving a sample.", "error")
            return redirect(url_for("face_enrollment", employee_id=employee_id or ""))
        employee = db.execute("SELECT * FROM employees WHERE id=?", (employee_id,)).fetchone() if employee_id else None
        if not employee:
            flash("Choose a valid employee first.", "error")
            return redirect(url_for("face_enrollment"))
        try:
            crop = decode_and_crop_face(request.form.get("image_data", ""))
            folder = FACE_DATA_DIR / str(employee_id)
            folder.mkdir(parents=True, exist_ok=True)
            filename = folder / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png")
            if not cv2.imwrite(str(filename), crop):
                raise ValueError("Could not save the face sample. Check folder permissions.")
            count = len(face_sample_files(employee_id))
            flash(f"Saved face sample {count} for {employee['name']}. Capture at least {FACE_MIN_SAMPLES} samples with slightly different natural angles.", "success")
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception as exc:
            app.logger.exception("Face enrollment failed")
            flash("Face enrollment failed. Check the terminal for details.", "error")
        return redirect(url_for("face_enrollment", employee_id=employee_id))

    employees = db.execute("SELECT id, name, email, department FROM employees ORDER BY name COLLATE NOCASE").fetchall()
    counts = face_sample_counts()
    selected = request.args.get("employee_id", type=int)
    return render_template("face_enrollment.html", employees=employees, sample_counts=counts,
                           selected_employee_id=selected, min_samples=FACE_MIN_SAMPLES)


@app.post("/face-enrollment/<int:employee_id>/delete")
@admin_required
def delete_face_profile(employee_id):
    employee = get_db().execute("SELECT id, name FROM employees WHERE id=?", (employee_id,)).fetchone()
    if not employee:
        abort(404)
    folder = FACE_DATA_DIR / str(employee_id)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    flash(f"Local face samples deleted for {employee['name']}.", "success")
    return redirect(url_for("face_enrollment"))


@app.route("/face-attendance", methods=["GET", "POST"])
@login_required
def face_attendance():
    user = get_current_user()
    if user["role"] == "employee" and not user["employee_id"]:
        abort(403)
    if request.method == "POST":
        challenge = session.pop("face_challenge", None)  # One-use challenge prevents simple form replay.
        try:
            now = int(time.time())
            locked_until = int(session.get("face_locked_until", 0))
            if now < locked_until:
                wait = locked_until - now
                raise ValueError(f"Too many unsuccessful scans. Wait {wait} seconds and try again.")
            if not challenge:
                raise ValueError("Live-check challenge expired. Refresh the page and try again.")
            crop = verify_head_turn_challenge(
                request.form.get("baseline_data", ""),
                request.form.get("image_data", ""),
                challenge,
            )
            employee, confidence = recognize_employee(crop)
            if user["role"] == "employee" and int(user["employee_id"]) != int(employee["id"]):
                raise ValueError("The detected face does not match the employee linked to your account.")
            # Prevent rapid accidental toggles from creating an immediate clock-in/out pair.
            latest = get_db().execute(
                "SELECT clock_in, clock_out FROM attendance WHERE employee_id=? AND work_date=? ORDER BY clock_in DESC LIMIT 1",
                (employee["id"], current_day()),
            ).fetchone()
            if latest:
                try:
                    latest_time = datetime.fromisoformat(latest["clock_out"] or latest["clock_in"])
                    if (datetime.now() - latest_time).total_seconds() < FACE_ATTEMPT_COOLDOWN_SECONDS:
                        raise ValueError("Attendance was just updated. Wait a few seconds before scanning again.")
                except (TypeError, ValueError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("Attendance was just"):
                        raise
            message = record_face_attendance(employee["id"])
            session.pop("face_failures", None)
            session.pop("face_locked_until", None)
            flash(message, "success")
        except (ValueError, RuntimeError) as exc:
            failures = int(session.get("face_failures", 0)) + 1
            session["face_failures"] = failures
            if failures >= 5:
                session["face_locked_until"] = int(time.time()) + 60
                session["face_failures"] = 0
            flash(str(exc), "error")
        except Exception:
            app.logger.exception("Face attendance failed")
            flash("Face recognition failed. Check the VS Code terminal and try again.", "error")
        return redirect(url_for("face_attendance"))
    challenge = random.choice(("left", "right"))
    session["face_challenge"] = challenge
    return render_template("face_attendance.html", min_samples=FACE_MIN_SAMPLES,
                           face_challenge=challenge)


# =========================
# PAYROLL MANAGEMENT
# =========================

@app.route("/payroll", methods=["GET", "POST"])
@admin_required
def payroll():
    db = get_db()
    month_value = request.args.get("month", date.today().strftime("%Y-%m"))
    
    if request.method == "POST":
        action = request.form.get("action")
        target_month = request.form.get("month", month_value)
        
        if action == "generate":
            # Fetch all attendance for the month
            month_start, month_end = month_bounds(target_month)
            records = db.execute("""
                SELECT a.*, e.name, e.email, e.department, e.hourly_rate 
                FROM attendance a JOIN employees e ON e.id=a.employee_id 
                WHERE a.work_date BETWEEN ? AND ?
            """, (month_start.isoformat(), month_end.isoformat())).fetchall()
            
            daily_summary = summarize_daily(records)
            
            # Aggregate by employee
            payroll_data = {}
            for row in daily_summary:
                eid = row["employee_id"]
                if eid not in payroll_data:
                    # Fetch hourly rate and custom multiplier
                    emp_data = db.execute("SELECT hourly_rate, overtime_multiplier FROM employees WHERE id=?", (eid,)).fetchone()
                    payroll_data[eid] = {
                        "regular": 0, 
                        "overtime": 0, 
                        "rate": emp_data["hourly_rate"],
                        "multiplier": dict(emp_data).get("overtime_multiplier", 1.5)
                    }
                
                payroll_data[eid]["regular"] += (row["minutes"] - row["overtime_minutes"])
                payroll_data[eid]["overtime"] += row["overtime_minutes"]
            
            # Insert or update payroll records
            for eid, data in payroll_data.items():
                reg_hours = data["regular"] / 60.0
                ot_hours = data["overtime"] / 60.0
                
                # Calculate using their custom multiplier instead of hardcoded 1.5
                gross_pay = (reg_hours * data["rate"]) + (ot_hours * data["rate"] * data["multiplier"])
                
                db.execute("""
                    INSERT INTO payroll (employee_id, period_month, total_regular_minutes, total_overtime_minutes, hourly_rate, gross_pay, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(employee_id, period_month) DO UPDATE SET
                    total_regular_minutes=excluded.total_regular_minutes,
                    total_overtime_minutes=excluded.total_overtime_minutes,
                    hourly_rate=excluded.hourly_rate,
                    gross_pay=excluded.gross_pay
                """, (eid, target_month, data["regular"], data["overtime"], data["rate"], gross_pay, datetime.now().isoformat(timespec="seconds")))
                
                db.execute("""
                    INSERT INTO payroll (employee_id, period_month, total_regular_minutes, total_overtime_minutes, hourly_rate, gross_pay, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(employee_id, period_month) DO UPDATE SET
                    total_regular_minutes=excluded.total_regular_minutes,
                    total_overtime_minutes=excluded.total_overtime_minutes,
                    hourly_rate=excluded.hourly_rate,
                    gross_pay=excluded.gross_pay
                """, (eid, target_month, data["regular"], data["overtime"], data["rate"], gross_pay, datetime.now().isoformat(timespec="seconds")))
            db.commit()
            flash(f"Payroll generated for {target_month}.", "success")
            
        elif action == "mark_paid":
            payroll_id = request.form.get("payroll_id")
            db.execute("UPDATE payroll SET status='Paid' WHERE id=?", (payroll_id,))
            db.commit()
            flash("Payslip marked as Paid.", "success")
            
        return redirect(url_for("payroll", month=target_month))

    # GET: Display payroll for the selected month
    payrolls = db.execute("""
        SELECT p.*, e.name, e.department, e.email 
        FROM payroll p JOIN employees e ON e.id = p.employee_id 
        WHERE p.period_month = ? ORDER BY e.name COLLATE NOCASE
    """, (month_value,)).fetchall()
    
    total_payout = sum(p["gross_pay"] for p in payrolls)
    
    return render_template("payroll.html", payrolls=payrolls, month_value=month_value, total_payout=total_payout)


@app.route("/my-payslips")
@login_required
def my_payslips():
    user = get_current_user()
    db = get_db()
    
    if not user["employee_id"]:
        flash("No employee profile linked to this account.", "error")
        return redirect(url_for("attendance"))
        
    payslips = db.execute("""
        SELECT * FROM payroll 
        WHERE employee_id = ? 
        ORDER BY period_month DESC
    """, (user["employee_id"],)).fetchall()
    
    return render_template("my_payslips.html", payslips=payslips)


# =========================
# LEAVE MANAGEMENT
# =========================


## =========================
# LEAVE MANAGEMENT
# =========================

def get_leave_balances(employee_id):
    db = get_db()
    emp = db.execute("SELECT casual_leave, sick_leave, personal_leave FROM employees WHERE id=?", (employee_id,)).fetchone()
    
    used = db.execute("""
        SELECT leave_type, SUM(CAST(julianday(end_date) - julianday(start_date) + 1 AS INTEGER)) as days
        FROM leave_requests
        WHERE employee_id=? AND status='Approved'
        GROUP BY leave_type
    """, (employee_id,)).fetchall()
    
    used_map = {row["leave_type"]: row["days"] for row in used}
    
    return {
        "Casual": {"total": emp["casual_leave"], "used": used_map.get("Casual", 0), "remaining": emp["casual_leave"] - used_map.get("Casual", 0)},
        "Sick": {"total": emp["sick_leave"], "used": used_map.get("Sick", 0), "remaining": emp["sick_leave"] - used_map.get("Sick", 0)},
        "Personal": {"total": emp["personal_leave"], "used": used_map.get("Personal", 0), "remaining": emp["personal_leave"] - used_map.get("Personal", 0)},
        "Other": {"total": 0, "used": used_map.get("Other", 0), "remaining": 0}
    }


@app.route("/leave", methods=["GET", "POST"])
@login_required
def leave():
    

    user = get_current_user()
    db = get_db()

    # Find the employee linked to this login
    employee = None

    if user and user["employee_id"]:
        employee = db.execute(
            "SELECT * FROM employees WHERE id = ?",
            (user["employee_id"],)
        ).fetchone()

    if not employee:
        flash("Your login is not linked to an employee profile.", "error")
        return redirect(url_for("attendance"))

    if request.method == "POST":
        leave_type = request.form.get("leave_type", "").strip()
        start_date = request.form.get("start_date", "").strip()
        end_date = request.form.get("end_date", "").strip()
        reason = request.form.get("reason", "").strip()

        if leave_type not in ("Casual", "Sick", "Personal", "Other"):
            flash("Please select a valid leave type.", "error")
        elif not start_date or not end_date or not reason:
            flash("Please complete all fields.", "error")
        elif len(reason) > 1000:
            flash("Reason must be 1000 characters or fewer.", "error")
        else:
            try:
                start = date.fromisoformat(start_date)
                end = date.fromisoformat(end_date)
                days_requested = (end - start).days + 1

                if start > end:
                    flash("End date cannot be before start date.", "error")
                else:
                    # NEW: Check leave balances first
                    balances = get_leave_balances(employee["id"])
                    if leave_type in balances and leave_type != "Other" and days_requested > balances[leave_type]["remaining"]:
                        flash(f"Insufficient {leave_type} leave. You requested {days_requested} day(s), but only have {balances[leave_type]['remaining']} remaining.", "error")
                    else:
                        overlap = db.execute(
                        """
                        SELECT id FROM leave_requests
                        WHERE employee_id = ?
                          AND status IN ('Pending', 'Approved')
                          AND start_date <= ?
                          AND end_date >= ?
                        LIMIT 1
                        """,
                        (employee["id"], end_date, start_date)
                    ).fetchone()

                    if overlap:
                        flash(
                            "You already have a pending or approved "
                            "leave request for overlapping dates.",
                            "error"
                        )
                    else:
                        db.execute(
                            """
                            INSERT INTO leave_requests
                            (employee_id, leave_type, start_date, end_date,
                             reason, status, created_at)
                            VALUES (?, ?, ?, ?, ?, 'Pending', ?)
                            """,
                            (
                                employee["id"],
                                leave_type,
                                start_date,
                                end_date,
                                reason,
                                datetime.now().isoformat(timespec="seconds")
                            )
                        )
                        db.commit()
                        flash("Leave request submitted successfully.", "success")
                        return redirect(url_for("leave"))

            except ValueError:
                flash("Please enter valid dates.", "error")

    requests = db.execute(
        """
        SELECT * FROM leave_requests
        WHERE employee_id = ?
        ORDER BY created_at DESC
        """,
        (employee["id"],)
    ).fetchall()

    
    return render_template(
        "leave.html",
        employee=employee,
        leave_requests=requests,
        balances=get_leave_balances(employee["id"]), # ADD THIS LINE HERE
        leave_types=[
            {"value": "Casual", "label": "Casual Leave"},
            {"value": "Sick", "label": "Sick Leave"},
            {"value": "Personal", "label": "Personal Leave"},
            {"value": "Other", "label": "Other"}
        ],
        today=date.today().isoformat(),
        user=user
    )


@app.route("/leave/cancel/<int:request_id>", methods=["POST"])
@login_required
def cancel_leave_request(request_id):
    user = get_current_user()
    db = get_db()

    employee = None

    if user and user["employee_id"]:
        employee = db.execute(
            "SELECT id FROM employees WHERE id = ?",
            (user["employee_id"],)
        ).fetchone()
    if not employee:
        abort(403)

    result = db.execute(
        """
        UPDATE leave_requests
        SET status = 'Cancelled'
        WHERE id = ?
          AND employee_id = ?
          AND status = 'Pending'
        """,
        (request_id, employee["id"])
    )

    db.commit()

    if result.rowcount:
        flash("Leave request cancelled.", "success")
    else:
        flash("Request not found or cannot be cancelled.", "error")

    return redirect(url_for("leave"))


@app.route("/leave-requests", methods=["GET", "POST"])
@admin_required
def leave_requests():
    db = get_db()
    user = get_current_user()

    if request.method == "POST":
        request_id = request.form.get("request_id", type=int)
        decision = request.form.get("decision", "").strip()
        admin_note = request.form.get("admin_note", "").strip()

        if not request_id or decision not in ("Approved", "Rejected"):
            flash("Invalid leave review request.", "error")
            return redirect(url_for("leave_requests"))

        if len(admin_note) > 1000:
            flash("Admin note must be 1000 characters or fewer.", "error")
            return redirect(url_for("leave_requests"))

        leave_item = db.execute(
            """
            SELECT * FROM leave_requests
            WHERE id = ? AND status = 'Pending'
            """,
            (request_id,)
        ).fetchone()

        if not leave_item:
            flash("This request is no longer pending.", "error")
            return redirect(url_for("leave_requests"))

        if decision == "Approved":
            overlap = db.execute(
                """
                SELECT id FROM leave_requests
                WHERE employee_id = ?
                  AND status = 'Approved'
                  AND id != ?
                  AND start_date <= ?
                  AND end_date >= ?
                LIMIT 1
                """,
                (
                    leave_item["employee_id"],
                    request_id,
                    leave_item["end_date"],
                    leave_item["start_date"]
                )
            ).fetchone()

            if overlap:
                flash(
                    "Cannot approve: this employee already has approved "
                    "leave for overlapping dates.",
                    "error"
                )
                return redirect(url_for("leave_requests"))

        result = db.execute(
            """
            UPDATE leave_requests
            SET status = ?,
                admin_note = ?,
                reviewed_by_user_id = ?,
                reviewed_at = ?
            WHERE id = ? AND status = 'Pending'
            """,
            (
                decision,
                admin_note or None,
                user["id"],
                datetime.now().isoformat(timespec="seconds"),
                request_id
            )
        )

        db.commit()

        if result.rowcount:
            flash(f"Leave request {decision.lower()}.", "success")
        else:
            flash("The request was already reviewed.", "error")

        return redirect(url_for("leave_requests"))

    status = request.args.get("status", "Pending")
    allowed_statuses = (
        "Pending", "Approved", "Rejected", "Cancelled", "All"
    )

    if status not in allowed_statuses:
        status = "Pending"

    if status == "All":
            
        requests_list = db.execute(
            """
            SELECT
                lr.*,
                e.name AS employee_name,
                e.email AS employee_email,
                e.department AS department,
                reviewer.username AS reviewer_username
            FROM leave_requests lr
            JOIN employees e ON e.id = lr.employee_id
            LEFT JOIN users reviewer
                ON reviewer.id = lr.reviewed_by_user_id
            ORDER BY
                CASE lr.status
                    WHEN 'Pending' THEN 0
                    ELSE 1
                END,
                lr.created_at DESC
            """
        ).fetchall()

    else:
       
        requests_list = db.execute(
            """
            SELECT
                lr.*,
                e.name AS employee_name,
                e.email AS employee_email,
                e.department AS department,
                reviewer.username AS reviewer_username
            FROM leave_requests lr
            JOIN employees e ON e.id = lr.employee_id
            LEFT JOIN users reviewer
                ON reviewer.id = lr.reviewed_by_user_id
            WHERE lr.status = ?
            ORDER BY lr.created_at DESC
            """,
            (status,)
        ).fetchall()

    return render_template(
        "leave_requests.html",
        leave_requests=[
            dict(item) for item in requests_list
        ],
        status_filter=status,
        counts={
            row["status"]: row["total"]
            for row in db.execute(
                """
                SELECT status, COUNT(*) AS total
                FROM leave_requests
                GROUP BY status
                """
            ).fetchall()
        },
        user=user
    )


@app.route("/reports")
@admin_required
def reports():
    db = get_db()
    start, end, employee_id, records = report_filters(db)
    staff = db.execute("SELECT id,name FROM employees ORDER BY name COLLATE NOCASE").fetchall()
    daily_rows = summarize_daily(records)
    return render_template("reports.html", records=records, daily_rows=daily_rows, employees=staff,
        start=start, end=end, employee_id=employee_id,
        total_hours=hours_text(sum(x["minutes"] for x in daily_rows)),
        total_overtime=hours_text(sum(x["overtime_minutes"] for x in daily_rows)),
        daily_standard_hours=DAILY_STANDARD_HOURS)


@app.route("/reports/export")
@admin_required
def export_reports():
    db = get_db()
    start, end, employee_id, records = report_filters(db)
    daily_rows = summarize_daily(records)
    daily_overtime = {(x["work_date"],x["employee_id"]):x["overtime"] for x in daily_rows}
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date","Employee","Email","Department","Clock In","Clock Out","Shift Duration","Daily Overtime (over 8h standard)"])
    for row in records:
        writer.writerow([row["work_date"],row["name"],row["email"],row["department"],row["clock_in"],row["clock_out"] or "",
                         duration_text(row["clock_in"],row["clock_out"]),daily_overtime.get((row["work_date"],row["employee_id"]),"0h 00m")])
    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=HUB pulse_attendance_report.csv"
    return response



@app.route("/payslip/<int:payroll_id>")
@login_required
def view_payslip(payroll_id):
    user = get_current_user()
    db = get_db()
    
    payslip = db.execute("""
        SELECT p.*, e.name, e.department, e.email 
        FROM payroll p JOIN employees e ON e.id = p.employee_id 
        WHERE p.id = ?
    """, (payroll_id,)).fetchone()
    
    if not payslip:
        abort(404, description="Payslip not found.")
        
    if user["role"] != "admin" and user["employee_id"] != payslip["employee_id"]:
        abort(403, description="You do not have permission to view this payslip.")
        
    return render_template("payslip.html", payslip=payslip)


def create_admin_cli():
    with app.app_context():
        init_db()
        db = get_db()

        existing_admin = db.execute("""
            SELECT id FROM users WHERE role = 'admin' LIMIT 1
        """).fetchone()

        if existing_admin:
            # Ensure the existing administrator is protected.
            db.execute("""
                UPDATE users
                SET is_permanent_admin = 1
                WHERE id = ?
            """, (existing_admin["id"],))
            db.commit()
            print("An administrator already exists.")
            print("The existing administrator has been marked permanent.")
            print("Sign in using that account.")
            return

        print("Create your permanent HUB pulse administrator account.")
        username = input("Admin username: ").strip()

        while not username:
            username = input("Username cannot be empty. Admin username: ").strip()

        while True:
            password = getpass.getpass("Choose a password (at least 12 characters): ")

            if len(password) < 12:
                print("Password must contain at least 12 characters.")
                continue

            confirm = getpass.getpass("Confirm password: ")

            if password != confirm:
                print("Passwords do not match. Try again.")
                continue

            break

        try:
            db.execute("""
                INSERT INTO users (
                    username,
                    password_hash,
                    role,
                    employee_id,
                    created_at,
                    is_permanent_admin
                )
                VALUES (?, ?, 'admin', NULL, ?, 1)
            """, (
                username,
                generate_password_hash(password),
                datetime.now().isoformat(timespec="seconds")
            ))

            db.commit()
            print("Permanent administrator created successfully.")
            print("Username:", username)
            print("Your password is stored as a secure hash.")

        except sqlite3.IntegrityError:
            db.rollback()
            print("That username is already in use. Choose another username.")




# if __name__ == "__main__":
#     if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
#         create_admin_cli()
#     else:
#         # ADD host="0.0.0.0" HERE:
#         app.run(host="0.0.0.0", debug=os.environ.get("FLASK_DEBUG", "0") == "1")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create-admin":
        create_admin_cli()
    else:
        with app.app_context():
            init_db()

        app.run(
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1"
        )

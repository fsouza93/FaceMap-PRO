# app.py — Biblioteca (FaceID como chave), integra com FaceMap
import sqlite3, os, time
from datetime import date, datetime
from flask import Flask, g, request, redirect, url_for, render_template, flash, jsonify
from jinja2 import DictLoader

# ======================================================================
# Config
# ======================================================================
APP_TITLE = "Biblioteca — Empréstimos & Pendências"
DB_PATH = "library.db"

# Se você expõe a API no mesmo Flask, a câmera pode apontar para:
# FACEMAP_API_BASE = http://127.0.0.1:5000
# (o lado da câmera normaliza se vier com /api no fim)
DUE_SOON_DAYS = 3  # “vencendo logo” = faltam <= 3 dias

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# ======================================================================
# DB helpers (+ migração leve)
# ======================================================================
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def column_exists(db, table, colname):
    row = db.execute("PRAGMA table_info(%s)" % table).fetchall()
    return any(r["name"] == colname for r in row)

def init_db():
    db = get_db()
    # Tabelas base (sem UNIQUE em RA; FaceID passa a ser único)
    db.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS students (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id   TEXT,           -- único via índice (pode estar NULL em migrações antigas)
            full_name TEXT,
            ra        TEXT,
            course    TEXT,
            semester  TEXT,
            phone     TEXT
        );

        CREATE TABLE IF NOT EXISTS loans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER NOT NULL,
            book_title    TEXT NOT NULL,
            checkout_date TEXT NOT NULL,
            due_date      TEXT NOT NULL,
            returned      INTEGER NOT NULL DEFAULT 0,
            returned_at   TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_loans_student ON loans(student_id);
        CREATE INDEX IF NOT EXISTS idx_loans_due     ON loans(due_date);
        """
    )

    # Migração leve: garante coluna face_id e índice único condicional
    if not column_exists(db, "students", "face_id"):
        db.execute("ALTER TABLE students ADD COLUMN face_id TEXT")

    # Único apenas quando NÃO for NULL (para não quebrar registros antigos)
    db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_students_face_id "
        "ON students(face_id) WHERE face_id IS NOT NULL"
    )
    db.commit()

def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def is_overdue(loan_row, today=None):
    if today is None:
        today = date.today()
    d = parse_date(loan_row["due_date"])
    return (loan_row["returned"] == 0) and d is not None and d < today

# ======================================================================
# Memória (sinalização do fluxo FaceID)
# ======================================================================
# A câmera faz:
#   POST /api/face-id {face_id}
# e depois fica consultando:
#   GET  /api/face-id/status/<face_id>
# Nós marcamos "confirmed" = True somente quando o usuário salvar o aluno (Nome+RA).
PENDING_FACE = {}  # face_id -> {"confirmed": bool, "cancelled": bool, "ts": float}

def set_pending(face_id, confirmed=False, cancelled=False):
    if not face_id:
        return
    PENDING_FACE[face_id] = {
        "confirmed": bool(confirmed),
        "cancelled": bool(cancelled),
        "ts": time.time(),
    }

# ======================================================================
# Templates (Jinja)
# ======================================================================
BASE = """
{% macro base(title, app_title) -%}
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8">
    <title>{{ title or "App" }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { padding-top: 70px; }
      .badge-overdue { background-color: #dc3545; }
      .badge-ok { background-color: #198754; }
      .table-fixed { table-layout: fixed; }
      .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .form-section { background: #f8f9fa; padding: 1rem; border-radius: .5rem; }
    </style>
  </head>
  <body>
    <nav class="navbar fixed-top navbar-expand-lg bg-dark navbar-dark">
      <div class="container-fluid">
        <a class="navbar-brand" href="{{ url_for('dashboard') }}">{{ app_title }}</a>
        <div class="ms-auto">
          <a class="btn btn-sm btn-outline-light" href="{{ url_for('new_student') }}">+ Novo Aluno</a>
          <a class="btn btn-sm btn-outline-warning" href="{{ url_for('new_loan') }}">+ Novo Empréstimo</a>
        </div>
      </div>
    </nav>
    <main class="container">
      {% with msgs = get_flashed_messages() %}
        {% if msgs %}
          <div class="alert alert-info">{{ msgs[0] }}</div>
        {% endif %}
      {% endwith %}
      {{ caller() }}
    </main>
  </body>
</html>
{%- endmacro %}
"""

DASHBOARD = """
{% from 'base' import base %}
{% call base(title, app_title) %}
  <form class="row g-2 mb-3" method="get">
    <div class="col-sm-4">
      <input class="form-control" name="q" placeholder="Buscar por nome, RA ou FaceID" value="{{ q or '' }}">
    </div>
    <div class="col-sm-3">
      <select class="form-select" name="filter">
        <option value="" {{ ''==filter and 'selected' or '' }}>Todos</option>
        <option value="overdue" {{ 'overdue'==filter and 'selected' or '' }}>Com pendencias (atraso)</option>
        <option value="clean"   {{ 'clean'==filter and 'selected' or '' }}>Sem pendencias</option>
      </select>
    </div>
    <div class="col-sm-2">
      <button class="btn btn-primary w-100">Filtrar</button>
    </div>
  </form>

  <div class="card">
    <div class="card-header">Alunos</div>
    <div class="table-responsive">
      <table class="table table-hover table-fixed align-middle mb-0">
        <thead>
          <tr>
            <th style="width:20%">Nome</th>
            <th style="width:12%">FaceID</th>
            <th style="width:12%">RA</th>
            <th style="width:14%">Curso</th>
            <th style="width:10%">Semestre</th>
            <th style="width:14%">Telefone</th>
            <th style="width:8%" class="text-center">Pend.</th>
            <th style="width:10%">Ações</th>
          </tr>
        </thead>
        <tbody>
          {% for s in students %}
            <tr>
              <td class="truncate"><a href="{{ url_for('student_detail', student_id=s.id) }}">{{ s.full_name or '—' }}</a></td>
              <td class="truncate">{{ s.face_id or '—' }}</td>
              <td class="truncate">{{ s.ra or '—' }}</td>
              <td class="truncate">{{ s.course or '-' }}</td>
              <td class="truncate">{{ s.semester or '-' }}</td>
              <td class="truncate">{{ s.phone or '-' }}</td>
              <td class="text-center">
                {% if s.overdue_count > 0 %}
                  <span class="badge rounded-pill badge-overdue">{{ s.overdue_count }}</span>
                {% else %}
                  <span class="badge rounded-pill badge-ok">0</span>
                {% endif %}
              </td>
              <td>
                <a class="btn btn-sm btn-outline-primary" href="{{ url_for('edit_student', student_id=s.id) }}">Editar</a>
              </td>
            </tr>
          {% endfor %}
          {% if not students %}
            <tr><td colspan="8" class="text-center text-secondary py-4">Nenhum aluno encontrado</td></tr>
          {% endif %}
        </tbody>
      </table>
    </div>
  </div>
{% endcall %}
"""

NEW_STUDENT = """
{% from 'base' import base %}
{% call base(title, app_title) %}
  <div class="row">
    <div class="col-lg-8">
      <div class="form-section">
        <h5 class="mb-3">Cadastro do aluno</h5>
        <form method="post">
          <div class="row g-3">
            <div class="col-sm-4">
              <label class="form-label">Face ID *</label>
              <input required class="form-control" name="face_id" value="{{ prefill_face_id or '' }}">
              <div class="form-text">Gerado pela câmera.</div>
            </div>
            <div class="col-sm-4">
              <label class="form-label">Nome completo *</label>
              <input required class="form-control" name="full_name" autofocus>
            </div>
            <div class="col-sm-4">
              <label class="form-label">RA *</label>
              <input required class="form-control" name="ra" value="{{ prefill_ra or '' }}">
            </div>
            <div class="col-sm-6">
              <label class="form-label">Curso</label>
              <input class="form-control" name="course">
            </div>
            <div class="col-sm-3">
              <label class="form-label">Semestre</label>
              <input class="form-control" name="semester" placeholder="ex.: 3º">
            </div>
            <div class="col-sm-3">
              <label class="form-label">Telefone</label>
              <input class="form-control" name="phone" placeholder="(xx) xxxxx-xxxx">
            </div>
          </div>
          <div class="mt-3 d-flex gap-2">
            <button class="btn btn-primary">Salvar</button>
            <a class="btn btn-outline-secondary" href="{{ url_for('dashboard') }}">Fechar</a>
            {% if prefill_face_id %}
            <form method="post" action="{{ url_for('cancel_faceid', face_id=prefill_face_id) }}" class="d-inline">
              <button class="btn btn-outline-danger">Cancelar cadastro</button>
            </form>
            {% endif %}
          </div>
        </form>
      </div>
    </div>
  </div>
{% endcall %}
"""

EDIT_STUDENT = """
{% from 'base' import base %}
{% call base(title, app_title) %}
  <div class="row">
    <div class="col-lg-8">
      <div class="form-section">
        <h5 class="mb-3">Editar aluno</h5>
        <form method="post">
          <div class="row g-3">
            <div class="col-sm-4">
              <label class="form-label">Face ID</label>
              <input class="form-control" name="face_id" value="{{ s.face_id or '' }}" readonly>
              <div class="form-text">FaceID é gerado pela câmera.</div>
            </div>
            <div class="col-sm-4">
              <label class="form-label">Nome completo *</label>
              <input required class="form-control" name="full_name" value="{{ s.full_name or '' }}">
            </div>
            <div class="col-sm-4">
              <label class="form-label">RA *</label>
              <input required class="form-control" name="ra" value="{{ s.ra or '' }}">
            </div>
            <div class="col-sm-6">
              <label class="form-label">Curso</label>
              <input class="form-control" name="course" value="{{ s.course or '' }}">
            </div>
            <div class="col-sm-3">
              <label class="form-label">Semestre</label>
              <input class="form-control" name="semester" value="{{ s.semester or '' }}">
            </div>
            <div class="col-sm-3">
              <label class="form-label">Telefone</label>
              <input class="form-control" name="phone" value="{{ s.phone or '' }}">
            </div>
          </div>
          <div class="mt-3 d-flex gap-2">
            <button class="btn btn-primary">Salvar alterações</button>
            <a class="btn btn-outline-secondary" href="{{ url_for('student_detail', student_id=s.id) }}">Cancelar</a>
          </div>
        </form>
      </div>
    </div>
  </div>
{% endcall %}
"""

STUDENT_DETAIL = """
{% from 'base' import base %}
{% call base(title, app_title) %}
  <div class="row g-3">
    <div class="col-lg-7">
      <div class="card">
        <div class="card-header d-flex justify-content-between align-items-center">
          <span>Dados do aluno</span>
          <div class="d-flex gap-2">
            <a class="btn btn-sm btn-outline-primary" href="{{ url_for('edit_student', student_id=s.id) }}">Editar</a>
            <form method="post"
                  action="{{ url_for('delete_student', student_id=s.id) }}"
                  onsubmit="return confirm('Excluir este aluno e TODOS os seus empréstimos?');">
              <button class="btn btn-sm btn-outline-danger">Excluir aluno</button>
            </form>
          </div>
        </div>
        <div class="card-body">
          <div class="row g-3">
            <div class="col-sm-6"><strong>Nome:</strong> {{ s.full_name or '—' }}</div>
            <div class="col-sm-3"><strong>FaceID:</strong> {{ s.face_id or '—' }}</div>
            <div class="col-sm-3"><strong>RA:</strong> {{ s.ra or '—' }}</div>
            <div class="col-sm-6"><strong>Curso:</strong> {{ s.course or '-' }}</div>
            <div class="col-sm-3"><strong>Semestre:</strong> {{ s.semester or '-' }}</div>
            <div class="col-sm-3"><strong>Telefone:</strong> {{ s.phone or '-' }}</div>
          </div>
        </div>
      </div>

      <div class="card mt-3">
        <div class="card-header d-flex justify-content-between align-items-center">
          <span>Empréstimos</span>
          <a class="btn btn-sm btn-warning" href="{{ url_for('new_loan', student_id=s.id) }}">+ Novo empréstimo</a>
        </div>
        <div class="table-responsive">
          <table class="table align-middle mb-0">
            <thead>
              <tr>
                <th>Livro</th>
                <th>Retirada</th>
                <th>Devolução</th>
                <th>Status</th>
                <th style="width:200px"></th>
              </tr>
            </thead>
            <tbody>
            {% for l in loans %}
              {% set overdue = (l.returned == 0) and (l.due_date < today_str) %}
              <tr class="{{ 'table-danger' if overdue else '' }}">
                <td class="truncate">{{ l.book_title }}</td>
                <td>{{ l.checkout_date }}</td>
                <td>{{ l.due_date }}</td>
                <td>
                  {% if l.returned %}
                    <span class="badge bg-success">Devolvido</span>
                  {% elif overdue %}
                    <span class="badge bg-danger">Em atraso</span>
                  {% else %}
                    <span class="badge bg-secondary">Em aberto</span>
                  {% endif %}
                </td>
                <td>
                  {% if not l.returned %}
                  <form class="d-inline" method="post" action="{{ url_for('return_loan', loan_id=l.id) }}">
                    <button class="btn btn-sm btn-outline-success">Marcar devolvido</button>
                  </form>
                  {% endif %}
                  <form class="d-inline" method="post"
                        action="{{ url_for('delete_loan', loan_id=l.id) }}"
                        onsubmit="return confirm('Excluir este empréstimo?');">
                    <button class="btn btn-sm btn-outline-danger">Excluir</button>
                  </form>
                </td>
              </tr>
            {% endfor %}
            {% if not loans %}
              <tr><td colspan="5" class="text-center text-secondary py-4">Nenhum empréstimo</td></tr>
            {% endif %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="col-lg-5">
      <div class="card">
        <div class="card-header">Resumo</div>
        <div class="card-body">
          <p><strong>Pendencias (atraso):</strong> {{ overdue_count }}</p>
          {% if overdue_titles %}
            <p class="mb-1"><strong>Livros em atraso:</strong></p>
            {% for t in overdue_titles %}
              <span class="badge bg-danger mb-1">{{ t }}</span>
            {% endfor %}
          {% else %}
            <p>Nenhuma pendencia 👌</p>
          {% endif %}
        </div>
      </div>
    </div>
  </div>
{% endcall %}
"""

NEW_LOAN = """
{% from 'base' import base %}
{% call base(title, app_title) %}
  <div class="row">
    <div class="col-lg-7">
      <div class="form-section">
        <h5 class="mb-3">Novo empréstimo</h5>
        <form method="post">
          <div class="row g-3">
            <div class="col-sm-6">
              <label class="form-label">Aluno *</label>
              <select required class="form-select" name="student_id">
                <option value="">Selecione...</option>
                {% for s in students %}
                  <option value="{{ s.id }}" {{ 'selected' if s.id == selected_id else '' }}>
                    {{ s.full_name or '—' }} — {{ s.face_id or '—' }}
                  </option>
                {% endfor %}
              </select>
            </div>
            <div class="col-sm-6">
              <label class="form-label">Livro *</label>
              <input required class="form-control" name="book_title" placeholder="Título do livro">
            </div>
            <div class="col-sm-6">
              <label class="form-label">Retirada *</label>
              <input required type="date" class="form-control" name="checkout_date" value="{{ today }}">
            </div>
            <div class="col-sm-6">
              <label class="form-label">Devolução (prevista) *</label>
              <input required type="date" class="form-control" name="due_date">
            </div>
          </div>
          <div class="mt-3">
            <button class="btn btn-warning">Salvar</button>
            <a class="btn btn-outline-secondary" href="{{ url_for('dashboard') }}">Cancelar</a>
          </div>
        </form>
      </div>
    </div>
  </div>
{% endcall %}
"""

app.jinja_loader = DictLoader({
    "base": BASE,
    "dashboard.html": DASHBOARD,
    "new_student.html": NEW_STUDENT,
    "edit_student.html": EDIT_STUDENT,
    "student_detail.html": STUDENT_DETAIL,
    "new_loan.html": NEW_LOAN,
})

# ======================================================================
# Rotas Web
# ======================================================================
@app.route("/")
def dashboard():
    q = (request.args.get("q") or "").strip()
    flt = (request.args.get("filter") or "").strip()

    db = get_db()
    params, where = [], ""
    if q:
        where = "WHERE (LOWER(full_name) LIKE ? OR LOWER(ra) LIKE ? OR LOWER(face_id) LIKE ?)"
        qlike = f"%{q.lower()}%"
        params += [qlike, qlike, qlike]

    students = db.execute(f"SELECT * FROM students {where} ORDER BY full_name ASC", params).fetchall()

    enriched = []
    for s in students:
        loans = db.execute("SELECT * FROM loans WHERE student_id=? ORDER BY due_date ASC", (s["id"],)).fetchall()
        overdue_titles = [l["book_title"] for l in loans if is_overdue(l)]
        overdue_count = len(overdue_titles)

        if flt == "overdue" and overdue_count == 0:  # só com pendências
            continue
        if flt == "clean" and overdue_count > 0:     # só sem pendências
            continue

        enriched.append({
            "id": s["id"], "full_name": s["full_name"], "ra": s["ra"],
            "face_id": s["face_id"], "course": s["course"], "semester": s["semester"], "phone": s["phone"],
            "overdue_count": overdue_count, "overdue_titles": overdue_titles,
        })

    return render_template("dashboard.html", title="Dashboard", app_title=APP_TITLE, students=enriched, q=q, filter=flt)

@app.route("/students/new", methods=["GET", "POST"])
def new_student():
    if request.method == "POST":
        face_id   = (request.form.get("face_id") or "").strip()
        full_name = (request.form.get("full_name") or "").strip()
        ra        = (request.form.get("ra") or "").strip()
        course    = (request.form.get("course") or "").strip()
        semester  = (request.form.get("semester") or "").strip()
        phone     = (request.form.get("phone") or "").strip()

        # Regras: precisa FaceID + Nome + RA
        if not face_id or not full_name or not ra:
            flash("Preencha FaceID, Nome e RA.")
            return redirect(url_for("new_student", face_id=face_id))

        db = get_db()
        try:
            db.execute(
                "INSERT INTO students(face_id, full_name, ra, course, semester, phone) VALUES (?,?,?,?,?,?)",
                (face_id, full_name, ra, course, semester, phone)
            )
            db.commit()
            # confirma para a câmera
            set_pending(face_id, confirmed=True, cancelled=False)
            flash("Aluno cadastrado!")
            return redirect(url_for("dashboard"))
        except sqlite3.IntegrityError:
            flash("FaceID já está cadastrado.")
            return redirect(url_for("new_student", face_id=face_id))

    # GET — FaceID pré-preenchido via querystring ?face_id=... (vindo da câmera)
    prefill_face_id = (request.args.get("face_id") or "").strip()
    prefill_ra = (request.args.get("ra") or "").strip()
    return render_template(
        "new_student.html", title="Novo aluno", app_title=APP_TITLE,
        prefill_face_id=prefill_face_id, prefill_ra=prefill_ra
    )

@app.route("/students/<int:student_id>")
def student_detail(student_id):
    db = get_db()
    s = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not s:
        flash("Aluno não encontrado.")
        return redirect(url_for("dashboard"))

    loans = db.execute(
        "SELECT * FROM loans WHERE student_id=? ORDER BY returned ASC, due_date ASC",
        (student_id,)
    ).fetchall()

    today_str = date.today().strftime("%Y-%m-%d")
    overdue_titles = [l["book_title"] for l in loans if is_overdue(l)]
    return render_template(
        "student_detail.html",
        title=s["full_name"] or "Aluno", app_title=APP_TITLE,
        s=s, loans=loans, overdue_count=len(overdue_titles),
        overdue_titles=overdue_titles, today_str=today_str
    )

@app.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
def edit_student(student_id):
    db = get_db()
    s = db.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not s:
        flash("Aluno não encontrado.")
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        ra        = (request.form.get("ra") or "").strip()
        course    = (request.form.get("course") or "").strip()
        semester  = (request.form.get("semester") or "").strip()
        phone     = (request.form.get("phone") or "").strip()

        if not full_name or not ra:
            flash("Nome e RA são obrigatórios.")
            return redirect(url_for("edit_student", student_id=student_id))

        db.execute(
            "UPDATE students SET full_name=?, ra=?, course=?, semester=?, phone=? WHERE id=?",
            (full_name, ra, course, semester, phone, student_id)
        )
        db.commit()
        flash("Alterações salvas.")
        return redirect(url_for("student_detail", student_id=student_id))

    return render_template("edit_student.html", title="Editar aluno", app_title=APP_TITLE, s=s)

# --------- emprestimos ----------
@app.route("/loans/new", methods=["GET", "POST"])
def new_loan():
    db = get_db()
    students = db.execute("SELECT * FROM students ORDER BY full_name ASC").fetchall()

    if request.method == "POST":
        student_id    = request.form.get("student_id")
        book_title    = (request.form.get("book_title") or "").strip()
        checkout_date = request.form.get("checkout_date")
        due_date      = request.form.get("due_date")

        if not student_id or not book_title or not checkout_date or not due_date:
            flash("Preencha todos os campos obrigatórios.")
            return redirect(url_for("new_loan"))

        if parse_date(checkout_date) is None or parse_date(due_date) is None:
            flash("Datas inválidas.")
            return redirect(url_for("new_loan"))

        db.execute(
            "INSERT INTO loans(student_id, book_title, checkout_date, due_date, returned) VALUES (?,?,?,?,0)",
            (int(student_id), book_title, checkout_date, due_date)
        )
        db.commit()
        flash("Empréstimo registrado!")
        return redirect(url_for("student_detail", student_id=student_id))

    selected_id = request.args.get("student_id", type=int)
    return render_template(
        "new_loan.html", title="Novo empréstimo", app_title=APP_TITLE,
        students=students, selected_id=selected_id, today=date.today().strftime("%Y-%m-%d")
    )

@app.post("/loans/<int:loan_id>/return")
def return_loan(loan_id):
    db = get_db()
    row = db.execute("SELECT student_id FROM loans WHERE id=?", (loan_id,)).fetchone()
    if not row:
        flash("Empréstimo não encontrado.")
        return redirect(url_for("dashboard"))
    db.execute(
        "UPDATE loans SET returned=1, returned_at=? WHERE id=? AND returned=0",
        (date.today().strftime("%Y-%m-%d"), loan_id)
    )
    db.commit()
    flash("Devolução registrada!")
    return redirect(url_for("student_detail", student_id=row["student_id"]))

@app.post("/loans/<int:loan_id>/delete")
def delete_loan(loan_id):
    db = get_db()
    row = db.execute("SELECT student_id FROM loans WHERE id=?", (loan_id,)).fetchone()
    if not row:
        flash("Empréstimo não encontrado.")
        return redirect(url_for("dashboard"))
    db.execute("DELETE FROM loans WHERE id=?", (loan_id,))
    db.commit()
    flash("Empréstimo excluído.")
    return redirect(url_for("student_detail", student_id=row["student_id"]))

@app.post("/students/<int:student_id>/delete")
def delete_student(student_id):
    db = get_db()
    row = db.execute("SELECT id, full_name FROM students WHERE id=?", (student_id,)).fetchone()
    if not row:
        flash("Aluno não encontrado.")
        return redirect(url_for("dashboard"))
    db.execute("DELETE FROM students WHERE id=?", (student_id,))
    db.commit()
    flash(f"Aluno '{row['full_name'] or '—'}' excluído.")
    return redirect(url_for("dashboard"))

# ======================================================================
# API para FaceMap (câmera)
# ======================================================================
@app.post("/api/face-id")
def api_receive_face_id():
    data = request.get_json(silent=True) or {}
    face_id = (data.get("face_id") or "").strip()
    if not face_id:
        return jsonify({"ok": False, "error": "missing face_id"}), 400
    set_pending(face_id, confirmed=False, cancelled=False)
    return jsonify({"ok": True})

@app.get("/api/face-id/status/<face_id>")
def api_face_id_status(face_id):
    db = get_db()
    rec = PENDING_FACE.get(face_id) or {"confirmed": False, "cancelled": False}
    # Se já existir aluno com esse FaceID, consideramos confirmado:
    row = db.execute("SELECT id FROM students WHERE face_id=?", (face_id,)).fetchone()
    confirmed = bool(row) or bool(rec.get("confirmed"))
    return jsonify({"confirmed": confirmed, "cancelled": bool(rec.get("cancelled")), "registered": bool(row)})

@app.post("/api/face-id/cancel/<face_id>")
def cancel_faceid(face_id):
    set_pending(face_id, confirmed=False, cancelled=True)
    return jsonify({"ok": True})

# Botão "Cancelar cadastro" na UI (faz POST e redireciona)
@app.post("/students/cancel/<face_id>")
def cancel_faceid_ui(face_id):
    set_pending(face_id, confirmed=False, cancelled=True)
    flash("Cadastro cancelado.")
    return redirect(url_for("dashboard"))

# API consultada pela câmera para obter status/pendências
@app.get("/api/student/<person_id>")
def api_student_status(person_id):
    db = get_db()

    # 1) FaceID exato
    student = db.execute("SELECT * FROM students WHERE face_id = ? LIMIT 1", (person_id,)).fetchone()

    # 2) RA exato
    if student is None:
        student = db.execute("SELECT * FROM students WHERE ra = ? LIMIT 1", (person_id,)).fetchone()

    # 3) fallback por nome completo (case-insensitive)
    if student is None:
        student = db.execute("SELECT * FROM students WHERE LOWER(full_name)=LOWER(?) LIMIT 1", (person_id,)).fetchone()

    if student is None:
        return jsonify({"found": False, "person": None, "overdue_count": 0, "due_soon_count": 0, "loans": []})

    s = dict(student)

    open_loans = db.execute(
        """SELECT id, book_title AS title, due_date, returned
           FROM loans
           WHERE student_id = ? AND returned = 0
           ORDER BY due_date ASC""",
        (s["id"],)
    ).fetchall()

    today = date.today()
    loans_out, overdue, due_soon = [], 0, 0
    for r in open_loans:
        d = dict(r)
        dd = datetime.strptime(d["due_date"], "%Y-%m-%d").date()
        days_over = (today - dd).days
        if days_over > 0:
            status = "overdue"; overdue += 1
        else:
            days_left = (dd - today).days
            status = "due_soon" if days_left <= DUE_SOON_DAYS else "ok"
            if status == "due_soon": due_soon += 1
        loans_out.append({
            "id": d["id"], "title": d["title"],
            "due_date": dd.isoformat(), "days_overdue": max(days_over, 0),
            "status": status
        })

    return jsonify({
        "found": True,
        "person": {
            "id": s["id"], "name": s.get("full_name"),
            "ra": s.get("ra"), "face_id": s.get("face_id"),
            "curso": s.get("course"), "semestre": s.get("semester"), "phone": s.get("phone"),
        },
        "overdue_count": overdue,
        "due_soon_count": due_soon,
        "loans": loans_out
    })

# ======================================================================
# start
# ======================================================================
if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)

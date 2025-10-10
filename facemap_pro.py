# facemap_pro.py — FaceID + cadastro multi-frame (sem modal) + integração API (status/pendência) + espera de confirmação
import os, json, time, uuid, webbrowser
from pathlib import Path
from collections import deque, Counter
import cv2
import numpy as np


# ===== integração com o app Flask =====
_API_BASE_RAW = os.environ.get("FACEMAP_API_BASE", "http://127.0.0.1:5000").rstrip("/")
API_BASE = _API_BASE_RAW[:-4] if _API_BASE_RAW.endswith("/api") else _API_BASE_RAW
API_TTL  = 2.0  # segundos (cache leve para não spammar a API)
try:
    import requests
except Exception:
    requests = None

# cores (BGR)
COL_GREEN  = (0,200,0)
COL_RED    = (0,0,255)
COL_ORANGE = (0,165,255)
COL_GRAY   = (180,180,180)

# ---------- modelos ----------
DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
REC_MODEL = "models/face_recognition_sface_2021dec.onnx"

# ---------- parâmetros ----------
DB_PATH      = Path("db_faces.json")

# thresholds de reconhecimento (cosine)
COS_THR   = 0.65
GAP_THR   = 0.15
MARGIN    = 0.00

# votação
VOTE_MIN  = 7
VOTE_SIZE = 25

FRAME_W, FRAME_H = 640, 480
DETECT_EVERY = 8
MESH_EVERY   = 8
DRAW_LANDMARKS = True

# Gate de qualidade p/ cadastro
MIN_FACE   = 100
MIN_IO     = 28.0
MIN_BLUR   = 40.0

# Cadastro multi-frame
ENROLL_SAMPLES = 12
ENROLL_TIMEOUT = 20.0

# Espera adicional após coleta p/ confirmação no app
CONFIRM_WAIT_MAX  = 20.0   # segundos
CONFIRM_WAIT_STEP = 0.6    # intervalo de polling

# FaceMesh MediaPipe
MP_OK = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    MP_OK = True
except Exception:
    face_mesh = None

MOUTH_OUTER = [61,146,91,181,84,17,314,405,321,375,291]
MOUTH_INNER = [78,95,88,178,87,14,317,402,318,324,308]
LEFT_EYE    = [33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173]
RIGHT_EYE   = [263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398]

# ---------- utils ----------
def db_load():
    if DB_PATH.exists():
        try:
            return json.loads(DB_PATH.read_text("utf-8"))
        except:
            return {}
    return {}

def db_save(db):
    DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")

def normalize(v):
    v = np.asarray(v, np.float32).reshape(-1)
    n = np.linalg.norm(v) + 1e-8
    return v / n

def cosine(a, b):
    a = normalize(a); b = normalize(b)
    return float(np.dot(a, b))

def best2_cosine(emb, db):
    """Retorna (id_top1, sim_top1, sim_top2) com base em cosine."""
    best_id, best, second = None, -1.0, -1.0
    for pid, vec in db.items():
        s = cosine(emb, vec)
        if s > best:
            second = best
            best = s
            best_id = pid
        elif s > second:
            second = s
    return best_id, best, second

def put_text(img, x, y, txt, scale=0.45, col=(255,255,255), bg=(0,0,0), pad=6):
    """Desenha o texto (com caixinha) e retorna o próximo y com espaçamento pad."""
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    yy = max(h + 6, int(y))
    if bg is not None:
        cv2.rectangle(img, (x-2, yy-h-6), (x+w+2, yy+4), bg, -1)
    cv2.putText(img, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return yy + h + pad

def draw_progress(frame, p, text="Coletando amostras..."):
    p = float(np.clip(p, 0, 1))
    h, w = frame.shape[:2]
    bw, bh = min(560, w-40), 24
    x, y = (w-bw)//2, h - 24 - 20
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (40,40,40), -1)
    fill = int(bw * p)
    cv2.rectangle(frame, (x, y), (x+fill, y+bh), (0,170,0), -1)
    put_text(frame, x+8, y-6, f"{text} ({int(p*100)}%)", 0.45, (255,255,255))

def parse_yunet_landmarks(row):
    r = row.astype(float).tolist()
    if len(r) < 15:
        x, y, w, h = map(int, r[:4]); return (x, y, int(w), int(h)), {}
    if 0 < r[4] <= 1.0:
        x,y,w,h = map(int, r[:4]); pts = r[5:15]
    else:
        x,y,w,h = map(int, r[:4]); pts = r[4:14]
    try:
        re = (int(pts[0]), int(pts[1])); le = (int(pts[2]), int(pts[3]))
        no = (int(pts[4]), int(pts[5])); rm = (int(pts[6]), int(pts[7])); lm = (int(pts[8]), int(pts[9]))
        landmarks = {"re": re, "le": le, "nose": no, "rm": rm, "lm": lm}
    except Exception:
        landmarks = {}
    return (x,y,int(w),int(h)), landmarks

def measures_from_landmarks(land, box):
    if not land or not {"re","le","rm","lm"} <= set(land.keys()):
        return None
    io = float(np.linalg.norm(np.array(land["re"]) - np.array(land["le"])))
    mw = float(np.linalg.norm(np.array(land["rm"]) - np.array(land["lm"])))
    ratio = (mw / io) if io > 1e-6 else 0.0
    x,y,w,h = box
    return {"io": io, "mw": mw, "ratio": ratio, "h": float(h), "w": float(w)}

def draw_basic_guides(frame, land):
    if not land: return
    for p in ("re","le"): cv2.circle(frame, land[p], 3, (0,255,255), -1)
    cv2.line(frame, land["re"], land["le"], (255,255,0), 2)
    if "rm" in land and "lm" in land:
        cv2.circle(frame, land["rm"], 3, (255,0,255), -1)
        cv2.circle(frame, land["lm"], 3, (255,0,255), -1)
        cv2.line(frame, land["rm"], land["lm"], (0,255,0), 2)

def draw_mesh_points(frame, pts, idxs, color):
    for k in idxs:
        p = tuple(np.round(pts[k]).astype(int))
        cv2.circle(frame, p, 1, color, -1)

def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def make_tracker():
    creators = [
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerMOSSE_create(),
        lambda: cv2.TrackerMOSSE_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT_create(),
    ]
    for c in creators:
        try: return c()
        except Exception: continue
    return None

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    sa, sb = aw*ah, bw*bh
    union = sa + sb - inter + 1e-6
    return inter/union

# ===== cache simples da API =====
_api_cache = {}  # key (face_id/ra) -> (timestamp, payload)

def get_status_cached(person_key):
    if not person_key: return None
    t = time.time()
    if person_key in _api_cache and (t - _api_cache[person_key][0]) < API_TTL:
        return _api_cache[person_key][1]
    if requests is None:
        return None
    try:
        r = requests.get(f"{API_BASE}/api/student/{person_key}", timeout=0.8)
        if r.ok:
            payload = r.json()
            _api_cache[person_key] = (t, payload)
            return payload
    except Exception:
        pass
    return None

# ===== FaceID helpers =====
def make_face_id(length: int = 4) -> str:   
    return uuid.uuid4().hex[:length].upper()


def post_face_id_to_app(face_id: str):
    if requests is None: return
    try:
        requests.post(f"{API_BASE}/api/face-id", json={"face_id": face_id}, timeout=1.0)
    except Exception:
        pass

def open_student_form(face_id: str):
    try:
        webbrowser.open(f"{API_BASE}/students/new?face_id={face_id}")
    except Exception:
        pass

def face_id_confirmed(face_id: str) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"{API_BASE}/api/face-id/status/{face_id}", timeout=1.0)
        if r.ok:
            data = r.json()
            return bool(data.get("confirmed"))
    except Exception:
        pass
    return False

def wait_confirm_face_id(face_id: str, max_wait=CONFIRM_WAIT_MAX, step=CONFIRM_WAIT_STEP) -> bool:
    t0 = time.time()
    while (time.time() - t0) < max_wait:
        if face_id_confirmed(face_id):
            return True
        time.sleep(step)
    return False

# ---------- modelos e câmera ----------
det = cv2.FaceDetectorYN_create(DET_MODEL, "", (FRAME_W, FRAME_H))
rec = cv2.FaceRecognizerSF_create(REC_MODEL, "")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("[Erro] Câmera não abriu")

db = db_load()
for k, v in list(db.items()):
    db[k] = normalize(v).tolist()

# ---------- estados ----------
fid=0; frames=0; t0=time.time(); fps=0.0
last_faces=[]; mesh_faces=[]
tracks=[]; next_tid=1

# cadastro no loop principal
# enroll = {"id", "samples", "t0", "deadline", "target_box"}
enroll = None

# ---------- loop ----------
while True:
    ok, frame = cap.read()
    if not ok: break
    frames += 1; fid += 1
    if time.time()-t0 >= 1.0:
        fps = frames/(time.time()-t0); frames=0; t0=time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FaceMesh opcional
    if MP_OK and fid % MESH_EVERY == 0:
        mesh_faces=[]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                pts = np.array([(p.x*FRAME_W, p.y*FRAME_H) for p in lm.landmark], dtype=np.float32)
                mesh_faces.append({"pts": pts, "center": pts.mean(axis=0)})

    # Detecção/embedding periódico
    if fid % DETECT_EVERY == 0:
        det.setInputSize((frame.shape[1], frame.shape[0]))
        out = det.detect(frame)
        faces = out[1] if out is not None else None

        last_faces=[]
        if faces is not None:
            for i, row in enumerate(faces):
                box, land = parse_yunet_landmarks(row)
                aligned = rec.alignCrop(frame, row)
                emb = normalize(rec.feature(aligned))
                meas = measures_from_landmarks(land, box)
                best_name, best_sim, second_sim = best2_cosine(emb, db)
                gap = best_sim - max(second_sim, -1.0)
                is_hit = (best_sim >= COS_THR) and (gap >= GAP_THR)
                last_faces.append({
                    "idx": i, "box": tuple(map(int, box)), "row": row,
                    "land": land, "meas": meas,
                    "emb": emb, "sim": best_sim, "gap": gap,
                    "match": (best_name if is_hit else None),
                    "is_hit": is_hit
                })

        # atualiza tracks por IoU
        new_tracks=[]; used=set()
        for tr in tracks:
            best_j, best_iou = -1, 0.0
            for j, f in enumerate(last_faces):
                if j in used: continue
                iou_val = iou(tr["box"], f["box"])
                if iou_val > best_iou:
                    best_j, best_iou = j, iou_val
            if best_j >= 0 and best_iou > 0.1:
                f = last_faces[best_j]; used.add(best_j)
                tr["box"] = f["box"]
                if f.get("is_hit"):
                    tr["vote"].append(f["match"])
                if tr["tracker"] is None:
                    trk = make_tracker()
                    if trk is not None: trk.init(frame, (*f["box"],)); tr["tracker"]=trk
                else:
                    try: tr["tracker"].clear()
                    except: pass
                    trk = make_tracker()
                    if trk is not None: trk.init(frame, (*f["box"],)); tr["tracker"]=trk
                new_tracks.append(tr)
        tracks = new_tracks

        for j, f in enumerate(last_faces):
            if j in used: continue
            trk = make_tracker()
            if trk is not None: trk.init(frame, (*f["box"],))
            tracks.append({"tid": next_tid, "tracker": trk, "box": f["box"],
                           "vote": deque(maxlen=VOTE_SIZE), "name": None})
            if f.get("is_hit"):
                tracks[-1]["vote"].append(f["match"])
            next_tid += 1
    else:
        # update de trackers entre detecções
        for tr in tracks:
            if tr["tracker"] is None: continue
            ok2, box = tr["tracker"].update(frame)
            if ok2:
                x,y,w,h = [int(v) for v in box]
                tr["box"] = (x,y,w,h)

    # resolve nome por voto (estável; não zera sem voto)
    for tr in tracks:
        prev = tr.get("name")
        if tr["vote"]:
            cand, cnt = Counter(tr["vote"]).most_common(1)[0]
            tr["name"] = cand if cnt >= VOTE_MIN else prev
        else:
            tr["name"] = prev

    # ---- CADASTRO (estado no loop) ----
    if enroll is not None:
        # coleta do alvo
        target = None
        if last_faces:
            if enroll["target_box"] is not None:
                best, biou = None, 0.0
                for f in last_faces:
                    iou_val = iou(enroll["target_box"], f["box"])
                    if iou_val > biou:
                        best, biou = f, iou_val
                target = best
            if target is None:
                target = max(last_faces, key=lambda f: f["box"][2]*f["box"][3])

        reason = "aproxime o rosto e olhe de frente"
        if target:
            x,y,w,h = target["box"]
            land = target["land"]; meas = target["meas"]
            blur = variance_of_laplacian(gray[y:y+h, x:x+w]) if w>0 and h>0 else 0.0
            ok_face = min(w,h) >= MIN_FACE
            ok_io   = (meas and meas["io"] >= MIN_IO)
            ok_blur = blur >= MIN_BLUR
            if ok_face and ok_io and ok_blur:
                enroll["samples"].append(target["emb"])
                enroll["target_box"] = target["box"]
                reason = f"amostra aceita ({len(enroll['samples'])}/{ENROLL_SAMPLES})"
                col=COL_GREEN
            else:
                if not ok_face: reason="Face pequena - aproxime"
                elif not ok_io: reason="Olhos pouco visíveis - olhe de frente"
                elif not ok_blur: reason="Desfocado - fique parado / melhore a luz"
                col=COL_RED
            cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)

        p = len(enroll["samples"])/ENROLL_SAMPLES
        draw_progress(frame, p, text=f"Cadastrando '{enroll['id']}'")
        put_text(frame, 10, 70, f"Status: {reason}", 0.6)

        # finalização — salva só se o app confirmou (nome+RA)
        timeout = (time.time() - enroll["t0"]) > ENROLL_TIMEOUT
        finished = len(enroll["samples"]) >= ENROLL_SAMPLES or timeout
        if finished:
            if enroll["samples"]:
                ref = normalize(np.median(np.stack(enroll["samples"],0), axis=0))
                confirmed = face_id_confirmed(enroll["id"])
                if not confirmed:
                    print(f"[Enroll] '{enroll['id']}' aguardando confirmação no app...")
                    confirmed = wait_confirm_face_id(enroll["id"])
                if confirmed:
                    db[enroll["id"]] = ref.tolist(); db_save(db)
                    print(f"[Enroll] '{enroll['id']}' salvo (confirmado no app).")
                else:
                    print(f"[Enroll] '{enroll['id']}' não salvo (cancelado ou não confirmado no app).")
            else:
                print("[Enroll] Cancelado/sem amostras")
            enroll = None
            # importantíssimo: evita qualquer acesso ao 'enroll' nesse frame
            cv2.imshow("FaceMap PRO", frame)
            continue  # <-- depois que finaliza, vai direto ao próximo frame

    # ---- desenhar tracks/faces ----
    for tr in tracks:
        x,y,w,h = tr["box"]

        status = get_status_cached(tr["name"]) if tr["name"] else None

        rect_color = COL_RED
        if tr["name"]:
            if status and status.get("found"):
                if status.get("overdue_count", 0) > 0:
                    rect_color = COL_RED
                elif status.get("due_soon_count", 0) > 0:
                    rect_color = COL_ORANGE
                else:
                    rect_color = COL_GREEN
            else:
                rect_color = COL_GREEN

        cv2.rectangle(frame, (x,y), (x+w, y+h), rect_color, 2)

        fbest, biou = None, 0.0
        for f in last_faces:
            iou_val = iou((x,y,w,h), f["box"])
            if iou_val > biou:
                biou, fbest = iou_val, f

        label = tr["name"] if tr["name"] else "?"
        if fbest:
            put_text(frame, x, y-8, f"[{fbest['idx']}] {label} (sim:{fbest['sim']:.2f} gap:{fbest['gap']:.2f})")

            # Lado: nome + pendência
            first_name = None
            pend_text  = None
            if status and status.get("found") and status.get("person"):
                person = status["person"]
                full = person.get("name") or person.get("full_name") or ""
                first_name = (full.split()[0] if full else None)

                if status.get("overdue_count", 0) > 0:
                    pend_text = f"Atraso: {status['overdue_count']}"
                elif status.get("due_soon_count", 0) > 0:
                    pend_text = f"Vence logo: {status['due_soon_count']}"
                else:
                    pend_text = "Sem pendencias"

            H, W = frame.shape[:2]
            side_margin = 6
            prefer_right_x = x + w + side_margin
            use_right = (prefer_right_x < W - 160)
            side_x = prefer_right_x if use_right else max(6, x - 160)
            side_y = y + 16

            if first_name:
                side_y = put_text(frame, side_x, side_y, first_name, 0.50, (255,255,255), pad=8)
            if pend_text:
                side_y = put_text(frame, side_x, side_y, pend_text, 0.48, rect_color, pad=10)

            # Base: vetores/medidas com gap maior
            ycur = y + h + 18
            prev = ", ".join(f"{v:+.2f}" for v in fbest["emb"][:8])
            ycur = put_text(frame, x, ycur, f"Emb[:8]: {prev}", pad=10)
            ycur = put_text(frame, x, ycur, f"||emb||2 = {np.linalg.norm(fbest['emb']):.2f}", pad=10)
            if fbest.get("meas"):
                ycur = put_text(
                    frame, x, ycur,
                    f"Olhos: {fbest['meas']['io']:.1f}px  Boca: {fbest['meas']['mw']:.1f}px  Razao: {fbest['meas']['ratio']:.2f}",
                    0.44, pad=10
                )

            if DRAW_LANDMARKS:
                draw_basic_guides(frame, fbest["land"])
                if MP_OK and mesh_faces:
                    cx, cy = x+w/2.0, y+h/2.0
                    j = int(np.argmin([np.linalg.norm(m["center"]-np.array([cx,cy])) for m in mesh_faces]))
                    pts = mesh_faces[j]["pts"]
                    draw_mesh_points(frame, pts, LEFT_EYE,  (0,255,255))
                    draw_mesh_points(frame, pts, RIGHT_EYE, (0,255,255))
                    draw_mesh_points(frame, pts, MOUTH_OUTER, (255,0,255))
                    draw_mesh_points(frame, pts, MOUTH_INNER, (0,255,0))
        else:
            put_text(frame, x, y-8, label)

    put_text(frame, 10, 20, f"FPS {fps:.1f} | DB: {len(db)} | det={DETECT_EVERY} mesh={MESH_EVERY}")
    put_text(frame, 10, 40, "0-9: cadastrar | ESC/Q: sair")

    # ================= teclado =================
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')): break

    # Início do cadastro ao apertar 0–9
    if ord('0') <= k <= ord('9') and enroll is None:
        idx = k - ord('0')
        cand = None
        for f in last_faces:
            if f["idx"] == idx:
                cand = f; break
        if cand is not None:
            face_id = make_face_id()
            post_face_id_to_app(face_id)
            open_student_form(face_id)
            enroll = {
                "id": face_id,
                "samples": [],
                "t0": time.time(),
                "deadline": time.time() + ENROLL_TIMEOUT,
                "target_box": cand["box"]
            }

    cv2.imshow("FaceMap PRO", frame)

cap.release()
cv2.destroyAllWindows()

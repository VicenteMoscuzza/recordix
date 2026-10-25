import hmac
import json
import os
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from zoneinfo import ZoneInfo

import requests
from openai import OpenAI

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USER_IDS = {int(x) for x in os.environ["ALLOWED_USER_ID"].split(",") if x.strip()}
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
CRON_SECRET = os.environ["CRON_SECRET"]
TZ = ZoneInfo(os.getenv("TZ_NAME", "America/Argentina/Buenos_Aires"))
MODEL = os.environ["LLM_MODEL"]
def env_ending(*suffixes):
    for k, v in os.environ.items():
        if k.endswith(suffixes) and v:
            return v
    raise KeyError(f"Falta una variable de entorno terminada en {suffixes}")


REDIS_URL = env_ending("REST_API_URL", "REDIS_REST_URL")
REDIS_TOKEN = env_ending("REST_API_TOKEN", "REDIS_REST_TOKEN")
MAX_HISTORY = 20

llm = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crear_recordatorio",
            "description": "Crea un recordatorio que se enviará por Telegram a la hora indicada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texto": {"type": "string", "description": "Qué recordar."},
                    "fecha_hora": {
                        "type": "string",
                        "description": "Fecha y hora local en ISO 8601, ej. 2026-10-25T09:00:00",
                    },
                },
                "required": ["texto", "fecha_hora"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_recordatorios",
            "description": "Lista los recordatorios pendientes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "borrar_recordatorio",
            "description": "Borra un recordatorio pendiente por su id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    },
]


# ---------- Redis (Upstash REST) ----------

def redis(*cmd):
    r = requests.post(
        REDIS_URL,
        headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
        json=list(cmd),
        timeout=8,
    )
    r.raise_for_status()
    return r.json().get("result")


def pending():
    """Devuelve [(dict, score, member_str)] de todos los recordatorios pendientes."""
    flat = redis("ZRANGE", "reminders", 0, -1, "WITHSCORES") or []
    return [(json.loads(flat[i]), float(flat[i + 1]), flat[i]) for i in range(0, len(flat), 2)]


# ---------- Telegram ----------

def send_message(chat_id, text):
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    r.raise_for_status()


# ---------- Herramientas ----------

def run_tool(name, args, chat_id):
    if name == "crear_recordatorio":
        try:
            due = datetime.fromisoformat(args["fecha_hora"])
        except ValueError:
            return "Error: fecha_hora inválida, usá ISO 8601."
        if due.tzinfo is None:
            due = due.replace(tzinfo=TZ)
        if due <= datetime.now(TZ):
            return "Error: la fecha ya pasó."
        rid = redis("INCR", "reminder:id")
        member = json.dumps({"id": rid, "chat_id": chat_id, "text": args["texto"]}, ensure_ascii=False)
        redis("ZADD", "reminders", int(due.timestamp()), member)
        return f"Creado id={rid} para {due.astimezone(TZ):%Y-%m-%d %H:%M}"
    if name == "listar_recordatorios":
        rows = [(d, s) for d, s, _ in pending() if d["chat_id"] == chat_id]
        if not rows:
            return "No hay recordatorios pendientes."
        return "\n".join(
            f"{d['id']}: {d['text']} ({datetime.fromtimestamp(s, TZ):%Y-%m-%d %H:%M})"
            for d, s in rows
        )
    if name == "borrar_recordatorio":
        for d, _, member in pending():
            if d["id"] == args["id"] and d["chat_id"] == chat_id:
                redis("ZREM", "reminders", member)
                return "Borrado."
        return "No existe ese id."
    return "Herramienta desconocida."


# ---------- LLM ----------

def ask_llm(chat_id, user_text):
    now = datetime.now(TZ)
    system = (
        "Sos un asistente personal por Telegram. Respondé corto y directo, en español rioplatense. "
        f"Fecha y hora actual: {now:%A %Y-%m-%d %H:%M} ({TZ.key}). "
        "Para recordatorios usá las herramientas y calculá fechas relativas a partir de la hora actual. "
        "Confirmá siempre el día y la hora exactos que programaste."
    )
    hist_key = f"hist:{chat_id}"
    raw = redis("GET", hist_key)
    hist = json.loads(raw) if raw else []

    messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": user_text}]
    final = ""
    for _ in range(6):
        resp = llm.chat.completions.create(
            model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            break
        messages.append(msg)
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": run_tool(call.function.name, args, chat_id),
                }
            )
    final = final or "No pude completar eso."

    hist += [{"role": "user", "content": user_text}, {"role": "assistant", "content": final}]
    redis("SET", hist_key, json.dumps(hist[-MAX_HISTORY:], ensure_ascii=False), "EX", 604800)
    return final


# ---------- Lógica de cada endpoint ----------

def handle_update(update):
    msg = update.get("message")
    if not msg or "text" not in msg:
        return
    if msg["from"]["id"] not in ALLOWED_USER_IDS:
        return
    # Telegram reintenta si tardamos; descartamos updates repetidos.
    if redis("SET", f"upd:{update['update_id']}", "1", "NX", "EX", 600) is None:
        return
    chat_id = msg["chat"]["id"]
    try:
        reply = ask_llm(chat_id, msg["text"])
    except Exception:
        traceback.print_exc()
        reply = "Error interno, mirá los logs en Vercel."
    send_message(chat_id, reply)


def send_due():
    now = int(time.time())
    members = redis("ZRANGEBYSCORE", "reminders", "-inf", now) or []
    sent = 0
    for m in members:
        if not redis("ZREM", "reminders", m):  # otro disparo ya lo tomó
            continue
        r = json.loads(m)
        try:
            send_message(r["chat_id"], f"⏰ {r['text']}")
            sent += 1
        except Exception:
            traceback.print_exc()
            redis("ZADD", "reminders", now, m)  # se reintenta en el próximo minuto
    return sent


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


class handler(BaseHTTPRequestHandler):
    def _reply(self, code, body="ok"):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode())

    # POST = webhook de Telegram
    def do_POST(self):
        if not same(self.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), WEBHOOK_SECRET):
            return self._reply(401, "unauthorized")
        try:
            length = int(self.headers.get("Content-Length", 0))
            update = json.loads(self.rfile.read(length))
        except Exception:
            return self._reply(400, "bad request")
        try:
            handle_update(update)
        except Exception:
            traceback.print_exc()
        self._reply(200)

    # GET = disparo del scheduler externo (envía los recordatorios vencidos)
    def do_GET(self):
        if not same(self.headers.get("Authorization", ""), f"Bearer {CRON_SECRET}"):
            return self._reply(401, "unauthorized")
        try:
            n = send_due()
        except Exception:
            traceback.print_exc()
            return self._reply(500, "error")
        self._reply(200, f"sent {n}")

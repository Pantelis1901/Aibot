import os
import uuid
import threading
import requests
from pathlib import Path
from flask import Flask, request, send_from_directory, jsonify
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

# ---------------- ENV ----------------
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
PORT = int(os.getenv("PORT", "3000"))

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------- APP ----------------
app = Flask(__name__)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

CONVERSATIONS = {}            # session memory
TTS_CACHE = {}                # NEW: TTS caching
TWILIO_HOLD_MUSIC = "http://com.twilio.music.classical.s3.amazonaws.com/BusyStrings.mp3"


# ---------------- HELPERS ----------------
def is_greek(text: str) -> bool:
    for ch in text:
        if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF":
            return True
    return False


# ---------------- GPT AGENT ----------------
def gpt_reply(call_sid: str, user_text: str) -> str:
    if call_sid not in CONVERSATIONS:
        CONVERSATIONS[call_sid] = [
            {
                "role": "system",
                "content": (
                    "Είσαι επαγγελματική τηλεφωνήτρια στην Ψησταριά της Βούλας στη Σπάρτη.\n"
                    "Μιλάς καθαρά, σύντομα και ευγενικά. Στυλ: φυσικό & ανθρώπινο.\n"
                    "Δεν κάνεις μεγάλα τετράστιχα. Μικρές, καθαρές απαντήσεις.\n"
                    "Στόχος:\n"
                    "- Καταγραφή παραγγελίας\n"
                    "- Ερωτήσεις για διευκρίνιση\n"
                    "- Επιβεβαίωση στο τέλος\n"
                    "Αν ρωτήσει 'τι έχει το μενού', λες κατηγορίες, όχι full λίστα.\n"
                )
            }
        ]

    conv = CONVERSATIONS[call_sid]
    conv.append({"role": "user", "content": user_text})

    payload = {
        "model": "gpt-4o-mini",
        "messages": conv,
        "temperature": 0.3,
        "max_tokens": 200,
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=payload,
            timeout=20,
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ GPT Error:", e)
        reply = "Μπορείτε να το επαναλάβετε; Δεν άκουσα καθαρά."

    conv.append({"role": "assistant", "content": reply})

    if len(conv) > 20:
        CONVERSATIONS[call_sid] = [conv[0]] + conv[-19:]

    return reply


# ---------------- TTS WITH CACHING ----------------
def tts_audio(text: str, label: str) -> str:
    """
    PRODUCES a guaranteed VALID MP3.
    - Caches every TTS response (massive speed boost)
    - Retries if the mp3 from OpenAI is too small (<500 bytes)
    - Falls back to <Say> if still bad.
    """

    # ---- 1) CHECK CACHE FIRST ----
    if text in TTS_CACHE:
        return TTS_CACHE[text]

    file_id = uuid.uuid4().hex
    path = AUDIO_DIR / f"{label}_{file_id}.mp3"

    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",
        "input": text,
        "format": "mp3",
    }

    def generate_once():
        try:
            r = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=25,
            )
            r.raise_for_status()
            return r.content
        except Exception as e:
            print("❌ TTS request error:", e)
            return b""

    # ---- first attempt ----
    audio_bytes = generate_once()

    # ---- retry if too small ----
    if len(audio_bytes) < 500:
        print("⚠️ TTS too small → retrying")
        audio_bytes = generate_once()

    # ---- fallback ----
    if len(audio_bytes) < 500:
        print("❌ TTS failed twice → fallback")
        return ""

    # ---- SAVE MP3 ----
    with open(path, "wb") as f:
        f.write(audio_bytes)

    final_url = f"{BASE_URL}/audio/{path.name}"

    # ---- SAVE TO CACHE ----
    TTS_CACHE[text] = final_url

    return final_url


# ---------------- DEEPGRAM STT ----------------
def deepgram_stt(audio_bytes: bytes) -> str:
    url = "https://api.deepgram.com/v1/listen?model=nova-3&language=el"
    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            data=audio_bytes,
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except Exception as e:
        print("❌ Deepgram STT error:", e)
        return ""


# ---------------- BACKGROUND PROCESS ----------------
def background_process(call_sid: str, recording_url: str):
    try:
        wav = recording_url + ".wav"
        print("🎧 Downloading:", wav)

        r = requests.get(
            wav,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30,
        )
        r.raise_for_status()
        audio_bytes = r.content

        # STT
        transcript = deepgram_stt(audio_bytes)
        print("🗣 USER:", transcript)

        if not transcript:
            bot_text = "Μπορείτε να το πείτε λίγο πιο καθαρά;"
        elif not is_greek(transcript):
            bot_text = "Μιλήστε μου στα ελληνικά για να σας εξυπηρετήσω."
        else:
            bot_text = gpt_reply(call_sid, transcript)

        print("🤖 BOT:", bot_text)

        # TTS
        audio_url = tts_audio(bot_text, call_sid)

        # Twilio update
        if audio_url:
            twiml = f"""
<Response>
    <Play>{audio_url}</Play>
    <Record action="/twilio/process" playBeep="false" timeout="6" maxLength="15" />
</Response>
"""
        else:
            twiml = f"""
<Response>
    <Say>{bot_text}</Say>
    <Record action="/twilio/process" playBeep="false" timeout="6" maxLength="15" />
</Response>
"""

        try:
            twilio_client.calls(call_sid).update(twiml=twiml)
            print("✅ Call updated.")
        except Exception as e:
            print("❌ Twilio update error:", e)

    except Exception as e:
        print("❌ BACKGROUND ERROR:", e)


# ---------------- ROUTES ----------------

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    resp = VoiceResponse()

    text = (
        "Καλησπέρα σας! Καλέσατε την Ψησταριά της Βούλας. "
        "Μιλάτε αφού τελειώσω, για να σας ακούω καθαρά. "
        "Τι θα θέλατε;"
    )
    intro = tts_audio(text, "intro")

    if intro:
        resp.play(intro)
    else:
        resp.say(text)

    resp.record(
        action="/twilio/process",
        playBeep=False,
        timeout=6,
        maxLength=15
    )

    return str(resp)


@app.route("/twilio/process", methods=["POST"])
def twilio_process():
    call_sid = request.form.get("CallSid")
    rec_url = request.form.get("RecordingUrl")
    print("📥 /twilio/process:", rec_url)

    threading.Thread(
        target=background_process,
        args=(call_sid, rec_url),
        daemon=True
    ).start()

    resp = VoiceResponse()
    resp.say("Ένα δευτερόλεπτο να ετοιμάσω την απάντηση.")
    resp.play(TWILIO_HOLD_MUSIC)
    return str(resp)


@app.route("/call-me")
def call_me():
    to = request.args.get("to")
    try:
        call = twilio_client.calls.create(
            to=to,
            from_=TWILIO_NUMBER,
            url=f"{BASE_URL}/twilio/voice"
        )
        return jsonify({"sid": call.sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)

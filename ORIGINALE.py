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

BASE_URL = os.getenv("BASE_URL")
PORT = int(os.getenv("PORT", 3000))

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

CONVERSATIONS = {}

# ---------------- ΒΟΗΘΗΤΙΚΟ ----------------
def is_greek(text):
    for ch in text:
        if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF":
            return True
    return False


# ---------------- OPENAI GPT ----------------
def gpt_reply(call_sid, user_text):

    if call_sid not in CONVERSATIONS:
        CONVERSATIONS[call_sid] = [{
            "role": "system",
            "content": (
                "Είσαι ευγενική και επαγγελματική τηλεφωνήτρια "
                "στην 'Ψησταριά της Βούλας'. "
                "Μιλάς φυσικά, σύντομα και καθαρά. "
                "Στόχος: να πάρεις σωστά την παραγγελία."
            )
        }]

    conv = CONVERSATIONS[call_sid]
    conv.append({"role": "user", "content": user_text})

    payload = {
        "model": "gpt-4o-mini",
        "messages": conv,
        "temperature": 0.3,
        "max_tokens": 250,
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json=payload
    )
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]

    conv.append({"role": "assistant", "content": reply})

    return reply


# ---------------- OPENAI TTS ----------------
def tts_audio(text, label):
    file_id = uuid.uuid4().hex
    path = AUDIO_DIR / f"{label}_{file_id}.mp3"

    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",
        "input": text,
        "format": "mp3"
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Accept": "audio/mpeg",
            "Content-Type": "application/json"
        },
        json=payload
    )
    r.raise_for_status()

    with open(path, "wb") as f:
        f.write(r.content)

    return f"{BASE_URL}/audio/{path.name}"


# ---------------- DEEPGRAM STT ----------------
def deepgram_stt(audio_bytes):
    url = "https://api.deepgram.com/v1/listen?model=nova-3&language=el"

    r = requests.post(
        url,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        },
        data=audio_bytes,
        timeout=20
    )
    r.raise_for_status()

    data = r.json()
    return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


# ---------------- BACKGROUND WORKER ----------------
def handle_background_logic(call_sid, recording_url):
    try:
        # 1) Download WAV
        wav_url = recording_url + ".wav"
        audio = requests.get(
            wav_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        ).content

        # 2) STT
        transcript = deepgram_stt(audio)
        print("🗣 USER:", transcript)

        if not transcript or not is_greek(transcript):
            bot_reply = "Δεν σας άκουσα καθαρά, μπορείτε να το επαναλάβετε;"
        else:
            bot_reply = gpt_reply(call_sid, transcript)

        print("🤖 BOT:", bot_reply)

        # 3) TTS
        audio_url = tts_audio(bot_reply, call_sid)

        # 4) Play URL to active call
        client.calls(call_sid).update(
            twiml=f"<Response><Play>{audio_url}</Play><Record action='/twilio/process' playBeep='false' timeout='6' maxLength='15' /></Response>"
        )

    except Exception as e:
        print("BACKGROUND ERROR:", e)


# ---------------- AUDIO SERVE ----------------
@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# ---------------- START CALL ----------------
@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    resp = VoiceResponse()

    intro = (
        "Καλησπέρα σας! Καλέσατε την Ψησταριά της Βούλας. "
        "Μιλάτε μόνο όταν τελειώσω, για να σας ακούω καθαρά. "
        "Τι θα θέλατε να παραγγείλετε;"
    )
    intro_url = tts_audio(intro, "intro")
    resp.play(intro_url)

    resp.record(
        action="/twilio/process",
        playBeep=False,
        timeout=6,
        maxLength=15
    )

    return str(resp)


# ---------------- PROCESS RECORDING (FAST RESPONSE!) ----------------
@app.route("/twilio/process", methods=["POST"])
def twilio_process():
    call_sid = request.form.get("CallSid")
    rec_url = request.form.get("RecordingUrl")

    # LAUNCH BACKGROUND THREAD
    threading.Thread(
        target=handle_background_logic,
        args=(call_sid, rec_url),
        daemon=True
    ).start()

    # RETURN FAST RESPONSE TO TWILIO (NO TIMEOUT EVER)
    resp = VoiceResponse()
    resp.say("Ένα λεπτό…", voice="alice")
    return str(resp)


# ---------------- OUTBOUND CALL ----------------
@app.route("/call-me")
def call_me():
    to = request.args.get("to")

    call = client.calls.create(
        to=to,
        from_=TWILIO_NUMBER,
        url=f"{BASE_URL}/twilio/voice"
    )

    return {"status": "calling", "sid": call.sid}


if __name__ == "__main__":
    print(f"Running on {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)

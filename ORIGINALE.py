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

if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_NUMBER and OPENAI_API_KEY and DEEPGRAM_API_KEY and BASE_URL):
    print("⚠️ Missing one or more required environment variables.")

# Twilio REST client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------- APP / AUDIO DIR ----------------
app = Flask(__name__)
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

# very simple in-memory conversation per call
CONVERSATIONS = {}

# Twilio demo hold music (κρατάει την κλήση ανοιχτή όσο σκέφτεται ο agent)
TWILIO_HOLD_MUSIC = "http://com.twilio.music.classical.s3.amazonaws.com/BusyStrings.mp3"


# ---------------- HELPERS ----------------
def is_greek(text: str) -> bool:
    for ch in text:
        if "\u0370" <= ch <= "\u03FF" or "\u1F00" <= ch <= "\u1FFF":
            return True
    return False


# ---------------- GPT AGENT ----------------
def gpt_reply(call_sid: str, user_text: str) -> str:
    """
    Επαγγελματική τηλεφωνήτρια στην 'Ψησταριά της Βούλας'.
    Κρατάμε context ανά CallSid.
    """

    if call_sid not in CONVERSATIONS:
        CONVERSATIONS[call_sid] = [
            {
                "role": "system",
                "content": (
                    "Είσαι επαγγελματική, ευγενική και σύντομη τηλεφωνήτρια "
                    "στην 'Ψησταριά της Βούλας' στη Σπάρτη.\n"
                    "Μιλάς φυσικά, σε δεύτερο πρόσωπο (πχ. 'να σας βάλω κάτι ακόμα;').\n"
                    "Στόχος σου είναι:\n"
                    "- Να καταλαβαίνεις αμέσως τι θέλει να παραγγείλει ο πελάτης.\n"
                    "- Να ρωτάς ξεκάθαρες διευκρινίσεις (πχ. τι κρέας, τι σως, πόσα τεμάχια).\n"
                    "- Να επιβεβαιώνεις στο τέλος την παραγγελία, καθαρά και οργανωμένα.\n"
                    "ΜΗΝ λες περιγραφές από e-food. Μίλα απλά, σαν άνθρωπος.\n"
                    "Αν ο πελάτης ρωτήσει 'τι έχει το μενού', πες συνοπτικά τις βασικές κατηγορίες:\n"
                    "τυλιχτά (γύρος, σουβλάκι), σκεπαστές, μερίδες, σαλάτες, ορεκτικά, burgers, αναψυκτικά.\n"
                )
            }
        ]

    conv = CONVERSATIONS[call_sid]
    conv.append({"role": "user", "content": user_text})

    payload = {
        "model": "gpt-4o-mini",
        "messages": conv,
        "temperature": 0.3,
        "max_tokens": 220,
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
        print("❌ OpenAI chat error:", e)
        reply = "Συγγνώμη, αντιμετωπίζω ένα τεχνικό πρόβλημα. Μπορείτε να επαναλάβετε λίγο πιο απλά;"

    conv.append({"role": "assistant", "content": reply})

    # μικρό trimming στο ιστορικό
    if len(conv) > 20:
        CONVERSATIONS[call_sid] = [conv[0]] + conv[-19:]

    return reply


# ---------------- TTS (OpenAI) ----------------
def tts_audio(text: str, label: str) -> str:
    """
    Δημιουργεί MP3 σε γυναικεία φωνή και επιστρέφει πλήρες URL για Twilio <Play>.
    """
    file_id = uuid.uuid4().hex
    path = AUDIO_DIR / f"{label}_{file_id}.mp3"

    payload = {
        "model": "gpt-4o-mini-tts",
        "voice": "coral",  # γυναικεία, καθαρή φωνή
        "input": text,
        "format": "mp3",
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return f"{BASE_URL}/audio/{path.name}"
    except Exception as e:
        print("❌ OpenAI TTS error:", e)
        # fallback: Twilio <Say>
        return ""


# ---------------- DEEPGRAM STT (Greek) ----------------
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
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        transcript = (
            data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        )
        return transcript
    except Exception as e:
        print("❌ Deepgram STT error:", e)
        return ""


# ---------------- BACKGROUND LOGIC ----------------
def background_process(call_sid: str, recording_url: str):
    """
    Τρέχει σε ξεχωριστό thread:
    - κατεβάζει το recording (wav)
    - κάνει STT στο Deepgram
    - ρίχνει το κείμενο στο GPT
    - κάνει TTS την απάντηση
    - ενημερώνει την ενεργή κλήση με νέο TwiML (Play + Record)
    """
    try:
        # 1) Download wav από Twilio
        wav_url = recording_url + ".wav"
        print(f"🎧 Downloading recording from {wav_url}")
        audio_resp = requests.get(
            wav_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30,
        )
        audio_resp.raise_for_status()
        audio_bytes = audio_resp.content

        # 2) STT (Deepgram)
        transcript = deepgram_stt(audio_bytes)
        print("🗣 USER:", transcript)

        if not transcript:
            bot_text = "Δεν σας άκουσα καθαρά. Μπορείτε να το επαναλάβετε λίγο πιο αργά;"
        elif not is_greek(transcript):
            bot_text = "Για να σας εξυπηρετήσω σωστά, μιλήστε μου στα ελληνικά, παρακαλώ."
        else:
            bot_text = gpt_reply(call_sid, transcript)

        print("🤖 BOT:", bot_text)

        # 3) TTS
        audio_url = tts_audio(bot_text, call_sid)

        # 4) Ενημέρωση ενεργής κλήσης
        if audio_url:
            twiml = f"""
<Response>
    <Play>{audio_url}</Play>
    <Record action="/twilio/process"
            playBeep="false"
            timeout="6"
            maxLength="15" />
</Response>
"""
        else:
            # fallback αν TTS απέτυχε
            twiml = """
<Response>
    <Say language="el-GR" voice="alice">
    Συγγνώμη, αντιμετωπίζω ένα τεχνικό θέμα με τον ήχο.
    Πείτε μου ξανά τι θα θέλατε να παραγγείλετε.
    </Say>
    <Record action="/twilio/process"
            playBeep="false"
            timeout="6"
            maxLength="15" />
</Response>
"""

        try:
            twilio_client.calls(call_sid).update(twiml=twiml)
            print("✅ Call updated with new TwiML.")
        except Exception as e:
            # Αν ο πελάτης έχει κλείσει, θα πάρουμε 400 εδώ. Δεν είναι κρίσιμο.
            print("❌ BACKGROUND UPDATE ERROR:", e)

    except Exception as e:
        print("❌ BACKGROUND FATAL ERROR:", e)


# ---------------- ROUTES ----------------
@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "message": "voice agent running"})


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


# ---- START OF CALL ----
@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    """
    Πρώτο entrypoint όταν χτυπάει το τηλέφωνο.
    Παίζουμε intro και ανοίγουμε Record.
    """
    resp = VoiceResponse()

    intro_text = (
        "Καλησπέρα σας! Καλέσατε την Ψησταριά της Βούλας. "
        "Μιλάτε αφού τελειώσω, για να σας ακούω καθαρά. "
        "Τι θα θέλατε να παραγγείλετε;"
    )
    intro_url = tts_audio(intro_text, "intro") or ""

    if intro_url:
        resp.play(intro_url)
    else:
        resp.say(
            "Καλησπέρα σας! Καλέσατε την Ψησταριά της Βούλας. "
            "Πείτε μου τι θα θέλατε να παραγγείλετε.",
            language="el-GR",
            voice="alice",
        )

    resp.record(
        action="/twilio/process",
        playBeep=False,
        timeout=6,   # σιωπή πριν σταματήσει το recording
        maxLength=15 # max διάρκεια ενός γύρου ομιλίας
    )

    return str(resp)


# ---- PROCESS RECORDING (ASYNC AGENT) ----
@app.route("/twilio/process", methods=["POST"])
def twilio_process():
    """
    Η Twilio μας στέλνει το recording.
    - Ξεκινάμε background thread για Deepgram+GPT+TTS
    - ΑΠΑΝΤΑΜΕ ΑΜΕΣΑ με TwiML (hold-music) για να ΜΗΝ κλείσει η κλήση
    """
    call_sid = request.form.get("CallSid")
    rec_url = request.form.get("RecordingUrl")

    print(f"📥 /twilio/process sid={call_sid} recording={rec_url}")

    if not call_sid or not rec_url:
        resp = VoiceResponse()
        resp.say(
            "Παρουσιάστηκε τεχνικό σφάλμα με την κλήση. Προσπαθήστε ξανά.",
            language="el-GR",
            voice="alice",
        )
        return str(resp)

    # Background επεξεργασία
    threading.Thread(
        target=background_process,
        args=(call_sid, rec_url),
        daemon=True,
    ).start()

    # ΑΜΕΣΗ απάντηση στην Twilio: μικρό μήνυμα + hold-music
    resp = VoiceResponse()
    resp.say(
        "Ένα δευτερόλεπτο να ετοιμάσω την παραγγελία σας.",
        language="el-GR",
        voice="alice",
    )
    # παίζουμε μουσική της Twilio ώστε η κλήση να παραμείνει ενεργή
    resp.play(TWILIO_HOLD_MUSIC)

    return str(resp)


# ---- OPTIONAL OUTBOUND HELPER ----
@app.route("/call-me", methods=["GET"])
def call_me():
    """
    Helper για να ξεκινάς κλήση από browser:
    /call-me?to=+3069xxxxxxx
    """
    to = request.args.get("to")
    if not to:
        return jsonify({"error": "missing 'to' parameter"}), 400

    try:
        call = twilio_client.calls.create(
            to=to,
            from_=TWILIO_NUMBER,
            url=f"{BASE_URL}/twilio/voice",
        )
        return jsonify({"status": "calling", "sid": call.sid})
    except Exception as e:
        print("❌ Error creating outbound call:", e)
        return jsonify({"error": "failed to create call"}), 500


if __name__ == "__main__":
    print(f"Running locally on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)

# improved_peinaleon.py
import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from flask import Flask, request, send_from_directory, jsonify
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import requests

# φορτώνουμε env
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

BASE_URL = os.getenv("BASE_URL", "http://localhost:3000").rstrip("/")
PORT = int(os.getenv("PORT", "3000"))

app = Flask(__name__)
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

# απλή μνήμη συνομιλιών στη RAM, ανά CallSid
CONVERSATIONS = {}

# --- βοηθητικά ---


def _mp3_path_for_text(call_sid: str, text: str) -> Path:
    """
    Δημιουργεί μοναδικό όνομα αρχείου βασισμένο σε hash του text,
    έτσι ώστε αν επαναλάβουμε την ίδια απάντηση δεν ξανακατεβάζουμε/ξαναγράφουμε.
    """
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return AUDIO_DIR / f"{call_sid}_{h}.mp3"


# --- OpenAI reply (συντομότερο/γρήγορο) ---


def openai_reply(call_sid: str, text: str) -> str:
    """
    Παίρνει το CallSid και το τελευταίο user μήνυμα,
    κρατάει ιστορικό και στέλνει ΟΛΗ τη συζήτηση στο OpenAI.
    Μειωμένα max_tokens & timeout για ταχύτερη απόκριση.
    """
    try:
        conv = CONVERSATIONS.get(call_sid)

        if not conv:
            conv = [
                {
                    "role": "system",
                    "content": (
                        "Είσαι τηλεφωνήτρια σε ελληνικό σουβλατζίδικο με όνομα Πειναλέων. "
                        "Μιλάς ΜΟΝΟ ελληνικά, σύντομα και φυσικά, σαν κανονικός υπάλληλος. "
                        "Στόχος σου είναι να πάρεις την παραγγελία, να ρωτήσεις ό,τι λείπει "
                        "και να την επιβεβαιώσεις, μαζί με διεύθυνση και τηλέφωνο.\n\n"
                        "- Στο ΠΡΩΤΟ μήνυμα της κλήσης μπορείς να πεις μια σύντομη χαιρετούρα.\n"
                        "- Στα επόμενα μηνύματα ΔΕΝ ξαναλες 'Καλησπέρα' χωρίς λόγο.\n"
                        "- Θυμάσαι τι έχει ήδη παραγγείλει ο πελάτης και τη διεύθυνση.\n"
                        "- Αν ο πελάτης πει ότι τελειώσαμε ή πει 'όχι, αυτά', 'ευχαριστώ', "
                        "'καλό βράδυ', κλείσε ευγενικά τη συνομιλία με μια σύντομη σύνοψη "
                        "της παραγγελίας και του χρόνου παράδοσης."
                    ),
                }
            ]
            CONVERSATIONS[call_sid] = conv

        conv.append({"role": "user", "content": text})

        # κρατάμε system + τελευταία 14 μηνύματα
        if len(conv) > 15:
            conv[:] = [conv[0]] + conv[-14:]

        payload = {
            "model": "gpt-4o-mini",
            "messages": conv,
            "temperature": 0.2,
            "max_tokens": 150,  # μειωμένο για ταχύτερες απαντήσεις
        }

        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,  # μικρότερο timeout
        )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"].strip()

        conv.append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        print("OpenAI error:", e)
        return (
            "Συγγνώμη, αντιμετωπίζω τεχνικό πρόβλημα αυτή τη στιγμή. "
            "Μπορείτε να καλέσετε λίγο αργότερα;"
        )


# --- ElevenLabs TTS με caching και μικρό timeout + fallback ---


def eleven_tts(text: str, call_sid: str, timeout_seconds: float = 3.0) -> Optional[str]:
    """
    Προσπαθεί να κατεβάσει mp3 από ElevenLabs μέσα σε timeout_seconds.
    Αν υπάρξει πρόβλημα ή αργήσει, επιστρέφει None (οπότε caller θα ακούσει resp.say()).
    Χρησιμοποιούμε caching με βάση hash του text & call_sid.
    """
    try:
        audio_path = _mp3_path_for_text(call_sid, text)

        # αν υπάρχει ήδη, επέστρεψε url αμέσως
        if audio_path.exists():
            return f"{BASE_URL}/audio/{audio_path.name}"

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }

        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}",
            headers={
                "xi-api-key": ELEVEN_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds,
        )
        r.raise_for_status()
        audio_bytes = r.content

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        return f"{BASE_URL}/audio/{audio_path.name}"

    except requests.Timeout:
        print("ElevenLabs timeout")
        return None
    except Exception as e:
        print("ElevenLabs error:", e)
        return None


# --- Flask routes ---


@app.route("/", methods=["GET"])
def index():
    return "Peinaleon AI Python bot running (improved)."


@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    """
    Ξεκινάει η συνομιλία με το bot, λέγοντας το μήνυμα υποδοχής
    και επιτρέπει την διακοπή (bargeIn=True) κατά την διάρκεια του TTS.
    """
    resp = VoiceResponse()
    
    # Gather πρώτου μηνύματος
    gather = Gather(
        input="speech",
        action="/twilio/process",
        language="el-GR",
        speech_timeout="auto",
        bargeIn=True,  # Επιτρέπει διακοπή κατά την διάρκεια του TTS
        timeout=5,  # Διάρκεια αναμονής για απάντηση
    )
    # Μήνυμα εισαγωγής
    gather.say(
        "Καλησπέρα σας, καλώς ήρθατε στο Πειναλέων. Τι θα θέλατε να παραγγείλετε;",
        language="el-GR",
    )
    
    # Προσθήκη του Gather στο VoiceResponse
    resp.append(gather)

    # Αν δεν υπάρχει απάντηση, θα επαναλάβει το Gather
    resp.redirect("/twilio/voice")
    
    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/twilio/process", methods=["POST"])
def twilio_process():
    """
    Εδώ παίρνουμε τη μετατροπή ομιλίας σε κείμενο από Twilio,
    στέλνουμε στο OpenAI και παίζουμε TTS.
    Αν το TTS αργεί ή αποτύχει, fallback σε resp.say() (αμεσότερο).
    Επίσης το επόμενό gather έχει bargeIn=True.
    """
    speech = request.form.get("SpeechResult", "") or ""
    call_sid = request.form.get("CallSid", str(uuid.uuid4()))
    resp = VoiceResponse()

    if not speech.strip():
        resp.say("Δεν σας άκουσα, μπορείτε να επαναλάβετε;", language="el-GR")
        resp.redirect("/twilio/voice")
        return str(resp), 200, {"Content-Type": "text/xml"}

    print("USER:", speech)
    reply = openai_reply(call_sid, speech)
    print("BOT :", reply)

    # Προσπαθούμε γρήγορα να πάρουμε ElevenLabs mp3 (μικρό timeout)
    audio_url = None
    if ELEVEN_API_KEY:
        audio_url = eleven_tts(reply, call_sid, timeout_seconds=3.0)

    # Αν έχουμε mp3, παίξε το — αλλιώς πες το με Twilio Say άμεσα
    if audio_url:
        # Twilio θα συνεχίσει να δέχεται barge-in αν ο χρήστης ξεκινήσει να μιλάει
        resp.play(audio_url)
    else:
        # fallback άμεσο, χωρίς network wait
        resp.say(reply, language="el-GR")

    # ΝΕΟ gather που επιτρέπει barge-in (ο χρήστης μπορεί να διακόψει την απάντηση)
    followup = Gather(
        input="speech",
        action="/twilio/process",
        language="el-GR",
        speech_timeout="auto",
        bargeIn=True,
    )
    followup.say(
        "Σας ακούω. Θέλετε κάτι άλλο ή να ολοκληρώσουμε την παραγγελία;",
        language="el-GR",
    )
    resp.append(followup)

    # Αν ο χρήστης δεν απαντήσει, επανέλαβε την αρχική ερώτηση
    resp.redirect("/twilio/voice")

    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


@app.route("/call-me", methods=["GET", "POST"])
def call_me():
    to_number = request.values.get("to")
    if not to_number:
        return jsonify({"error": "missing 'to'"}), 400
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_NUMBER:
        return jsonify({"error": "missing Twilio creds"}), 500
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    call = client.calls.create(
        to=to_number,
        from_=TWILIO_NUMBER,
        url=f"{BASE_URL}/twilio/voice",
    )
    return jsonify({"status": "calling", "sid": call.sid})


if __name__ == "__main__":
    print(f"Running on port {PORT}")
    # σε production βγάζουμε debug=True
    app.run(host="0.0.0.0", port=PORT, debug=False)

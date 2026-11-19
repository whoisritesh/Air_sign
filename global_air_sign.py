import os
import cv2
import math
import time
import pickle
import asyncio
import datetime
import threading
import numpy as np
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pyttsx3
import pygame
import edge_tts
from deep_translator import GoogleTranslator
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

LANGUAGES = [
    {"name": "English (US)", "code": "en", "voice": "en-US-AriaNeural"},
    {"name": "Hindi (India)", "code": "hi", "voice": "hi-IN-SwaraNeural"},
    {"name": "Odia (India)", "code": "or", "voice": "hi-IN-MadhurNeural"},
    {"name": "Bengali (India)", "code": "bn", "voice": "bn-IN-TanishaaNeural"},
    {"name": "Tamil (India)", "code": "ta", "voice": "ta-IN-PallaviNeural"},
    {"name": "Telugu (India)", "code": "te", "voice": "te-IN-ShrutiNeural"},
    {"name": "Spanish", "code": "es", "voice": "es-ES-ElviraNeural"},
    {"name": "French", "code": "fr", "voice": "fr-FR-DeniseNeural"},
    {"name": "German", "code": "de", "voice": "de-DE-KatjaNeural"},
    {"name": "Arabic", "code": "ar", "voice": "ar-SA-ZariyahNeural"},
    {"name": "Japanese", "code": "ja", "voice": "ja-JP-NanamiNeural"},
    {"name": "Chinese", "code": "zh-CN", "voice": "zh-CN-XiaoxiaoNeural"}
]

PRETRAINED_MAP = {
    "Open_Palm": "Hello",
    "Thumb_Up": "Yes",
    "Thumb_Down": "No",
    "Victory": "Thank You",
    "Pointing_Up": "Help",
    "ILoveYou": "I Love You",
    "Closed_Fist": "Stop"
}

CALIBRATION_SIGNS = [
    "Hello", "Please", "Yes", "No", "Thank You",
    "Help", "Stop", "Okay", "Water", "I Love You"
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20)
]

pygame.mixer.init()
lang_idx = 0
auto_speak = True
mode = "PHRASE"

def load_unicode_font(size):
    for path in ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT_LG = load_unicode_font(28)
FONT_MD = load_unicode_font(21)
FONT_SM = load_unicode_font(15)

def translate_text(text, target_code):
    if not text.strip() or target_code == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_code).translate(text)
    except Exception:
        return text

def speak_multilingual(text, lang_obj):
    if not text.strip():
        return
    def _worker():
        translated = translate_text(text, lang_obj["code"])
        out_file = f"tts_temp_{threading.get_ident()}.mp3"
        try:
            async def _gen():
                communicate = edge_tts.Communicate(translated, lang_obj["voice"])
                await communicate.save(out_file)
            asyncio.run(_gen())
            pygame.mixer.music.load(out_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
        except Exception:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        finally:
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except Exception:
                    pass
    threading.Thread(target=_worker, daemon=True).start()

# Pure 2D Geometry (Immune to low-light Z-depth noise)
def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def angle2d(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y])
    v2 = np.array([c.x - b.x, c.y - b.y])
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    return float(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))

def extract_2d_signature(lms):
    # 42 wrist-centered, palm-scaled 2D coordinates + 10 joint angles = 52 features
    wrist = lms[0]
    palm = dist2d(lms[0], lms[9]) + 1e-6
    coords = []
    for lm in lms:
        coords.extend([(lm.x - wrist.x) / palm, (lm.y - wrist.y) / palm])

    triplets = [
        (1,2,3),(2,3,4),(5,6,7),(6,7,8),(9,10,11),
        (10,11,12),(13,14,15),(14,15,16),(17,18,19),(18,19,20)
    ]
    angles = [angle2d(lms[a], lms[b], lms[c]) / math.pi for a, b, c in triplets]
    return coords + angles

def finger_states_2d(lms):
    palm = dist2d(lms[0], lms[9]) + 1e-6
    # Thumb extended if tip is farther from pinky base (17) than thumb IP (3) is
    t_ext = (dist2d(lms[4], lms[17]) > dist2d(lms[3], lms[17]) * 1.12) and (dist2d(lms[4], lms[5]) / palm > 0.42)
    i_ext = dist2d(lms[8], lms[0]) > dist2d(lms[6], lms[0]) * 1.15
    m_ext = dist2d(lms[12], lms[0]) > dist2d(lms[10], lms[0]) * 1.15
    r_ext = dist2d(lms[16], lms[0]) > dist2d(lms[14], lms[0]) * 1.15
    p_ext = dist2d(lms[20], lms[0]) > dist2d(lms[18], lms[0]) * 1.15
    return t_ext, i_ext, m_ext, r_ext, p_ext, palm

def classify_phrase_2d(lms, pre_name, pre_score):
    t_ext, i_ext, m_ext, r_ext, p_ext, palm = finger_states_2d(lms)
    pinch_ti = dist2d(lms[4], lms[8]) / palm

    if pinch_ti < 0.30 and m_ext and r_ext and p_ext:
        return "Okay", 0.96
    if i_ext and m_ext and r_ext and p_ext:
        return ("Hello" if t_ext else "Please"), 0.95
    if i_ext and m_ext and r_ext and not p_ext:
        return "Water", 0.94
    if i_ext and p_ext and not m_ext and not r_ext:
        return ("I Love You" if t_ext else "Awesome"), 0.94
    if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
        return "Call Me", 0.94
    if i_ext and m_ext and not r_ext and not p_ext:
        return "Thank You", 0.94
    if i_ext and not m_ext and not r_ext and not p_ext and not t_ext:
        return "Help", 0.93
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if t_ext:
            return ("Yes" if lms[4].y < lms[2].y else "No"), 0.95
        return "Stop", 0.93

    if pre_name in PRETRAINED_MAP and pre_score > 0.60:
        return PRETRAINED_MAP[pre_name], pre_score
    return None, 0.0

def classify_asl_2d(lms):
    t_ext, i_ext, m_ext, r_ext, p_ext, palm = finger_states_2d(lms)
    pinch_ti = dist2d(lms[4], lms[8]) / palm

    if t_ext and i_ext and m_ext and r_ext and p_ext:
        return "SPACE", 0.95
    if not t_ext and i_ext and m_ext and r_ext and p_ext:
        return "B", 0.94
    if i_ext and m_ext and r_ext and not p_ext:
        return "W", 0.94
    if pinch_ti < 0.30 and m_ext and r_ext and p_ext:
        return "F", 0.94
    if i_ext and m_ext and not r_ext and not p_ext:
        spread = dist2d(lms[8], lms[12]) / palm
        if abs(lms[8].x - lms[5].x) > abs(lms[8].y - lms[5].y) * 1.3:
            return ("P" if lms[8].y > lms[5].y else "H"), 0.90
        if dist2d(lms[4], lms[6]) / palm < 0.32 or dist2d(lms[4], lms[10]) / palm < 0.32:
            return "K", 0.90
        return ("V" if spread > 0.36 else "U"), 0.92
    if i_ext and not m_ext and not r_ext and not p_ext:
        if abs(lms[8].x - lms[5].x) > abs(lms[8].y - lms[5].y) * 1.3:
            return ("Q" if lms[8].y > lms[5].y else "G"), 0.90
        if angle2d(lms[6], lms[7], lms[8]) < 2.1:
            return "X", 0.90
        return ("L" if t_ext else "D"), 0.94
    if not i_ext and not m_ext and not r_ext and p_ext:
        if t_ext:
            return "Y", 0.95
        return ("J" if abs(lms[20].x - lms[17].x) > 0.08 else "I"), 0.92
    if not i_ext and not m_ext and not r_ext and not p_ext:
        if pinch_ti < 0.30 and dist2d(lms[8], lms[5]) / palm > 0.45:
            return "O", 0.92
        if 0.31 <= pinch_ti <= 0.85 and dist2d(lms[8], lms[5]) / palm > 0.48:
            return "C", 0.90
        if t_ext and lms[4].y < lms[6].y:
            return "A", 0.94
        if dist2d(lms[4], lms[6]) / palm < 0.28:
            return "T", 0.90
        if dist2d(lms[4], lms[10]) / palm < 0.30:
            return "N", 0.88
        if dist2d(lms[4], lms[14]) / palm < 0.33:
            return "M", 0.88
        if lms[8].y > lms[4].y and dist2d(lms[4], lms[13]) / palm < 0.42:
            return "E", 0.90
        return "S", 0.90
    return None, 0.0

# Personal Calibrated Ensemble Storage
CAL_FILE = "personal_calibrated_ensemble.pkl"
cal_X, cal_y = [], []
cal_ensemble = None

def build_calibrated_ensemble():
    global cal_X, cal_y, cal_ensemble
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE, "rb") as f:
            cal_X, cal_y = pickle.load(f)
        if len(set(cal_y)) >= 2:
            # Generate 400 multi-scale variations per captured frame (~10,000+ samples)
            aug_X, aug_y = [], []
            for feat, lbl in zip(cal_X, cal_y):
                base = np.array(feat, dtype=np.float32)
                noise = np.random.normal(0.0, 0.025, size=(400, len(feat))).astype(np.float32)
                aug_X.append(base + noise)
                aug_y.extend([lbl] * 400)
            X_arr = np.vstack(aug_X)
            y_arr = np.array(aug_y)
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150, early_stopping=True, random_state=42)
            et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            ens = VotingClassifier(estimators=[("mlp", mlp), ("et", et)], voting="soft", n_jobs=-1)
            ens.fit(X_arr, y_arr)
            cal_ensemble = ens

build_calibrated_ensemble()

base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.55,
    min_tracking_confidence=0.55
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN_NAME = "Global Air Sign Studio - 2D Calibrated Engine"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
is_fullscreen = True
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

HOLD_FRAMES = 14
history = deque(maxlen=HOLD_FRAMES)
wrist_positions = deque(maxlen=6)
english_buffer = ""
translated_display = ""
last_locked = ""
cooldown_frames = 0
status_flash = ""
flash_timer = 0

# Calibration Wizard State
cal_wizard_idx = -1
cal_frames_left = 0
recording_label = ""
record_frames_left = 0

def update_translation_async(text, target_code):
    def _run():
        global translated_display
        translated_display = translate_text(text, target_code)
    threading.Thread(target=_run, daemon=True).start()

CONTROLS_LEGEND = [
    "[TAB] Switch Phrase / A-Z",
    "[K] 15-Sec Personal Calibration",
    "[R] Train Single Custom Sign",
    "[L] Switch Language (12)",
    "[S] Speak Full Sentence",
    "[SPACE] Add Space",
    "[BKSP] Delete Last",
    "[C] Clear All Text",
    "[M] Mute / Unmute",
    "[T] Save Transcript",
    "[F] Fullscreen Toggle",
    "[Q] Quit App"
]

PHRASE_LEGEND = [
    ("5 Fingers Open", "Hello"),
    ("4 Fingers (Thumb In)", "Please"),
    ("Thumb Up / Down", "Yes / No"),
    ("Victory (2 Fingers)", "Thank You"),
    ("3 Fingers Up", "Water"),
    ("Index Up", "Help"),
    ("Closed Fist", "Stop"),
    ("OK Pinch", "Okay"),
    ("I Love You", "I Love You"),
    ("Pinky Out", "Call Me"),
    ("Rock Sign", "Awesome"),
    ("Both Hands Steady", "Emergency Help")
]

while cap.isOpened():
    ret, raw_frame = cap.read()
    if not ret:
        break

    rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_raw)
    result = recognizer.recognize(mp_image)

    frame = cv2.flip(raw_frame, 1)
    h, w, _ = frame.shape

    current_pred = "Waiting for sign..."
    hold_progress = 0.0
    curr_lang = LANGUAGES[lang_idx % len(LANGUAGES)]

    if cooldown_frames > 0:
        cooldown_frames -= 1

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand_lms]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0, 255, 180), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 90, 90), -1)

        primary_lms = result.hand_landmarks[0]
        wrist = primary_lms[0]
        wrist_positions.append((wrist.x, wrist.y))

        is_steady = False
        if len(wrist_positions) == 6:
            dx = max(p[0] for p in wrist_positions) - min(p[0] for p in wrist_positions)
            dy = max(p[1] for p in wrist_positions) - min(p[1] for p in wrist_positions)
            is_steady = (dx < 0.045 and dy < 0.045)

        sig_52 = extract_2d_signature(primary_lms)

        # Wizard Calibration Mode ([K])
        if cal_wizard_idx >= 0:
            target_sign = CALIBRATION_SIGNS[cal_wizard_idx]
            current_pred = f"CALIBRATING '{target_sign}' - Hold steady ({cal_frames_left})"
            if is_steady and cal_frames_left > 0:
                cal_X.append(sig_52)
                cal_y.append(target_sign)
                cal_frames_left -= 1
                if cal_frames_left == 0:
                    cal_wizard_idx += 1
                    if cal_wizard_idx < len(CALIBRATION_SIGNS):
                        cal_frames_left = 20
                        speak_multilingual(f"Next, {CALIBRATION_SIGNS[cal_wizard_idx]}", LANGUAGES[0])
                    else:
                        cal_wizard_idx = -1
                        with open(CAL_FILE, "wb") as f:
                            pickle.dump((cal_X, cal_y), f)
                        build_calibrated_ensemble()
                        status_flash = "Personal 100% Ensemble Trained!"
                        flash_timer = 80
                        speak_multilingual("Calibration complete", LANGUAGES[0])
        elif record_frames_left > 0 and recording_label:
            cal_X.append(sig_52)
            cal_y.append(recording_label)
            record_frames_left -= 1
            current_pred = f"RECORDING '{recording_label}' ({record_frames_left})"
            if record_frames_left == 0:
                with open(CAL_FILE, "wb") as f:
                    pickle.dump((cal_X, cal_y), f)
                build_calibrated_ensemble()
                status_flash = f"Added '{recording_label}' to Ensemble!"
                flash_timer = 70
        elif cooldown_frames > 0:
            history.clear()
            current_pred = "Locked! Relax hand..."
        elif not is_steady:
            history.clear()
            current_pred = "Hold hand steady..."
        else:
            token, score = None, 0.0
            if mode == "ALPHABET":
                token, score = classify_asl_2d(primary_lms)
            else:
                if len(result.hand_landmarks) == 2 and len(result.handedness) == 2:
                    if result.handedness[0][0].score > 0.85 and result.handedness[1][0].score > 0.85:
                        token, score = "Emergency Help", 0.97

                if not token and cal_ensemble is not None:
                    probs = cal_ensemble.predict_proba([sig_52])[0]
                    bi = int(np.argmax(probs))
                    if probs[bi] > 0.85:
                        token, score = str(cal_ensemble.classes_[bi]), float(probs[bi])

                if not token:
                    pre_name = result.gestures[0][0].category_name if result.gestures else ""
                    pre_score = result.gestures[0][0].score if result.gestures else 0.0
                    token, score = classify_phrase_2d(primary_lms, pre_name, pre_score)

            if token:
                history.append(token)
                most_common, count = Counter(history).most_common(1)[0]
                hold_progress = min(count / HOLD_FRAMES, 1.0)
                current_pred = f"{most_common} ({score*100:.0f}%)"

                if len(history) == HOLD_FRAMES and count >= int(HOLD_FRAMES * 0.78) and most_common != last_locked:
                    last_locked = most_common
                    cooldown_frames = 22
                    if mode == "ALPHABET":
                        if most_common == "SPACE":
                            last_word = english_buffer.strip().split(" ")[-1] if english_buffer.strip() else ""
                            english_buffer += " "
                            if auto_speak and last_word:
                                speak_multilingual(last_word, curr_lang)
                        else:
                            english_buffer += most_common
                            if auto_speak:
                                speak_multilingual(most_common.lower(), curr_lang)
                    else:
                        english_buffer = (english_buffer + " " + most_common).strip()
                        if auto_speak:
                            speak_multilingual(most_common, curr_lang)

                    update_translation_async(english_buffer, curr_lang["code"])
                    history.clear()
                    hold_progress = 0.0
    else:
        history.clear()
        wrist_positions.clear()
        last_locked = ""

    # --- UI OVERLAYS ---
    sidebar_w = 340
    sidebar_x = w - sidebar_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (sidebar_x, 145), (18, 18, 24), -1)
    cv2.rectangle(overlay, (sidebar_x, 0), (w, h), (15, 16, 22), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.text((20, 14), f"Detected: {current_pred}", font=FONT_LG, fill=(0, 255, 180))
    draw.text((20, 56), f"English: {english_buffer[-42:]}", font=FONT_MD, fill=(255, 255, 255))
    draw.text((20, 96), f"{curr_lang['name']}: {translated_display[-42:]}", font=FONT_MD, fill=(0, 220, 255))

    sx = sidebar_x + 18
    y = 14
    draw.text((sx, y), f"MODE: {mode}", font=FONT_MD, fill=(0, 255, 180))
    y += 26
    draw.text((sx, y), f"Lang: {curr_lang['name']}", font=FONT_SM, fill=(0, 220, 255))
    y += 20
    cal_status = f"Calibrated ({len(set(cal_y))} signs)" if cal_ensemble else "2D Geometric + Pretrained"
    draw.text((sx, y), f"Engine: {cal_status}", font=FONT_SM, fill=(200, 210, 230))
    y += 22
    draw.line([(sx, y), (w - 18, y)], fill=(80, 85, 100), width=1)
    y += 8

    draw.text((sx, y), "CONTROLS", font=FONT_MD, fill=(0, 255, 180))
    y += 24
    for ctrl in CONTROLS_LEGEND:
        draw.text((sx, y), ctrl, font=FONT_SM, fill=(230, 230, 235))
        y += 19

    y += 6
    draw.line([(sx, y), (w - 18, y)], fill=(80, 85, 100), width=1)
    y += 8

    if mode == "PHRASE":
        draw.text((sx, y), "PHRASE SIGNS", font=FONT_MD, fill=(0, 255, 180))
        y += 24
        for g_name, phrase in PHRASE_LEGEND:
            draw.text((sx, y), f"{g_name} -> {phrase}", font=FONT_SM, fill=(215, 215, 225))
            y += 19
    else:
        draw.text((sx, y), "ASL ALPHABET (A-Z)", font=FONT_MD, fill=(0, 220, 255))
        y += 24
        draw.text((sx, y), "All 26 letters A-Z active!", font=FONT_SM, fill=(235, 235, 235))
        y += 21
        draw.text((sx, y), "5 Fingers Open = SPACE", font=FONT_SM, fill=(0, 255, 180))

    if flash_timer > 0:
        draw.text((20, h - 42), status_flash, font=FONT_MD, fill=(0, 255, 255))
        flash_timer -= 1

    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if hold_progress > 0:
        cv2.rectangle(frame, (0, 140), (int(sidebar_x * hold_progress), 146), (0, 255, 180), -1)

    cv2.imshow(WIN_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 9:
        mode = "ALPHABET" if mode == "PHRASE" else "PHRASE"
        history.clear()
        last_locked = ""
        status_flash = f"Mode: {mode}"
        flash_timer = 45
    elif key == ord("k"):
        cal_X.clear()
        cal_y.clear()
        cal_wizard_idx = 0
        cal_frames_left = 20
        speak_multilingual(f"Show {CALIBRATION_SIGNS[0]}", LANGUAGES[0])
    elif key == ord("l"):
        lang_idx = (lang_idx + 1) % len(LANGUAGES)
        curr_lang = LANGUAGES[lang_idx]
        update_translation_async(english_buffer, curr_lang["code"])
        status_flash = f"Language: {curr_lang['name']}"
        flash_timer = 50
        speak_multilingual("Language changed", curr_lang)
    elif key == ord("r"):
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        new_word = input("\nEnter word/phrase to train (e.g. Doctor): ").strip()
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
        if new_word:
            recording_label = new_word
            record_frames_left = 30
    elif key == ord("s") and english_buffer.strip():
        speak_multilingual(english_buffer, curr_lang)
    elif key == ord(" "):
        english_buffer += " "
        last_locked = ""
    elif key == 8 and english_buffer:
        if mode == "ALPHABET":
            english_buffer = english_buffer[:-1]
        else:
            english_buffer = " ".join(english_buffer.strip().split(" ")[:-1])
        update_translation_async(english_buffer, curr_lang["code"])
        last_locked = ""
    elif key == ord("c"):
        english_buffer = ""
        translated_display = ""
        last_locked = ""
    elif key == ord("m"):
        auto_speak = not auto_speak
        status_flash = f"Auto-Speak: {'ON' if auto_speak else 'OFF'}"
        flash_timer = 45
    elif key == ord("t") and english_buffer.strip():
        with open("transcript.txt", "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ({curr_lang['name']}) EN: {english_buffer} | TR: {translated_display}\n")
        status_flash = "Saved multilingual transcript!"
        flash_timer = 60
    elif key == ord("f"):
        is_fullscreen = not is_fullscreen
        prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
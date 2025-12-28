import os
import cv2
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)

# ================= CONFIG =================
TOKEN = "8596835385:AAGIvKyUkoL1GWx5zGjpDfuTVP5ms2Rn8nM"
ORIGINAL, QUESTIONED = range(2)
AUDIT_FILE = "audit_log.json"

# ================= IMAGE PROCESSING =================
def crop_to_signature(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    pad = 10
    return image[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = crop_to_signature(img)
    img = cv2.resize(img, (300, 300))
    return img

# ================= FEATURES =================
def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ink_density = np.sum(img < 128)
    stroke_count = len(contours)
    edge_strength = np.mean(edges)
    aspect_ratio = img.shape[0] / img.shape[1]

    dist = cv2.distanceTransform(255-img, cv2.DIST_L2, 5)
    stroke_variation = np.std(dist)
    tremor = np.std([cv2.arcLength(c, False) for c in contours]) if contours else 0

    return np.array([
        ink_density,
        stroke_count,
        edge_strength,
        aspect_ratio,
        stroke_variation,
        tremor
    ], dtype=np.float32)

# ================= SECURITY =================
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def write_audit(log):
    data = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r") as f:
            data = json.load(f)
    data.append(log)
    with open(AUDIT_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ================= AI EXPLANATION (RULE-BASED) =================
def explain_result(final_score):
    if final_score >= 75:
        return "✅ High structural and pattern similarity detected. Signature is likely genuine."
    elif final_score >= 60:
        return "⚠ Moderate similarity. Minor stroke or size variations observed."
    else:
        return "❌ Low similarity. Stroke pattern, tremor, or structure differs significantly."

# ================= COMPARISON =================
def compare_signatures(p1, p2, out_img):
    img1 = preprocess(p1)
    img2 = preprocess(p2)
    if img1 is None or img2 is None:
        return None, None, None

    ssim_score, diff = ssim(img1, img2, full=True)
    ssim_pct = ssim_score * 100

    f1 = extract_features(img1)
    f2 = extract_features(img2)
    ml_sim = max(0, 100 - np.linalg.norm(f1 - f2))

    final = (ssim_pct + ml_sim) / 2

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(img2, cmap="gray"); ax[1].set_title("Questioned")
    ax[2].imshow(diff, cmap="gray"); ax[2].set_title("Difference")
    for a in ax: a.axis("off")
    plt.savefig(out_img)
    plt.close()

    return ssim_pct, ml_sim, final

# ================= BOT HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✍ Smart Signature Verification Bot\n\n"
        "📄 Send ORIGINAL scanned signature"
    )
    return ORIGINAL

async def receive_original(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    path = f"orig_{update.effective_user.id}.jpg"
    await photo.download_to_drive(path)
    context.user_data["orig"] = path
    await update.message.reply_text("Now send QUESTIONED signature")
    return QUESTIONED

async def receive_questioned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    qpath = f"ques_{update.effective_user.id}.jpg"
    await photo.download_to_drive(qpath)

    opath = context.user_data["orig"]
    out = f"result_{update.effective_user.id}.png"

    ssim_s, ml_s, final = compare_signatures(opath, qpath, out)
    explanation = explain_result(final)

    report = (
        f"🔍 Signature Analysis Report\n\n"
        f"SSIM Similarity : {ssim_s:.2f}%\n"
        f"ML Similarity   : {ml_s:.2f}%\n"
        f"Final Score    : {final:.2f}%\n\n"
        f"{explanation}"
    )

    await update.message.reply_text(report)
    await update.message.reply_photo(open(out, "rb"))

    write_audit({
        "time": str(datetime.now()),
        "score": final,
        "hash_original": file_hash(opath),
        "hash_questioned": file_hash(qpath)
    })

    for f in [opath, qpath, out]:
        os.remove(f)

    return ConversationHandler.END

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ORIGINAL: [MessageHandler(filters.PHOTO, receive_original)],
            QUESTIONED: [MessageHandler(filters.PHOTO, receive_questioned)],
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
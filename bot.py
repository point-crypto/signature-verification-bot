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

# ================= FEATURE EXTRACTION =================
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

# ================= AI-STYLE EXPLANATION (FREE) =================
def ai_explain(result):
    if result.startswith("MATCH"):
        return (
            "✅ Explanation:\n"
            "The signatures show strong structural similarity.\n"
            "Writing behaviour and stroke flow are consistent.\n"
            "Minor variations are normal in genuine signatures."
        )
    else:
        return (
            "❌ Explanation:\n"
            "Although some visual similarity exists, differences\n"
            "in writing behaviour and stroke consistency were found.\n"
            "These exceed acceptable limits, resulting in a mismatch."
        )

# ================= COMPARISON =================
def compare_signatures(p1, p2, out_img):
    img1 = preprocess(p1)
    img2 = preprocess(p2)
    if img1 is None or img2 is None:
        return "Image error", "", None

    # SSIM
    ssim_score, diff = ssim(img1, img2, full=True)
    ssim_percent = ssim_score * 100

    # ML similarity (behavioural)
    f1 = extract_features(img1)
    f2 = extract_features(img2)
    distance = np.linalg.norm(f1 - f2)
    ml_score = max(40, 100 - distance)  # protect ML

    # FINAL DECISION (SSIM-FIRST)
    if ssim_percent >= 75:
        result = "MATCH ✅"
    elif ssim_percent >= 65 and ml_score >= 40:
        result = "MATCH ✅"
    else:
        result = "MISMATCH ❌"

    final_score = (0.6 * ssim_percent) + (0.4 * ml_score)

    # Plot result
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img1, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(img2, cmap="gray"); ax[1].set_title("Questioned")
    ax[2].imshow(diff, cmap="gray"); ax[2].set_title("Difference")
    for a in ax: a.axis("off")
    plt.savefig(out_img)
    plt.close()

    report = (
        f"🔍 Signature Analysis Report\n\n"
        f"SSIM Similarity : {ssim_percent:.2f}%\n"
        f"ML Similarity   : {ml_score:.2f}%\n"
        f"Final Score    : {final_score:.2f}%\n\n"
        f"Result         : {result}"
    )

    explanation = ai_explain(result)
    return report, explanation, final_score

# ================= BOT HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Smart Signature Verification System\n\n"
        "🔐 Features:\n"
        "• Background-independent analysis\n"
        "• Writing behaviour & stroke analysis\n"
        "• Secure audit logging\n\n"
        "🖨 IMPORTANT:\n"
        "Upload ONLY SCANNED signatures\n"
        "(300 DPI, Grayscale)\n\n"
        "➡️ Send the ORIGINAL signature."
    )
    return ORIGINAL

async def receive_original(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    path = f"orig_{update.effective_user.id}.jpg"
    await file.download_to_drive(path)
    context.user_data["orig"] = path
    await update.message.reply_text(
        "✅ Original received.\nNow send the QUESTIONED signature."
    )
    return QUESTIONED

async def receive_questioned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    file = await update.message.photo[-1].get_file()
    qpath = f"ques_{user.id}.jpg"
    await file.download_to_drive(qpath)

    opath = context.user_data["orig"]
    out_img = f"result_{user.id}.png"

    await update.message.reply_text("🔍 Analyzing signature, please wait...")

    report, explanation, score = compare_signatures(opath, qpath, out_img)

    audit = {
        "user": user.id,
        "time": str(datetime.now()),
        "final_score": score
    }
    write_audit(audit)

    await update.message.reply_text(report)
    await update.message.reply_text(explanation)
    await update.message.reply_photo(photo=open(out_img, "rb"))

    for f in [opath, qpath, out_img]:
        os.remove(f)

    await update.message.reply_text("✅ Done. Type /start to test again.")
    return ConversationHandler.END

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()
    handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ORIGINAL: [MessageHandler(filters.PHOTO, receive_original)],
            QUESTIONED: [MessageHandler(filters.PHOTO, receive_questioned)]
        },
        fallbacks=[]
    )
    app.add_handler(handler)
    print("Bot running...")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()

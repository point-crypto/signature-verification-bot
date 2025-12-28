import os
import cv2
import json
import hashlib
import numpy as np
from datetime import datetime

from skimage.metrics import structural_similarity as ssim

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)

# ================= CONFIG =================
TOKEN = "8596835385:AAGIvKyUkoL1GWx5zGjpDfuTVP5ms2Rn8nM"

ORIGINAL, TEST = range(2)
AUDIT_FILE = "audit_log.json"

# ================= IMAGE PROCESSING =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (300, 150))
    return img

# ================= FEATURE EXTRACTION =================
def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ink_density = np.sum(img < 128)
    stroke_count = len(contours)
    edge_strength = np.mean(edges)
    aspect_ratio = img.shape[1] / img.shape[0]

    return np.array([
        ink_density,
        stroke_count,
        edge_strength,
        aspect_ratio
    ], dtype=np.float32)

# ================= SIMILARITY =================
def ml_similarity(f1, f2):
    diff = np.abs(f1 - f2)
    score = 1 / (1 + np.mean(diff))
    return score * 100

# ================= AUDIT =================
def write_audit(data):
    logs = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r") as f:
            logs = json.load(f)

    logs.append(data)

    with open(AUDIT_FILE, "w") as f:
        json.dump(logs, f, indent=4)

# ================= BOT COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Signature Verification Bot*\n\n"
        "📄 Step 1: Send the *ORIGINAL* scanned signature\n"
        "📄 Step 2: Send the *TEST* scanned signature\n\n"
        "⚠️ Only SCANNED images are allowed",
        parse_mode="Markdown"
    )
    return ORIGINAL

async def original_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    path = "original.jpg"
    await photo.download_to_drive(path)

    context.user_data["original"] = path
    await update.message.reply_text("✅ Original signature saved.\n\n➡️ Send TEST signature")
    return TEST

async def test_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    orig_path = context.user_data.get("original")

    img1 = preprocess(orig_path)
    img2 = preprocess(test_path)

    if img1 is None or img2 is None:
        await update.message.reply_text("❌ Image processing failed")
        return ConversationHandler.END

    ssim_score = ssim(img1, img2) * 100
    f1 = extract_features(img1)
    f2 = extract_features(img2)
    ml_score = ml_similarity(f1, f2)

    final_score = (0.7 * ssim_score) + (0.3 * ml_score)

    if final_score >= 75:
        result = "MATCH ✅"
        reason = (
            "✔ High structural similarity\n"
            "✔ Writing pattern consistent\n"
            "✔ Stroke flow stable"
        )
    else:
        result = "MISMATCH ❌"
        reason = (
            "✖ Stroke pattern mismatch\n"
            "✖ Writing pressure differs\n"
            "✖ Structural variation detected"
        )

    report = (
        "🔍 *Signature Analysis Report*\n\n"
        f"SSIM Similarity : `{ssim_score:.2f}%`\n"
        f"ML Similarity   : `{ml_score:.2f}%`\n"
        f"Final Score    : `{final_score:.2f}%`\n\n"
        f"*Result*: {result}\n\n"
        f"*Reason*\n{reason}"
    )

    await update.message.reply_text(report, parse_mode="Markdown")

    write_audit({
        "time": str(datetime.now()),
        "ssim": round(ssim_score, 2),
        "ml": round(ml_score, 2),
        "final": round(final_score, 2),
        "result": result
    })

    return ConversationHandler.END

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ORIGINAL: [MessageHandler(filters.PHOTO, original_image)],
            TEST: [MessageHandler(filters.PHOTO, test_image)],
        },
        fallbacks=[]
    )

    app.add_handler(conv)

    print("✅ Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
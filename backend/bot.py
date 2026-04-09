import os
import uuid
import asyncio
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

application = None

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "🤖 *DeepFake Guardian Bot*\n\n"
        "Hello! I am an AI-powered deepfake detection bot. "
        "Send me any video or a document containing a video, "
        "and I will analyze it to see if it's Real or Fake!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.video and not update.message.document:
        await update.message.reply_text("Please send a valid video file.")
        return

    file_id = None
    if update.message.video:
        file_id = update.message.video.file_id
    elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("video"):
        file_id = update.message.document.file_id
    else:
        await update.message.reply_text("Please send a valid video file.")
        return

    message = await update.message.reply_text("⏳ Processing your video using my deep neural networks. This might take a moment...")
    
    try:
        new_file = await context.bot.get_file(file_id)
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4()}.mp4"
        
        await new_file.download_to_drive(custom_path=str(temp_path))
        
        import aiohttp
        port = os.getenv("PORT", 8000)
        async with aiohttp.ClientSession() as session:
            with open(temp_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename='video.mp4')
                
                # Make a request to our own API seamlessly
                async with session.post(f'http://127.0.0.1:{port}/predict', data=data) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        verdict = res["verdict"]
                        prob = res["fake_prob"]
                        elapsed = res["elapsed"]
                        
                        emoji = "🔴" if verdict == "FAKE" else "🟢"
                        text = (
                            f"{emoji} *Verdict:* {verdict}\n"
                            f"🧠 *Confidence (Fake):* {prob:.2%}\n"
                            f"⏱️ *Analysis Time:* {elapsed:.2f}s"
                        )
                        await message.edit_text(text, parse_mode="Markdown")
                    else:
                        error_data = await resp.json()
                        err = error_data.get("error", "Unknown error")
                        await message.edit_text(f"❌ Error processing the video: {err}")
                        
        if temp_path.exists():
            temp_path.unlink()
            
    except Exception as e:
        await message.edit_text(f"❌ An error occurred: {str(e)}")

async def start_bot():
    global application
    if not BOT_TOKEN:
        print("[Bot] No TELEGRAM_BOT_TOKEN environment variable found. Skipping bot startup.")
        return
        
    print("[Bot] Starting Telegram Bot...")
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("[Bot] Telegram Bot is polling...")

async def stop_bot():
    global application
    if application:
        print("[Bot] Stopping Telegram Bot...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        print("[Bot] Stopped.")

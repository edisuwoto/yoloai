from moviepy.editor import VideoFileClip

# Daftar video yang mau dikonversi
input_videos = {
    "demo/flood_detection_output.mp4": "demo/flood_detection_output.gif",
    "demo/fire_output.mp4": "demo/fire_output.gif"
}

for mp4_path, gif_path in input_videos.items():
    print(f"Converting {mp4_path} → {gif_path} ...")
    clip = VideoFileClip(mp4_path)

    # Resize biar tidak terlalu besar (lebar 500px)
    clip_resized = clip.resize(width=500)

    # Export ke GIF
    clip_resized.write_gif(gif_path, fps=10)

print("✅ Semua video berhasil dikonversi jadi GIF!")

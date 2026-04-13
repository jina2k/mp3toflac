import librosa
import numpy as np
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import csv

# -------------------------
# User-editable constants
# -------------------------
WINDOW_SEC = 3.3        # duration of each MP3 chunk in seconds
SR_TARGET = 16000      # target sample rate
CONFIDENCE_THRESHOLD = 0.5
MAX_THREADS = 4        # threads for cross-correlation
# -------------------------

# GPU detection
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU detected: using CuPy for FFT")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available: falling back to CPU")

# -------------------------
# Helper Functions
# -------------------------

def update_osu_audio_filenames(mp3_file):
    folder = os.path.dirname(mp3_file)
    mp3_name = os.path.splitext(os.path.basename(mp3_file))[0]

    for file in os.listdir(folder):
        if file.endswith(".osu"):
            osu_path = os.path.join(folder, file)

            with open(osu_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_general_section = False
            modified = False

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Detect [General] section
                if stripped == "[General]":
                    in_general_section = True
                    continue

                # If we hit another section, stop tracking
                if stripped.startswith("[") and stripped != "[General]":
                    in_general_section = False

                # Replace AudioFilename inside [General]
                if in_general_section and stripped.startswith("AudioFilename:"):
                    lines[i] = f"AudioFilename: {mp3_name}.flac\n"
                    modified = True

            if modified:
                with open(osu_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                print(f"Updated: {osu_path}")

def windows_to_wsl_path(path):
    path = path.replace("\\", "/")
    if ":" in path:
        drive, rest = path.split(":", 1)
        path = f"/mnt/{drive.lower()}{rest}"
    return path

def audio_to_array(y, sr, target_sr=SR_TARGET):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y

def run_ffmpeg(*args):
    try:
        subprocess.run(["ffmpeg", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")

def cross_correlation(a, b):
    try:
        if GPU_AVAILABLE:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b[::-1])
            n = a_gpu.size + b_gpu.size - 1
            corr = cp.fft.ifft(cp.fft.fft(a_gpu, n=n) * cp.fft.fft(b_gpu, n=n)).real
            return float(cp.max(corr) / (cp.linalg.norm(a_gpu) * cp.linalg.norm(cp.asarray(b))))
        else:
            n = len(a) + len(b) - 1
            corr = np.fft.ifft(np.fft.fft(a, n=n) * np.fft.fft(b[::-1], n=n)).real
            return np.max(corr) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception:
        # fallback to CPU
        n = len(a) + len(b) - 1
        corr = np.fft.ifft(np.fft.fft(a, n=n) * np.fft.fft(b[::-1], n=n)).real
        return np.max(corr) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_silence_flac(duration_sec, sr, output_file):
    run_ffmpeg(
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sr}:cl=stereo",
        "-t", str(duration_sec),
        "-c:a", "flac",
        output_file
    )

def pad_segment(segment_file, intended_len_samples, sr):
    y, _ = librosa.load(segment_file, sr=sr)
    if len(y) >= intended_len_samples:
        return segment_file

    pad_sec = (intended_len_samples - len(y)) / sr
    tmp_silence = segment_file.replace(".flac", "_silence.flac")
    generate_silence_flac(pad_sec, sr, tmp_silence)

    concat_txt = segment_file.replace(".flac", "_concat.txt")
    with open(concat_txt, "w") as f:
        f.write(f"file '{os.path.abspath(segment_file)}'\n")
        f.write(f"file '{os.path.abspath(tmp_silence)}'\n")

    padded_file = segment_file.replace(".flac", "_padded.flac")
    run_ffmpeg(
        "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c:a", "flac", padded_file
    )

    os.remove(tmp_silence)
    os.remove(concat_txt)
    os.replace(padded_file, segment_file)
    return segment_file

# -------------------------
# Process Chunk
# -------------------------
def process_chunk(idx, mp3_chunk, mp3_start_sample, y_flac, window_size, sr, flac_file, segment_dir, prev_flac_end=0):
    if len(mp3_chunk) < window_size:
        pad_len = window_size - len(mp3_chunk)
        mp3_chunk = np.concatenate([mp3_chunk, np.zeros(pad_len, dtype=mp3_chunk.dtype)])

    best_score = -np.inf
    best_flac_start = prev_flac_end
    hop = max(window_size // 4, 1)

    # Multithreaded cross-correlation search
    def check_corr(start):
        flac_chunk = y_flac[start:start + len(mp3_chunk)]
        score = cross_correlation(flac_chunk, mp3_chunk)
        return start, score

    starts = list(range(prev_flac_end, len(y_flac) - len(mp3_chunk), hop))
    if GPU_AVAILABLE or len(starts) < 10:
        # small arrays, single-thread
        for start in starts:
            _, score = check_corr(start)
            if score > best_score:
                best_score = score
                best_flac_start = start
    else:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            results_threaded = executor.map(check_corr, starts)
            for start, score in results_threaded:
                if score > best_score:
                    best_score = score
                    best_flac_start = start

    mp3_end = mp3_start_sample + len(mp3_chunk)
    flac_end = best_flac_start + len(mp3_chunk)

    os.makedirs(segment_dir, exist_ok=True)
    segment_file = os.path.join(segment_dir, f"segment_{idx}.flac")

    if best_score < CONFIDENCE_THRESHOLD:
        generate_silence_flac(len(mp3_chunk)/sr, sr, segment_file)
    else:
        run_ffmpeg(
            "-y",
            "-i", flac_file,
            "-ss", str(best_flac_start/sr),
            "-to", str(flac_end/sr),
            "-c:a", "flac",
            segment_file
        )

    pad_segment(segment_file, len(mp3_chunk), sr)

    print(f"Chunk {idx}: MP3 {mp3_start_sample/sr:.3f}-{mp3_end/sr:.3f}s, "
          f"FLAC {best_flac_start/sr:.3f}-{flac_end/sr:.3f}s, score={best_score:.4f}", flush=True)

    return idx, mp3_start_sample, mp3_end, best_flac_start, flac_end, best_score, segment_file

# -------------------------
# Main
# -------------------------
def main():
    flac_file = windows_to_wsl_path(input("Drag the FLAC file here and press Enter:\n").strip('"'))
    mp3_file = windows_to_wsl_path(input("Drag the MP3 file here and press Enter:\n").strip('"'))

    # Update .osu files to use FLAC
    segment_dir = os.path.join(os.path.dirname(mp3_file), "segments")
    os.makedirs(segment_dir, exist_ok=True)

    print("Loading audio...")
    y_flac, _ = librosa.load(flac_file, sr=SR_TARGET)
    y_mp3, _ = librosa.load(mp3_file, sr=SR_TARGET)
    y_flac = audio_to_array(y_flac, SR_TARGET)
    y_mp3 = audio_to_array(y_mp3, SR_TARGET)

    window_size = int(WINDOW_SEC * SR_TARGET)
    results = []

    prev_flac_end = 0
    for idx, start in enumerate(range(0, len(y_mp3), window_size)):
        mp3_chunk = y_mp3[start:start+window_size]
        res = process_chunk(idx, mp3_chunk, start, y_flac, window_size, SR_TARGET, flac_file, segment_dir, prev_flac_end)
        results.append(res)
        prev_flac_end = res[4]  # flac_end of this chunk

    # Save CSV
    score_file = os.path.join(segment_dir, "chunk_confidence_scores.csv")
    with open(score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_index", "mp3_start_sec", "mp3_end_sec",
                         "flac_start_sec", "flac_end_sec", "confidence_score", "segment_file"])
        for row in results:
            idx, mp3_start, mp3_end, flac_start, flac_end, score, seg_file = row
            writer.writerow([idx, round(mp3_start/SR_TARGET,3),
                             round(mp3_end/SR_TARGET,3),
                             round(flac_start/SR_TARGET,3),
                             round(flac_end/SR_TARGET,3),
                             round(score,4),
                             seg_file])
    print(f"Confidence scores saved to: {score_file}")

     # Concatenate final FLAC
    concat_txt = os.path.join(segment_dir, "segments_list.txt")
    with open(concat_txt, "w") as f:
        for _, _, _, _, _, _, seg_file in results:
            f.write(f"file '{os.path.abspath(seg_file)}'\n")

    final_flac = os.path.join(os.path.dirname(mp3_file),
                              os.path.splitext(os.path.basename(mp3_file))[0] + "_aligned.flac")
    run_ffmpeg(
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c:a", "flac",
        final_flac
    )

    # Trim final FLAC to MP3 length (still FLAC for intermediate)
    mp3_duration = librosa.get_duration(filename=mp3_file)
    trimmed_flac = final_flac.replace("_aligned.flac", ".flac")
    run_ffmpeg(
        "-y", "-i", final_flac,
        "-t", f"{mp3_duration:.3f}",
        "-c:a", "flac",
        trimmed_flac
    )

    update_osu_audio_filenames(mp3_file)
    archive_dir = os.path.join(os.path.dirname(mp3_file), "archive")
    os.makedirs(archive_dir, exist_ok=True)

    archived_mp3_path = os.path.join(archive_dir, os.path.basename(mp3_file))
    # Optionally reset terminal (Linux/WSL)
    os.system("reset")
    print(f"Aligned and trimmed flac saved to: {trimmed_flac}")

    try:
        os.replace(mp3_file, archived_mp3_path)
        print(f"Moved MP3 to archive: {archived_mp3_path}")
    except Exception as e:
        print(f"Failed to move MP3 to archive: {e}")



if __name__ == "__main__":
    main()
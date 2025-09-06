import os
import sys
import subprocess

TK_AVAILABLE = False
if sys.platform != "darwin":
    try:
        import tkinter as tk
        from tkinter import filedialog
        TK_AVAILABLE = True
    except Exception:
        TK_AVAILABLE = False

def choose_open_file(title="Open", pattern="public.audio", filetypes=None):
    if filetypes is None:
        filetypes = [("Audio", "*.wav *.mp3 *.flac *.aiff *.aif *.ogg"), ("All files", "*.*")]
    
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose file with prompt "{title}" of type {{"{pattern}"}})'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            out = res.stdout.strip()
            return out if res.returncode == 0 and out else None
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.askopenfilename(title=title, filetypes=filetypes)
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None

def choose_save_file(title="Save As", default_name="mix.wav", filetypes=None):
    if filetypes is None:
        filetypes = [("WAV", "*.wav")]
    
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose file name with prompt "{title}" default name "{default_name}")'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            path = res.stdout.strip() if res.returncode == 0 else None
            if path and not os.path.splitext(path)[1]:
                path += ".wav"
            return path
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.asksaveasfilename(
                title=title, defaultextension=".wav", filetypes=filetypes, initialfile=default_name
            )
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None

def choose_folder(title="Choose Folder"):
    if sys.platform == "darwin":
        try:
            osa = f'POSIX path of (choose folder with prompt "{title}")'
            res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True)
            out = res.stdout.strip()
            return out if res.returncode == 0 and out else None
        except Exception:
            return None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw(); root.update()
            fp = filedialog.askdirectory(title=title)
            root.destroy()
            return fp if fp else None
        except Exception:
            return None
    return None

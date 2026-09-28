import os, tempfile, shutil, subprocess, sys
try:
    import pyttsx3
except Exception as e:
    print("pyttsx3 import failed:", e); sys.exit(1)

def try_direct(text, vol=1.0):
    try:
        engine = pyttsx3.init(driverName="sapi5" if os.name=="nt" else None)
        engine.setProperty("volume", vol)
        engine.say(text)
        engine.runAndWait()
        print("pyttsx3 direct OK")
        return True
    except Exception as e:
        print("pyttsx3 direct failed:", repr(e))
        return False

def try_wav_play(text, vol=1.0):
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="tts_")
        wav_path = os.path.join(tmp_dir, "out.wav")
        engine = pyttsx3.init(driverName="sapi5" if os.name=="nt" else None)
        engine.setProperty("volume", vol)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        print("WAV saved to", wav_path)
        if os.name == "nt":
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            print("Played WAV with winsound")
        else:
            # try ffplay / aplay fallback (may not exist)
            subprocess.run(["ffplay","-nodisp","-autoexit",wav_path], check=False)
            print("Tried playing WAV via ffplay")
        return True
    except Exception as e:
        print("WAV playback failed:", repr(e))
        return False
    finally:
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

def try_powershell(text, vol=100):
    if os.name != "nt":
        print("PowerShell SAPI only available on Windows")
        return False
    try:
        safe = text.replace('"', '\\"')
        cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Volume = {int(vol)}; $s.Speak(\"{safe}\")'
        r = subprocess.run(["powershell","-NoProfile","-Command",cmd], capture_output=True, text=True)
        print("PowerShell returncode:", r.returncode, "stderr:", r.stderr.strip())
        return r.returncode == 0
    except Exception as e:
        print("PowerShell SAPI failed:", repr(e))
        return False

if __name__ == "__main__":
    text = "This is a local python text to speech test at full volume."
    ok = try_direct(text)
    if not ok:
        print("direct failed, trying WAV playback")
        if not try_wav_play(text):
            print("WAV play failed, trying PowerShell SAPI")
            ok2 = try_powershell(text)
            print("PowerShell result:", ok2)
    else:
        print("Direct succeeded")

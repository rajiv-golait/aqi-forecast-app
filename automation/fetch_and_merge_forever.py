import subprocess
import time
from datetime import datetime, timedelta
import re

FETCH_SCRIPT = 'scripts/fetch_and_merge_all_data.py'
WAIT_SECONDS = 3600  # 1 hour

def run_fetch_and_merge():
    """Run the fetch and merge script and parse output for next allowed fetch time."""
    print(f"[{datetime.now()}] Running fetch and merge...")
    proc = subprocess.Popen(['python', FETCH_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    next_allowed = None
    for line in proc.stdout:
        print(line, end='')
        # Look for 'Next fetch allowed after ...' in output
        m = re.search(r'Next fetch allowed after ([\d\-: ]+)', line)
        if m:
            try:
                next_allowed = datetime.strptime(m.group(1).strip(), '%Y-%m-%d %H:%M')
            except Exception:
                pass
    proc.wait()
    return next_allowed

def main():
    while True:
        next_allowed = run_fetch_and_merge()
        now = datetime.now()
        if next_allowed and next_allowed > now:
            wait_seconds = (next_allowed - now).total_seconds()
            print(f"[{datetime.now()}] Waiting {int(wait_seconds)} seconds until next allowed fetch...")
            time.sleep(wait_seconds)
        else:
            print(f"[{datetime.now()}] Waiting {WAIT_SECONDS} seconds before next fetch...")
            time.sleep(WAIT_SECONDS)

if __name__ == "__main__":
    main() 
import logging
import json
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # attach extra if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: Path, level: int = logging.INFO):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # if already configured, keep existing handlers
    if root.handlers:
        for h in root.handlers:
            root.removeHandler(h)

    root.setLevel(level)

    fmt = JSONLineFormatter()

    # file handler with daily rotation
    file_path = log_dir / "diamond.log"
    fh = TimedRotatingFileHandler(str(file_path), when="midnight", backupCount=14, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # console handler (human readable)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    root.addHandler(ch)

    return root

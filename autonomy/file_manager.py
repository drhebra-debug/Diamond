from pathlib import Path

class FileManager:
    def __init__(self, root):
        self.root = Path(root)

    def read(self, path):
        p = self.root / path
        return p.read_text(encoding="utf-8")

    def write(self, path, content):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def list_files(self):
        return [str(p.relative_to(self.root))
                for p in self.root.rglob("*")
                if p.is_file()]

from pathlib import Path
import unittest


class RangeDownloaderTests(unittest.TestCase):
    def test_downloader_entrypoint_exists(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "tools" / "range_download.py").is_file())


if __name__ == "__main__":
    unittest.main()

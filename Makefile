install:
    python3 setup_env.py

export-public-mirror:
    python3 tools/public_mirror/export_public_mirror.py --output tmp/public-mirror --report tmp/public-mirror-report.json
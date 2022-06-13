from pathlib import Path

def get_repo_root() -> Path:
    return Path(__file__).parent.parent

def get_project11_root() -> Path:
    return Path(__file__).parent / "project11"


def get_project12_root() -> Path:
    return Path(__file__).parent / "project12"


def get_project2_root() -> Path:
    return Path(__file__).parent / "project2"


def get_project3_root() -> Path:
    return Path(__file__).parent / "project3"

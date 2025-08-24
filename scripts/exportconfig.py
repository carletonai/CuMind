#!./.venv/bin/python3.12
if __name__ == "__main__":
    from cumind.utils.config import Configuration

    Configuration()._to_json("configuration.json")

from cumind.config import Config


def main() -> None:
    config = Config()
    config.to_json("configuration.json")


if __name__ == "__main__":
    main()
